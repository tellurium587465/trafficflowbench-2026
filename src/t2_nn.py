"""Task 2 corridor-wide neural net (runs in WSL on the AMD GPU via PyTorch/ROCm).

Input per window: (n_links, F) link features from the LightGBM feature cache; NaNs are zero-filled with a
missing-indicator channel. Model: per-link MLP -> dilated residual 1D convolutions along the corridor
(receptive field ~120 links) -> 6 logits per link (T+5..T+30). Month-grouped 3-fold CV writes OOF
probabilities keyed by the cache window index (same format as the LightGBM OOF pickles).

usage: python t2_nn.py <tag> [epochs]
"""
import sys, time, pickle, numpy as np, torch, torch.nn as nn

TAG = sys.argv[1]; EPOCHS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
D = np.load("/mnt/c/Users/user/tfb/work/t2/nn_data.npz", allow_pickle=True)
X = D["X"]; Y = D["Y"]; EL = D["EL"]; OFF = D["off"]; names = list(D["names"])
cond = D["cond"]; date = D["date"]; is_sel = D["is_sel"]; NW = len(OFF) - 1
for c in ("prior_on", "prior_og"):              # fold-unsafe in this export -> drop
    X[:, names.index(c)] = 0.0
FOLDS = [("2030-06", "2030-07", "2030-08"), ("2030-09", "2030-10", "2030-11"), ("2030-12", "2031-01", "2031-02")]
dev = torch.device("cuda")
mon = np.array([d[:7] for d in date])


class DilConv(nn.Module):
    """Dilated 1D conv along links implemented as shifted copies + one GEMM (avoids slow MIOpen conv paths)."""
    def __init__(self, h, d, k=5):
        super().__init__()
        self.d, self.k = d, k
        self.lin = nn.Linear(h * k, h)

    def forward(self, z):                       # z (B, L, H)
        L = z.shape[1]; half = self.k // 2; pad = half * self.d
        zp = nn.functional.pad(z, (0, 0, pad, pad))
        taps = [zp[:, i * self.d:i * self.d + L] for i in range(self.k)]
        return self.lin(torch.cat(taps, dim=-1))


class Net(nn.Module):
    def __init__(self, f, h=int(__import__("os").environ.get("NN_H", "96"))):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * f, h), nn.GELU(), nn.Linear(h, h))
        ds = (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
        self.convs = nn.ModuleList([DilConv(h, d) for d in ds])
        self.mlps = nn.ModuleList([nn.Sequential(nn.GELU(), nn.Linear(h, h)) for _ in ds])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in ds])
        self.out = nn.Linear(h, 6)

    def forward(self, x, m):                    # x (B, L, 2F), m (B, L)
        z = self.inp(x)
        mm = m[:, :, None]
        for c, f, n in zip(self.convs, self.mlps, self.norms):
            z = z + f(c(n(z) * mm))
        return self.out(z)                      # B, L, 6


def stats(idx):
    rows = np.concatenate([np.arange(OFF[k], OFF[k + 1]) for k in idx[::7]])
    s = X[rows]; mu = np.nanmean(s, 0); sd = np.nanstd(s, 0) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


LMAX = int(max(OFF[k + 1] - OFF[k] for k in range(NW)))


def batch(ks, mu, sd, B=None):
    L = LMAX; F = X.shape[1]; B = B or len(ks)   # fixed shape -> MIOpen tunes once
    xb = np.zeros((B, L, 2 * F), np.float32); yb = np.zeros((B, L, 6), np.float32)
    eb = np.zeros((B, L, 6), np.float32); mb = np.zeros((B, L), np.float32); wb = np.zeros(B, np.float32)
    for i, k in enumerate(ks):
        a, b = OFF[k], OFF[k + 1]; n = b - a
        v = (X[a:b] - mu) / sd; nan = ~np.isfinite(v)
        xb[i, :n, :F] = np.where(nan, 0, np.clip(v, -10, 10)); xb[i, :n, F:] = nan
        yb[i, :n] = Y[a:b]; eb[i, :n] = EL[a:b]; mb[i, :n] = 1
        wb[i] = 100.0 / max((Y[a:b] & EL[a:b]).sum(), 3)
    t = lambda z: torch.from_numpy(z).to(dev)
    return t(xb), t(yb), t(eb), t(mb), t(wb)


FINAL = len(sys.argv) > 3 and sys.argv[3] == "final"
oof = {}
t0 = time.time()
if FINAL:
    FOLDS = [("NONE",)]
for months in FOLDS:
    tr = np.flatnonzero(~np.isin(mon, months) & ~is_sel)
    te = np.flatnonzero(np.isin(mon, months) & (D["is_eval"] | is_sel))
    mu, sd = stats(tr); mu = np.nan_to_num(mu); sd = np.where(np.isfinite(sd), sd, 1.0).astype(np.float32)
    net = Net(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = EPOCHS * (len(tr) // 64); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    rng = np.random.default_rng(0)
    for ep in range(EPOCHS):
        perm = rng.permutation(tr); tot = 0.0
        for i in range(0, len(perm) - 63, 64):
            xb, yb, eb, mb, wb = batch(perm[i:i + 64], mu, sd)
            logit = net(xb, mb)
            l = nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="none") * eb
            loss = (l.sum(dim=(1, 2)) * wb).sum() / (eb.sum() + 1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f"fold {months[0]} epoch {ep} loss {tot:.3f} ({time.time()-t0:.0f}s)", flush=True)
    net.eval()
    with torch.no_grad():
        for i in range(0, len(te), 128):
            ks = te[i:i + 128]; xb, yb, eb, mb, wb = batch(ks, mu, sd, B=128)
            p = torch.sigmoid(net(xb, mb)).cpu().numpy()
            for j, k in enumerate(ks):
                n = OFF[k + 1] - OFF[k]; pk = p[j, :n].T.copy()
                if cond[k] == "queue_onset":
                    pk[:5] = 0.0
                oof[int(k)] = pk
if FINAL:
    Dt = np.load("/mnt/c/Users/user/tfb/work/t2/nn_test.npz", allow_pickle=True)
    Xt, offt, wid, condt = Dt["X"], Dt["off"], Dt["window_id"], Dt["cond"]
    Xt[:, names.index("prior_on")] = 0.0; Xt[:, names.index("prior_og")] = 0.0
    probs = {}
    with torch.no_grad():
        for i in range(len(wid)):
            a, b = offt[i], offt[i + 1]; n = b - a; F = Xt.shape[1]
            xb = np.zeros((1, LMAX, 2 * F), np.float32); mb = np.zeros((1, LMAX), np.float32)
            v = (Xt[a:b] - mu) / sd; nan = ~np.isfinite(v)
            xb[0, :n, :F] = np.where(nan, 0, np.clip(v, -10, 10)); xb[0, :n, F:] = nan; mb[0, :n] = 1
            p = torch.sigmoid(net(torch.from_numpy(xb).to(dev), torch.from_numpy(mb).to(dev))).cpu().numpy()[0, :n].T.copy()
            if condt[i] == "queue_onset":
                p[:5] = 0.0
            probs[str(wid[i])] = p
    pickle.dump(probs, open(f"/mnt/c/Users/user/tfb/work/t2/probs_{TAG}.pkl", "wb"))
else:
    pickle.dump(oof, open(f"/mnt/c/Users/user/tfb/work/t2/oof_{TAG}.pkl", "wb"))
print("done", len(oof), f"{time.time()-t0:.0f}s")
