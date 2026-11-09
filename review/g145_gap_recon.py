"""G145 (10/31, ledger #106 gate 1, strategy D2): a causal masked reconstruction of the history grid (rows T-12 .. T-1 x links, R.npy channels v, q, observed; only cells <= T-1 of the window are seen).
Fold as production fold 2: fit on windows of 2030-06 .. 2030-11, read on 2030-12 .. 2031-02 (a window sample). Artificial mask: 15 % of the observed cells at random (fit: 10-20 %). Scored on the masked observed cells, L1 per channel.
Against what the inputs use today (LOCF = last observation of the link, the last_v / last_q rule; mean over rows for the summaries) and the two-sided linear interpolation in time within the window.
Model: inputs = linear fill of v, q + the observed flag; 3 residual 3 x 3 convolutions; output = correction of the fill, so a zero output is the linear fill.
usage: py g145_gap_recon.py dry | run"""
import os, sys, time
import numpy as np, pandas as pd, torch
torch.set_num_threads(3)
R = r"C:/Users/user/tfb/"; OUT = R + "work/t2/gap_recon/"; os.makedirs(OUT, exist_ok=True); sys.path.insert(0, R + "src")
from common import NN
MODE = sys.argv[1]; NTR, NTE, EP, BS = (300, 200, 1, 8) if MODE == "dry" else (12000, 3000, 3, 8); T00 = time.time()
d = "nn_data_afdem"; Rr = np.load(f"{NN}/{d}/R.npy", mmap_mode="r"); off = np.load(f"{NN}/{d}/off.npy"); date = np.load(f"{NN}/{d}/date.npy", allow_pickle=True); mon = np.array([str(x)[:7] for x in date]); rng0 = np.random.default_rng(0)
trk = rng0.choice(np.flatnonzero(mon < "2030-12"), NTR, replace=False); tek = rng0.choice(np.flatnonzero(mon >= "2030-12"), NTE, replace=False)


def grid(k):
    return np.array(Rr[off[k]:off[k + 1], :12], dtype=np.float32).transpose(1, 0, 2)                # (12, n, 3): rows T-12 .. T-1


def fills(g, seen):
    """LOCF (first rows: next observation) and two-sided linear interpolation in time, per link, from the cells in seen"""
    T, n = seen.shape; lo = np.zeros((T, n, 2), np.float32); li = np.zeros((T, n, 2), np.float32)
    for j in range(n):
        o = np.flatnonzero(seen[:, j])
        if len(o) == 0:
            continue
        for c in range(2):
            v = g[o, j, c]; li[:, j, c] = np.interp(np.arange(T), o, v); ix = np.searchsorted(o, np.arange(T), "right") - 1; lo[:, j, c] = v[np.maximum(ix, 0)]
    return lo, li


class Net(torch.nn.Module):
    def __init__(s, h=32):
        super().__init__(); s.i = torch.nn.Conv2d(3, h, 3, padding=1); s.c = torch.nn.ModuleList([torch.nn.Conv2d(h, h, 3, padding=1) for _ in range(6)]); s.o = torch.nn.Conv2d(h, 2, 1); torch.nn.init.zeros_(s.o.weight); torch.nn.init.zeros_(s.o.bias)

    def forward(s, x):
        z = s.i(x)
        for a, b in zip(s.c[::2], s.c[1::2]):
            z = z + b(torch.relu(a(torch.relu(z))))
        return x[:, :2] + s.o(torch.relu(z))


def sample(k, rng, p):
    g = grid(k); obs = g[..., 2] > 0; mask = obs & (rng.random(obs.shape) < p); seen = obs & ~mask; lo, li = fills(g, seen); x = np.concatenate([li, seen[..., None].astype(np.float32)], -1).transpose(2, 0, 1)
    return torch.from_numpy(x)[None], torch.from_numpy(g[..., :2].transpose(2, 0, 1))[None], torch.from_numpy(mask)[None, None], lo, li, g, mask, obs


net = Net(); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4); print(f"history cells observed in a sample: {np.mean([(grid(k)[..., 2] > 0).mean() for k in tek[:200]]):.4f}", flush=True)
for ep in range(EP):
    rng = np.random.default_rng(ep); order = rng.permutation(trk); tot = 0.0; nb = 0
    for b0, k in enumerate(order):
        x, y, m, *_ = sample(k, rng, rng.uniform(0.1, 0.2)); l = (torch.abs(net(x) - y) * m.float()).sum() / (2 * m.sum().clamp(min=1)); l.backward(); tot += float(l); nb += 1
        if (b0 + 1) % BS == 0:
            opt.step(); opt.zero_grad(set_to_none=True)
    print(f"epoch {ep} L1 {tot / nb:.4f} ({time.time() - T00:.0f}s)", flush=True)
net.eval(); rng = np.random.default_rng(99); rows = []
with torch.no_grad():
    for k in tek:
        x, y, m, lo, li, g, mask, obs = sample(k, rng, 0.15); p = net(x)[0].numpy().transpose(1, 2, 0); last = np.zeros_like(mask); last[-1] = True
        for nm, mk in (("all", mask), ("last_row", mask & last)):
            if mk.sum():
                rows.append(dict(k=k, month=mon[k], set=nm, n=int(mk.sum()), **{f"{a}_{c}": float(np.abs(f[..., i] - g[..., i])[mk].sum()) for a, f in (("locf", lo), ("lin", li), ("net", p)) for i, c in enumerate("vq")}))
D = pd.DataFrame(rows); D.to_csv(OUT + "g145.csv", index=False)
for nm, g_ in D.groupby("set"):
    s = g_.drop(columns=["k", "month", "set"]).sum(); print(nm, "cells", int(s.n), " ".join(f"{a}_{c} {s[f'{a}_{c}'] / s.n:.4f}" for a in ("locf", "lin", "net") for c in "vq"), "| net / min(locf, lin):", " ".join(f"{c} {s[f'net_{c}'] / min(s[f'locf_{c}'], s[f'lin_{c}']):.3f}" for c in "vq"), flush=True)
    rb = np.random.default_rng(0); idx = rb.integers(0, len(g_), (1000, len(g_)))
    for c in "vq":
        a = g_[f"net_{c}"].to_numpy(); b = np.minimum(g_[f"lin_{c}"].sum(), g_[f"locf_{c}"].sum()); col = "lin" if g_[f"lin_{c}"].sum() <= g_[f"locf_{c}"].sum() else "locf"; bb = g_[f"{col}_{c}"].to_numpy(); r = a[idx].sum(1) / bb[idx].sum(1); print(f"  {c}: ratio {a.sum() / b:.3f} (SE {r.std():.3f})", flush=True)
print(f"total {time.time() - T00:.0f}s", flush=True)
