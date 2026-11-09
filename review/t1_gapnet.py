"""P9 (10/31, ledger #94): a small space-time network for the long all-link blanks of Task 1 (the T2 window gaps, rows t+1..t+18), from the
observations around the gap only (rows t-12..t+30). Output = residual to the per-link linear interpolation across the gap. Linear-only dilated
blocks along time and links (no Conv: no MIOpen kernels on ROCm). Trained on the training months with the window list of gapml, February held out.
Checkpoint every epoch, restart-safe.
  data  (Windows, g134 data) writes C:/tfb_data/tfb_nn/gapnet/<panel>.npz and windows.npz
usage (WSL): python t1_gapnet.py <seed> [epochs=10]"""
import os, sys, time, pickle, numpy as np, torch, torch.nn as nn
SEED = int(sys.argv[1]); EP = int(sys.argv[2]) if len(sys.argv) > 2 else 10; ROOT = os.environ.get("GAP_ROOT", "/mnt/c/tfb_data/tfb_nn/gapnet/")
G, PRE, POST = 18, 12, 12; R = PRE + 1 + G + POST; BS, NE, LR = 12, int(os.environ.get("GAP_NE", "6000")), 1e-3
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
W = np.load(ROOT + "windows.npz", allow_pickle=True); PANELS = [str(p) for p in W["panels"]]
D = {}
for p in PANELS:
    z = np.load(ROOT + p + ".npz")
    D[p] = {k: torch.from_numpy(z[k]).to(dev) for k in ("v", "q", "blank", "fin")}; D[p]["n"] = D[p]["v"].shape[1]
TR = {p: W["t_" + p] for p in PANELS}; TE = {p: W["te_" + p] for p in PANELS}
print("windows", {p: (len(TR[p]), len(TE[p])) for p in PANELS}, flush=True)


class DB(nn.Module):
    """x + mlp(concat of k neighbours at dilation d along dim -2)"""
    def __init__(self, c, d, k):
        super().__init__(); self.d, self.k = d, k; self.n = nn.LayerNorm(c); self.l = nn.Linear(c * k, c); self.a = nn.Sequential(nn.GELU(), nn.Linear(c, c))

    def forward(self, z):
        L = z.shape[-2]; pad = (self.k // 2) * self.d; y = nn.functional.pad(self.n(z), (0, 0, pad, pad))
        return z + self.a(self.l(torch.cat([y[..., i * self.d:i * self.d + L, :] for i in range(self.k)], dim=-1)))


class GapNet(nn.Module):
    def __init__(self, f=7, c=64):
        super().__init__(); self.i = nn.Sequential(nn.Linear(f, c), nn.GELU(), nn.Linear(c, c))
        self.t = nn.ModuleList([DB(c, d, 3) for d in (1, 2, 4, 8)]); self.s = nn.ModuleList([DB(c, d, 5) for d in (1, 2, 4, 8)]); self.o = nn.Linear(c, 2)

    def forward(self, x):                                       # x (B, R, n, f)
        z = self.i(x)
        for t, s in zip(self.t, self.s):
            z = t(z.transpose(1, 2)).transpose(1, 2)            # time: dim -2 = rows
            z = s(z)                                            # links: dim -2 = links
        return self.o(z)


def base_fill(a, vis):
    """(B, R, n): the mean of the last 3 visible rows before the gap and the first 3 after it, linear across the gap (one side missing: the other; none: 1)"""
    def side(rows):
        s = (a[:, rows] * vis[:, rows]).sum(1); c = vis[:, rows].sum(1); return s / c.clamp(min=1), c > 0
    pre, ok0 = side(list(range(PRE - 3, PRE + 1))); post, ok1 = side(list(range(PRE + 1 + G, PRE + 4 + G)))
    pre = torch.where(ok0, pre, torch.where(ok1, post, torch.ones_like(pre))); post = torch.where(ok1, post, pre)
    fr = ((torch.arange(G, device=dev, dtype=torch.float32) + 1) / (G + 1))[None, :, None]
    return pre[:, None, :] * (1 - fr) + post[:, None, :] * fr


def batch(p, ts):
    d = D[p]; t = torch.as_tensor(ts, device=dev)[:, None] + torch.arange(-PRE, G + POST + 1, device=dev)[None, :]
    v = d["v"][t].float(); q = d["q"][t].float(); fin = d["fin"][t]; bl = d["blank"][t].clone(); bl[:, :PRE] = False            # the window history is published unmasked
    gap = torch.zeros_like(fin); gap[:, PRE + 1:PRE + 1 + G] = True; vis = fin & ~bl & ~gap; vf = vis.float()
    bv, bq = base_fill(torch.nan_to_num(v), vf), base_fill(torch.nan_to_num(q), vf); B_v = torch.zeros_like(v); B_q = torch.zeros_like(q); B_v[:, PRE + 1:PRE + 1 + G] = bv; B_q[:, PRE + 1:PRE + 1 + G] = bq
    rp = ((torch.arange(R, device=dev, dtype=torch.float32) - PRE) / 30.0)[None, :, None].expand_as(v)
    x = torch.stack([torch.nan_to_num(v) * vf, torch.nan_to_num(q) * vf, vf, gap.float(), B_v, B_q, rp], dim=-1)
    return x, v[:, PRE + 1:PRE + 1 + G], q[:, PRE + 1:PRE + 1 + G], fin[:, PRE + 1:PRE + 1 + G], bv, bq


net = GapNet().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-3); ck = ROOT + f"ck_s{SEED}.pt"; start = 0
sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EP * (NE // BS))
if os.path.exists(ck):
    c = torch.load(ck, map_location=dev, weights_only=False); net.load_state_dict(c["net"]); opt.load_state_dict(c["opt"]); sched.load_state_dict(c["sched"]); start = c["ep"] + 1; rng.bit_generator.state = c["rng"]; print("resumed at epoch", start, flush=True)
t0 = time.time(); ps = np.array([len(TR[p]) for p in PANELS], float); ps /= ps.sum()
for ep in range(start, EP):
    net.train(); tot = 0.0
    for b in range(NE // BS):
        p = PANELS[rng.choice(len(PANELS), p=ps)]; ts = rng.choice(TR[p], BS, replace=False); x, yv, yq, ok, bv, bq = batch(p, ts); o = net(x)[:, PRE + 1:PRE + 1 + G]
        rv = o[..., 0] - torch.nan_to_num(yv - bv); rq = o[..., 1] - torch.nan_to_num(yq - bq); wt = ok.float() * (1.0 + 4.0 * ((torch.nan_to_num(yv, nan=9.0) <= 1.0) | (bv <= 1.0)).float())
        loss = ((rv ** 2 + rq ** 2) * wt).sum() / wt.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step(); tot += float(loss.detach())
    print(f"seed {SEED} epoch {ep} loss {tot / (NE // BS):.5f} ({time.time() - t0:.0f}s)", flush=True)
    torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), ep=ep, rng=rng.bit_generator.state), ck)
net.eval(); out = {}
with torch.no_grad():
    for p in PANELS:
        for t in TE[p]:
            x, _, _, _, bv, bq = batch(p, [int(t)]); o = net(x)[:, PRE + 1:PRE + 1 + G]
            out[(p, int(t))] = ((bv + o[..., 0])[0].cpu().numpy(), (bq + o[..., 1])[0].cpu().numpy())
pickle.dump(out, open(ROOT + f"pred_s{SEED}.pkl", "wb")); print("ALLDONE", len(out), f"{time.time() - t0:.0f}s", flush=True)
