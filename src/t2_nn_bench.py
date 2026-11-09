"""Step-time benchmark for the T2 corridor NN on synthetic data (same shapes as nn_data): which speedups are real.

usage: python t2_nn_bench.py [variant ...]   variants: base nosync fused b512 b1024 bucket compile all
"""
import os, sys, time, numpy as np, torch, torch.nn as nn

dev = torch.device("cuda"); F = 74; LMAX = 352; LMEAN = 211; NW = 20000
rng = np.random.default_rng(0)
lens_np = rng.choice([150, 180, 200, 211, 230, 260, 352], NW).astype(np.int64)
OFF = np.concatenate([[0], np.cumsum(lens_np)])
Xg = torch.randn(int(OFF[-1]), F, device=dev); Mg = torch.rand(int(OFF[-1]), F, device=dev) < 0.05
Yg = torch.rand(int(OFF[-1]), 6, device=dev) < 0.1; Eg = torch.ones(int(OFF[-1]), 6, dtype=torch.bool, device=dev)
starts = torch.from_numpy(OFF[:-1]).to(dev); lens = torch.from_numpy(lens_np).to(dev)


class DilConv(nn.Module):
    def __init__(self, h, d, k=5):
        super().__init__(); self.d, self.k = d, k; self.lin = nn.Linear(h * k, h)

    def forward(self, z):
        L = z.shape[1]; pad = (self.k // 2) * self.d
        zp = nn.functional.pad(z, (0, 0, pad, pad))
        return self.lin(torch.cat([zp[:, i * self.d:i * self.d + L] for i in range(self.k)], dim=-1))


class Net(nn.Module):
    def __init__(self, f, h):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(2 * f, h), nn.GELU(), nn.Linear(h, h))
        ds = (1, 2, 4, 8, 16, 1, 2, 4, 8, 16)
        self.convs = nn.ModuleList([DilConv(h, d) for d in ds])
        self.mlps = nn.ModuleList([nn.Sequential(nn.GELU(), nn.Linear(h, h)) for _ in ds])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in ds])
        self.out = nn.Linear(h, 6)

    def forward(self, x, m):
        z = self.inp(x); mm = m[:, :, None]
        for c, f, n in zip(self.convs, self.mlps, self.norms):
            z = z + f(c(n(z) * mm))
        return self.out(z)


def gather(k, L):
    ar = torch.arange(L, device=dev)
    idx = starts[k][:, None] + ar[None, :]; valid = ar[None, :] < lens[k][:, None]
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    x = Xg[idx]; nm = Mg[idx]
    x = torch.where(nm, torch.zeros_like(x), x.clamp(-10, 10))
    xb = torch.cat([x, nm.float()], dim=-1) * valid[:, :, None]
    return xb, Yg[idx].float() * valid[:, :, None], Eg[idx].float() * valid[:, :, None], valid.float()


def bench(name, B=256, sync=True, fused=False, L=LMAX, comp=False, steps=60):
    torch.manual_seed(0); net = Net(F, 128).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4, fused=fused)
    fwd = torch.compile(net) if comp else net
    perm_h = rng.permutation(NW); perm_d = torch.randperm(NW, device=dev)
    acc = torch.zeros((), device=dev)

    def step(i):
        nonlocal acc
        k = torch.as_tensor(perm_h[i * B:(i + 1) * B], device=dev) if sync else perm_d[i * B:(i + 1) * B]
        xb, yb, eb, mb = gather(k, L)
        wb = 100.0 / torch.clamp((yb * eb).sum(dim=(1, 2)), min=3)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = fwd(xb, mb)
        l = nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * eb
        loss = (l.sum(dim=(1, 2)) * wb).sum() / (eb.sum() + 1)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if sync:
            loss.item()
        else:
            acc = acc + loss.detach()

    nb = NW // B
    t = time.time()
    for i in range(min(15, nb)):
        step(i % nb)
    torch.cuda.synchronize(); tw = time.time() - t
    t = time.time()
    for i in range(steps):
        step(i % nb)
    torch.cuda.synchronize(); dt = (time.time() - t) / steps
    print(f"{name:10s} B={B:4d} L={L} sync={sync} fused={fused} compile={comp}: {dt*1e3:7.1f} ms/step  "
          f"{dt/B*1e6:6.1f} us/window  (warmup {tw:.1f}s)", flush=True)


V = {"base": dict(), "nosync": dict(sync=False), "fused": dict(sync=False, fused=True),
     "b512": dict(sync=False, fused=True, B=512), "b1024": dict(sync=False, fused=True, B=1024),
     "bucket": dict(sync=False, fused=True, L=LMEAN), "compile": dict(sync=False, fused=True, comp=True),
     "bcomp": dict(sync=False, fused=True, L=LMEAN, comp=True)}
for v in (sys.argv[1:] if len(sys.argv) > 1 and sys.argv[1] != "all" else list(V)):
    try:
        bench(v, **V[v])
    except Exception as e:
        print(v, "FAILED", type(e).__name__, str(e)[:200], flush=True)
