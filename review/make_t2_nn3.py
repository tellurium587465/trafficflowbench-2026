"""P2(c) (10/31, ledger #86): review/t2_nn3.py = src/t2_nn2.py + a temporal encoder over the raw history (rows T-12..T-1 x v/vcut,
q/cap, observed; nn_data_*/R.npy from t2_raw_hist.py). src/ is not changed. Each replacement must match exactly once.
The encoder uses the same Linear-only dilated block as the spatial stack (no Conv1d: no MIOpen kernels on ROCm)."""
R = r"C:/Users/user/tfb/"
s = open(R + "src/t2_nn2.py", encoding="utf-8", newline="").read()
REP = [
 ('class Net(nn.Module):\n    def __init__(self, fin, h, depth=10, k=5, drop=0.0, glob=0, attn=0, heads=4, aux=0, dom=0):\n        super().__init__()\n',
  'class TEnc(nn.Module):\n'
  '    """10/31 P2(c): the raw history of one link -> h: rows T-12..T-1 x 3 channels, time-wise dilated blocks, last + mean"""\n'
  '    def __init__(self, h, c=48):\n'
  '        super().__init__()\n'
  '        self.i = nn.Linear(3, c); self.b = nn.ModuleList([DilConv(c, d, 3) for d in (1, 2, 4)]); self.n = nn.ModuleList([nn.LayerNorm(c) for _ in range(3)])\n'
  '        self.o = nn.Sequential(nn.Linear(2 * c, h), nn.GELU(), nn.Linear(h, h))\n\n'
  '    def forward(self, r):\n'
  '        B, L, _ = r.shape; z = self.i(r.view(B * L, 12, 3))\n'
  '        for b, n in zip(self.b, self.n):\n'
  '            z = z + nn.functional.gelu(b(n(z)))\n'
  '        return self.o(torch.cat([z[:, -1], z.mean(1)], dim=-1)).view(B, L, -1)\n\n\n'
  'class Net(nn.Module):\n    def __init__(self, fin, h, depth=10, k=5, drop=0.0, glob=0, attn=0, heads=4, aux=0, dom=0, raw=0):\n        super().__init__()\n'
  '        self.fin = fin; self.tenc = TEnc(h) if raw else None           # 10/31 P2(c): the raw history is appended to x\n'),
 ('        z = self.inp(x); mm = m[:, :, None]\n',
  '        z = self.inp(x[..., :self.fin]); mm = m[:, :, None]\n'
  '        if self.tenc is not None:\n'
  '            z = z + self.tenc(x[..., self.fin:])\n'),
 ('Yg = torch.from_numpy(D["Y"]).to(dev); Eg = torch.from_numpy(D["EL"]).to(dev)\n',
  'Yg = torch.from_numpy(D["Y"]).to(dev); Eg = torch.from_numpy(D["EL"]).to(dev)\n'
  'RAW = int(os.environ.get("NN_RAW", "0")); Rg = None                 # 10/31 P2(c): rows T-12..T-1 of R.npy (row T is empty under NO_T)\n'
  'if RAW:\n'
  '    Rg = torch.from_numpy(np.load(os.environ["NN_RAWFILE"], mmap_mode="r")[:, :12].astype(np.float16)).to(dev)\n'
  '    assert Rg.shape[0] == Xg.shape[0], "raw history rows differ from X"\n'),
 ('    xb = encode(xr, nm, mu, sd, valid) * valid[:, :, None]\n    yb = Yg[idx].float() * valid[:, :, None]',
  '    xb = encode(xr, nm, mu, sd, valid) * valid[:, :, None]\n'
  '    if RAW:\n'
  '        xb = torch.cat([xb, Rg[idx].float().flatten(2) * valid[:, :, None]], dim=-1)\n'
  '    yb = Yg[idx].float() * valid[:, :, None]'),
 ('    for c in [c for c in ZERO.split("+") if c]:\n        COLMASK[names.index(c)] = 0.0\n    grp = mon\n',
  '    for c in [c for c in ZERO.split("+") if c]:\n        if c == "ALL":\n            COLMASK[:] = 0.0                                    # 10/31 X3a: the raw history alone\n            continue\n        COLMASK[names.index(c)] = 0.0\n    grp = mon\n'),
 ('net = Net(enc_width(), H, DEPTH, K, DROP, GLOBAL, ATTN, HEADS, AUX > 0, DOM).to(dev)', 'net = Net(enc_width(), H, DEPTH, K, DROP, GLOBAL, ATTN, HEADS, AUX > 0, DOM, RAW).to(dev)'),
 ('    if FINAL:\n', '    assert not (RAW and FINAL), "P2(c): FINAL with the raw history needs the test R.npy path (added only if the probe passes)"\n    if FINAL:\n'),
 # P5 (10/31): Lovasz hinge on the ongoing windows next to the BCE (LOV=lambda in the config line)
 ('    return (ce * (ts > 0)).sum()\n',
  '    return (ce * (ts > 0)).sum()\n\n\n'
  'LOV = 0.0\n\n\n'
  'def lovasz_ongoing(logit, yb, eb, kidx):\n'
  '    """10/31 P5: Lovasz hinge (Berman et al. 2018) per ongoing window over its eligible cells, summed over the batch; vectorised:\n'
  '    masked cells sort last and carry no weight"""\n'
  '    og = ~ONS[kidx]\n'
  '    if not bool(og.any()):\n'
  '        return logit.new_zeros(())\n'
  '    z = logit[og].float().flatten(1); y = yb[og].flatten(1); e = eb[og].flatten(1) > 0\n'
  '    err = (1.0 - z * (2.0 * y - 1.0)).masked_fill(~e, -1e9); es, ix = torch.sort(err, dim=1, descending=True)\n'
  '    gs = torch.gather(y * e, 1, ix); vs = torch.gather(e.float(), 1, ix); gts = gs.sum(1, keepdim=True)\n'
  '    jac = 1.0 - (gts - gs.cumsum(1)) / (gts + ((1.0 - gs) * vs).cumsum(1)).clamp(min=1.0)\n'
  '    grad = torch.cat([jac[:, :1], jac[:, 1:] - jac[:, :-1]], dim=1)\n'
  '    return (es.clamp(min=0.0) * vs * grad.detach()).sum()\n'),
 ('                if LISTW > 0:\n                    loss = loss + LISTW * 100.0 * listwise_onset(logit, yb, eb, ks) / den\n                if AUX > 0:\n',
  '                if LISTW > 0:\n                    loss = loss + LISTW * 100.0 * listwise_onset(logit, yb, eb, ks) / den\n'
  '                if LOV > 0:\n                    loss = loss + LOV * 100.0 * lovasz_ongoing(logit, yb, eb, ks) / den\n'
  '                if AUX > 0:\n'),
 ('    KNOWN = {"H", "B",', '    KNOWN = {"LOV", "H", "B",'),
 ('    g = lambda k, d: type(d)(kv.get(k, d))\n', '    g = lambda k, d: type(d)(kv.get(k, d))\n    global LOV; LOV = float(kv.get("LOV", 0))\n'),
 # P6 (10/31): one uniform factor per window on the flow columns (JMODE=w, TJIT=20 -> 1 +- 0.2); TJIT=0 never calls jitter
 ('    fac = 1.0 + jit * torch.randn(xr.shape[:2], device=dev, generator=gen)\n',
  '    fac = 1.0 + jit * torch.randn(xr.shape[:2], device=dev, generator=gen)\n'
  '    if JWIN:                                                    # 10/31 P6: one uniform factor per window\n'
  '        fac = (1.0 + jit * (2.0 * torch.rand(xr.shape[0], 1, device=dev, generator=gen) - 1.0)).expand(xr.shape[:2])\n'),
 ('JCOLS, JNB = VCOL, NBV\n', 'JCOLS, JNB = VCOL, NBV\nJWIN = False\n'),
 ('    global JCOLS, JNB\n    JCOLS, JNB = {"v": (VCOL, NBV), "q": (QCOL, NBQ), "vq": (VCOL + QCOL, NBV + NBQ)}[JMODE]\n',
  '    global JCOLS, JNB, JWIN\n    JCOLS, JNB = {"v": (VCOL, NBV), "q": (QCOL, NBQ), "w": (QCOL, NBQ), "vq": (VCOL + QCOL, NBV + NBQ)}[JMODE]; JWIN = JMODE == "w"\n'),
 # P8 (10/31): the mean of the (EMA) weights over the last TAIL epochs (TAIL=10 in the config line)
 ('            print(f"{TAG} fold {months[0]} epoch {ep} loss {tot.item():.3f} ({time.time()-t0:.0f}s)", flush=True)\n        diverged |=',
  '            print(f"{TAG} fold {months[0]} epoch {ep} loss {tot.item():.3f} ({time.time()-t0:.0f}s)", flush=True)\n'
  '            if TAILN > 0 and ep >= EPOCHS - TAILN:                  # 10/31 P8\n'
  '                with torch.no_grad():\n'
  '                    cur = ema_s if EMA > 0 else [q.detach() for q in net.parameters()]; tail_c[0] += 1\n'
  '                    tail_c[1] = [q.clone() for q in cur] if tail_c[1] is None else [a + (b - a) / tail_c[0] for a, b in zip(tail_c[1], cur)]\n'
  '        diverged |='),
 ('        net.eval()\n        if SAVEF and not FINAL:',
  '        if TAILN > 0:\n'
  '            with torch.no_grad():\n'
  '                torch._foreach_copy_([q.data for q in net.parameters()], tail_c[1])\n'
  '            print(f"{TAG}: mean of the last {tail_c[0]} epochs", flush=True)\n'
  '        net.eval()\n        if SAVEF and not FINAL:'),
 ('        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb)\n', '        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb); tail_c = [0, None]\n'),
 ('LOV = 0.0\n\n\n', 'LOV = 0.0; TAILN = 0\n\n\n'),
 ('global LOV; LOV = float(kv.get("LOV", 0))\n', 'global LOV, TAILN; LOV = float(kv.get("LOV", 0)); TAILN = int(kv.get("TAIL", 0))\n'),
 ('    KNOWN = {"LOV", "H", "B",', '    KNOWN = {"LOV", "TAIL", "H", "B",'),
 # P7 (10/31): the auxiliary head regresses the first difference of v/vf between consecutive horizon rows, x10 (AUX=0.3,AUXD=1)
 ('    ya = AUXG[idx].float(); ok = torch.isfinite(ya) & (valid[:, :, None] > 0)\n',
  '    ya = AUXG[idx].float(); ok = torch.isfinite(ya) & (valid[:, :, None] > 0)\n'
  '    if AUXD:                                                    # 10/31 P7\n'
  '        ya = torch.cat([ya[..., :1], 10.0 * (ya[..., 1:] - ya[..., :-1])], dim=-1); ok = torch.cat([torch.zeros_like(ok[..., :1]), ok[..., 1:] & ok[..., :-1]], dim=-1)\n'),
 ('LOV = 0.0; TAILN = 0\n', 'LOV = 0.0; TAILN = 0; AUXD = 0\n'),
 ('global LOV, TAILN; LOV =', 'global LOV, TAILN, AUXD; AUXD = int(kv.get("AUXD", 0)); LOV ='),
 ('    KNOWN = {"LOV", "TAIL", "H", "B",', '    KNOWN = {"LOV", "TAIL", "AUXD", "H", "B",'),
]
for a, b in REP:
    n = s.count(a); assert n == 1, (n, a[:70]); s = s.replace(a, b)
open(R + "review/t2_nn3.py", "w", encoding="utf-8", newline="").write(s); print("ok", len(s))
