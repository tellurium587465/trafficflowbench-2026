"""Task 2 corridor NN, GPU-resident version.

All link-feature rows live on the GPU (fp16 values + NaN mask); per-fold normalization, batch gathering and
padding happen on the GPU; bf16 autocast; configurable batch size. Same model/outputs as t2_nn.py.

usage: python t2_nn2.py <tag> <epochs> [final] ; env NN_H, NN_B, NN_LR, NN_SEED
"""
import os, sys, time, pickle, numpy as np, torch, torch.nn as nn
RW_ = 1.0

TAG = sys.argv[1]; EPOCHS = int(sys.argv[2]); FINAL = len(sys.argv) > 3 and sys.argv[3] == "final"
H = int(os.environ.get("NN_H", "128")); B = int(os.environ.get("NN_B", "256"))
LR = float(os.environ.get("NN_LR", "4e-3")); SEED = int(os.environ.get("NN_SEED", "0"))
torch.manual_seed(SEED)
ROOT_E, ROOT_C = ("E:/", "C:/") if os.name == "nt" else ("/mnt/e/", "/mnt/c/")   # Windows CPU runs (10/13)
ROOT_E = os.environ.get("TFB_ROOT_E", "C:/tfb_data/" if os.name == "nt" else "/mnt/c/tfb_data/")   # 10/22: <root>tfb_nn/ off E:
dev = torch.device(os.environ.get("NN_DEV", "cuda"))        # NN_DEV=cpu: same code on the CPU (10/13, GPU lent to RSNA)
AMP = os.environ.get("NN_AMP", "bf16")                    # 10/22: bf16 (local, default) | fp16 (Kaggle T4/P100: no bf16) | off
AC = dict(device_type=dev.type, dtype=torch.float16 if AMP == "fp16" else torch.bfloat16, enabled=dev.type == "cuda" and AMP != "off")
SCALER = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda" and AMP == "fp16")   # no-op unless fp16
OUT = os.environ.get("TFB_OUT", f"{ROOT_C}Users/user/tfb/work/t2/")                 # probs / oof pickles
MODELS = os.environ.get("TFB_MODELS", f"{ROOT_E}tfb_nn/models")                     # FINAL weights
t0 = time.time()
def npdir(name):
    """E:/tfb_nn/<name>/*.npy (t2_nn_extract.py) if present: one big read each, off the C: SSD; else the npz."""
    d = f"{ROOT_E}tfb_nn/{name}"
    if os.path.isdir(d):
        return {k[:-4]: np.load(f"{d}/{k}", allow_pickle=True, mmap_mode="r" if (k == "X.npy" and os.environ.get("NN_MMAP", "0") == "1") else None)
                for k in os.listdir(d) if k.endswith(".npy")}
    return np.load(f"{ROOT_C}Users/user/tfb/work/t2/{name}.npz", allow_pickle=True)


D = npdir(os.environ.get("NN_DATA", "nn_data"))   # nn_data_dem: + weekday x ToD demand forecast
MMAP = os.environ.get("NN_MMAP", "0") == "1"       # 10/13: X read in chunks from disk (peak RAM ~1.25 x X instead of ~2 x)
Xn = D["X"]; names = list(D["names"]); OFF = D["off"]; NW = len(OFF) - 1
PRIOR_COLS = [names.index(c) for c in ("prior_on", "prior_og")]
if not MMAP:
    for c in PRIOR_COLS:
        Xn[:, c] = 0.0
cond = D["cond"]; date = D["date"]; is_sel = D["is_sel"]; is_eval = D["is_eval"]
mon = np.array([d[:7] for d in date])
if "fold_month" in D:                               # 10/14 audit B: synthetic windows sit in their template month's fold
    mon = np.array([str(m) for m in D["fold_month"]])
    assert len(mon) == NW, "fold_month length differs from the window count"
F = Xn.shape[1]
# chunked host->GPU copy: keeps WSL RAM near one copy of X (a full-size where/astype copy peaked ~11 GB and
# pushed WSL into swap on the C: SSD; the PC hard-reset during that run)
Xg = torch.empty(Xn.shape, dtype=torch.float32, device=dev); Mg = torch.empty(Xn.shape, dtype=torch.bool, device=dev)
for a in range(0, len(Xn), 1 << 20):
    c = Xn[a:a + (1 << 20)]; m = ~np.isfinite(c)
    Xg[a:a + len(c)] = torch.from_numpy(np.where(m, 0, c).astype(np.float32, copy=False)).to(dev)
    Mg[a:a + len(c)] = torch.from_numpy(m).to(dev)
if MMAP:                                                    # same as zeroing the prior columns before the copy
    Xg[:, PRIOR_COLS] = 0.0; Mg[:, PRIOR_COLS] = False
Yg = torch.from_numpy(D["Y"]).to(dev); Eg = torch.from_numpy(D["EL"]).to(dev)
RAW = int(os.environ.get("NN_RAW", "0")); Rg = None                 # 10/31 P2(c): rows T-12..T-1 of R.npy (row T is empty under NO_T)
if RAW:
    Rg = torch.from_numpy(np.load(os.environ["NN_RAWFILE"], mmap_mode="r")[:, :12].astype(np.float16)).to(dev)
    assert Rg.shape[0] == Xg.shape[0], "raw history rows differ from X"
del Xn, c, m
if isinstance(D, dict):                                     # 10/22 speed: the host copy of X is not read after the upload (8.4 GB)
    D.pop("X", None)
LMAX = int(np.max(np.diff(OFF)))
starts = torch.from_numpy(OFF[:-1]).to(dev); lens = torch.from_numpy(np.diff(OFF)).to(dev)
ar = torch.arange(LMAX, device=dev)
print(f"loaded to GPU in {time.time()-t0:.0f}s, rows {Xg.shape}", flush=True)


class DilConv(nn.Module):
    def __init__(self, h, d, k=5):
        super().__init__(); self.d, self.k = d, k; self.lin = nn.Linear(h * k, h)

    def forward(self, z):
        L = z.shape[1]; pad = (self.k // 2) * self.d
        zp = nn.functional.pad(z, (0, 0, pad, pad))
        return self.lin(torch.cat([zp[:, i * self.d:i * self.d + L] for i in range(self.k)], dim=-1))


class TEnc(nn.Module):
    """10/31 P2(c): the raw history of one link -> h: rows T-12..T-1 x 3 channels, time-wise dilated blocks, last + mean"""
    def __init__(self, h, c=48):
        super().__init__()
        self.i = nn.Linear(3, c); self.b = nn.ModuleList([DilConv(c, d, 3) for d in (1, 2, 4)]); self.n = nn.ModuleList([nn.LayerNorm(c) for _ in range(3)])
        self.o = nn.Sequential(nn.Linear(2 * c, h), nn.GELU(), nn.Linear(h, h))

    def forward(self, r):
        B, L, _ = r.shape; z = self.i(r.view(B * L, 12, 3))
        for b, n in zip(self.b, self.n):
            z = z + nn.functional.gelu(b(n(z)))
        return self.o(torch.cat([z[:, -1], z.mean(1)], dim=-1)).view(B, L, -1)


class Net(nn.Module):
    def __init__(self, fin, h, depth=10, k=5, drop=0.0, glob=0, attn=0, heads=4, aux=0, dom=0, raw=0):
        super().__init__()
        self.fin = fin; self.tenc = TEnc(h) if raw else None           # 10/31 P2(c): the raw history is appended to x
        self.dom = nn.Parameter(torch.zeros(h)) if dom else None       # 10/14: learned shift for synthetic windows (d=1)
        self.auxh = nn.Linear(h, 6) if aux else None                   # auxiliary future-speed head (training only)
        self.attn = nn.ModuleList([nn.TransformerEncoderLayer(h, heads, 2 * h, drop, activation="gelu", batch_first=True, norm_first=True)
                                   for _ in range(attn)])          # all-pairs link attention after the conv stack
        self.glob = glob                                          # corridor-wide context every 5th block (padded batches only)
        self.gl = nn.ModuleList([nn.Linear(h, h) for _ in range(depth // 5)]) if glob else None
        self.inp = nn.Sequential(nn.Linear(fin, h), nn.GELU(), nn.Linear(h, h))     # fin = encode() width
        ds = [(1, 2, 4, 8, 16)[i % 5] for i in range(depth)]
        self.convs = nn.ModuleList([DilConv(h, d, k) for d in ds])
        self.mlps = nn.ModuleList([nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(h, h)) for _ in ds])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in ds])
        self.out = nn.Linear(h, 6)

    def forward(self, x, m, d=None):
        z = self.inp(x[..., :self.fin]); mm = m[:, :, None]
        if self.tenc is not None:
            z = z + self.tenc(x[..., self.fin:])
        if self.dom is not None and d is not None:                     # d: (B,) 1 = synthetic window; inference passes none (real)
            z = z + d[:, None, None].to(z.dtype) * self.dom
        for i, (c, f, n) in enumerate(zip(self.convs, self.mlps, self.norms)):
            z = z + f(c(n(z) * mm))
            if self.glob and i % 5 == 4:
                zn = n(z) * mm
                g = zn.sum(1) / mm.sum(1).clamp(min=1.0)             # masked mean over the window's links
                z = z + self.gl[i // 5](g)[:, None, :]
        for a in self.attn:
            z = a(z, src_key_padding_mask=m < 0.5)
        return torch.cat([self.out(z), self.auxh(z)], dim=-1) if self.auxh is not None else self.out(z)


LENS = np.diff(OFF)
ELW = np.add.reduceat(D["EL"].sum(1), OFF[:-1])               # eligible cells per window (loss normaliser)
BUCKET = int(os.environ.get("NN_BUCKET", "0"))   # 0 padded mixed batches, 1 equal-length batches (hurts), 2 packed mixed rows,
# 3 (10/14) = the batches of 0, each split by window length for compute and the gradients summed (no padding work)
COMPILE = os.environ.get("NN_COMPILE", "0") == "1"
import torch._dynamo as _dyn                                    # 10/22: raised whatever turns compile on (the env or the config
for _k in ("cache_size_limit", "recompile_limit", "accumulated_cache_size_limit", "accumulated_recompile_limit"):   # line's COMPILE;
    if hasattr(_dyn.config, _k):                                # at the default 8 a run silently fell back to eager)
        setattr(_dyn.config, _k, 256)
_LU = np.unique(LENS)                                           # 10/22 BUCKET 4: the cut that minimises the padded work with 2 pad lengths
LCUT = int(_LU[np.argmin([np.where(LENS <= c_, c_, LENS.max()).sum() for c_ in _LU])])
CHECK4 = os.environ.get("NN_B4_CHECK", "0") == "1"             # BUCKET 4: compare the first step's gradient with the padded batch
SAVEF = os.environ.get("NN_SAVEF", "0") == "1"                # 10/23: save each CV fold model (instrument v3 §2.1)
FOLD_SEL = os.environ.get("NN_FOLDS", "")                      # e.g. "0": screen on one fold only


VCOL = [names.index(c) for c in ("v_T", "v_T1", "v_T2", "v_m3", "v_m12", "v_min", "last_v", "v_trend") if c in names]
NBV = [(names.index(f"nb{o}_v"), o) for o in range(-5, 6) if f"nb{o}_v" in names]
# 10/12 JMODE=q: the same per-link multiplicative jitter on the flow/capacity columns (the link fingerprint, F45/G21),
# so the model leans less on the exact fingerprint; JMODE=v (default) = the speed columns as before
QCOL = [names.index(c) for c in ("q_T", "q_T1", "q_T2", "q_m3", "q_m12", "q_max", "last_q", "q_trend", "s_flow", "dem10_c",
                                 "dem20_c", "dem30_c", "dem30_over_dnmin") + tuple(f"arr{i}" for i in range(7)) if c in names]
NBQ = [(names.index(f"nb{o}_{k}"), o) for o in range(-5, 6) for k in ("q", "qmax") if f"nb{o}_{k}" in names]
JCOLS, JNB = VCOL, NBV
JWIN = False


def jitter(xr, jit, gen=None):
    """free-flow speed redraw of the test months (0.5-1.1% per link): own speed ratios x f_i, neighbour o x f_(i+o)"""
    fac = 1.0 + jit * torch.randn(xr.shape[:2], device=dev, generator=gen)
    if JWIN:                                                    # 10/31 P6: one uniform factor per window
        fac = (1.0 + jit * (2.0 * torch.rand(xr.shape[0], 1, device=dev, generator=gen) - 1.0)).expand(xr.shape[:2])
    xr = xr.clone(); xr[:, :, JCOLS] *= fac[:, :, None]
    L = xr.shape[1]; a = torch.arange(L, device=dev)
    for c, o in JNB:
        xr[:, :, c] *= fac[:, (a + o).clamp(0, L - 1)]
    return xr


TODC, DOWC = names.index("tod"), names.index("dow")
SCOL = [names.index(c) for c in ("capl", "lanes", "length", "vf", "w") if c in names]


def counterfact(xr, kind, gen, n):
    """M1 audit 2.4 memory stress (eval only, labels unchanged): t = tod +-U(1,3) h clipped to 5-23 h, r = static link
    columns rolled >= 10 links along the corridor, d = weekday <-> weekend."""
    xr = xr.clone(); B = xr.shape[0]
    u = lambda: torch.rand(B, device=dev, generator=gen)
    if "t" in kind:
        sh = (1 + 2 * u()) * torch.where(u() < 0.5, -1.0, 1.0)
        xr[:, :, TODC] = (xr[:, :, TODC] + sh[:, None]).clamp(5, 23)
    if "r" in kind:
        nn_ = n.clamp(min=1); s = torch.where(nn_ > 20, 10 + (u() * (nn_ - 20).float()).long(), nn_ // 2)
        pos = (torch.arange(xr.shape[1], device=dev)[None, :] + s[:, None]) % nn_[:, None]
        sub = xr[:, :, SCOL]; xr[:, :, SCOL] = torch.gather(sub, 1, pos[:, :, None].expand(-1, -1, len(SCOL)))
    if "d" in kind:
        wk = xr[:, :, DOWC] < 5
        xr[:, :, DOWC] = torch.where(wk, 5 + (u() < 0.5).float()[:, None], (u() * 5).floor()[:, None])
    return xr


ENC = {"rstat": 0, "tok": 0, "edges": None, "sent": 0}
SENTC = [names.index(c) for c in ("dq_up", "dq_dn", "head_rel", "tail_rel") if c in names]   # 99 = "no queue": SENT=1 makes it missing (10/10 audit)               # per-run input encoding, shared by training, OOF and FINAL inference
CAPC, LANEC = names.index("capl"), names.index("lanes")


def _sh(t, o, fill):
    """value of link i+o along the window (fill outside it)"""
    out = torch.full_like(t, fill)
    if o > 0:
        out[:, :-o] = t[:, o:]
    else:
        out[:, -o:] = t[:, :o]
    return out


def rel_static(xr, valid):
    """RSTAT: a link's relation to its neighbours instead of its own static values (no link fingerprint): lane change
    to the next link down / from the link up, downstream bottleneck ratio min(cap[i+1..i+3]) / cap[i], local narrowness
    cap[i] / max(cap[i-1], cap[i+1]); ratios rounded to 0.05. Links outside the window (padding, gaps) count as absent."""
    ln = xr[:, :, LANEC]
    capv = xr[:, :, CAPC] * ln if ENC["rstat"] == 2 else xr[:, :, CAPC]   # capl is per lane; RSTAT=2: total capacity (M5 audit H2)
    cap = torch.where(valid, capv, torch.zeros_like(capv))
    inf = float("inf"); big = torch.where(cap > 0, cap, torch.full_like(cap, inf))
    dn, up = _sh(cap, 1, 0.0) > 0, _sh(cap, -1, 0.0) > 0
    lane_dn = torch.where(dn, _sh(ln, 1, 0.0) - ln, torch.zeros_like(ln)); lane_up = torch.where(up, ln - _sh(ln, -1, 0.0), torch.zeros_like(ln))
    dmin = torch.minimum(torch.minimum(_sh(big, 1, inf), _sh(big, 2, inf)), _sh(big, 3, inf))
    r_dn = torch.where(torch.isfinite(dmin) & (cap > 0), dmin / cap.clamp(min=1), torch.ones_like(cap)).clamp(0, 3)
    nb = torch.maximum(_sh(cap, -1, 0.0), _sh(cap, 1, 0.0))
    r_loc = torch.where((nb > 0) & (cap > 0), cap / nb.clamp(min=1), torch.ones_like(cap)).clamp(0, 3)
    q = lambda t: torch.round(t * 20) / 20 - 1
    return torch.stack([lane_dn.clamp(-3, 3), lane_up.clamp(-3, 3), q(r_dn), q(r_loc)], dim=-1)


def encode(xr, nm, mu, sd, valid):
    """raw inputs (missing already 0, nm = missing mask) -> network input: normalised values with COLMASK, the missing
    mask, + RSTAT relations, + TOK piecewise-linear bin encodings (Gorishniy et al. 2022) of every normalised value."""
    if ENC["sent"]:                                             # the 99 km sentinel is "not applicable", not a distance
        add = torch.zeros_like(nm); add[..., SENTC] = xr[..., SENTC] >= 98.5; nm = nm | add
    nm = nm & (COLMASK > 0)
    x = torch.where(nm, torch.zeros_like(xr), (((xr - mu) / sd) * COLMASK).clamp(-10, 10))
    parts = [x, nm.float()]
    if ENC["rstat"]:
        parts.append(rel_static(xr, valid))
    if ENC["tok"]:
        e = ENC["edges"]; lo, hi = e[:, :-1], e[:, 1:]                       # (F, K)
        t = ((x[..., None] - lo) / (hi - lo).clamp(min=1e-6)).clamp(0, 1)   # (B, L, F, K)
        t = torch.where(nm[..., None], torch.zeros_like(t), t) * COLMASK[:, None]
        parts.append(t.flatten(-2))
    return torch.cat(parts, dim=-1)


def enc_width():
    return 2 * F + (4 if ENC["rstat"] else 0) + (F * ENC["tok"] if ENC["tok"] else 0)


def gather(ks, mu, sd, L=LMAX, fdrop=0.0, sdrop=0.0, jit=0.0, gen=None, cf="", crop=0.0):
    k = ks if torch.is_tensor(ks) else torch.as_tensor(ks, device=dev)
    a = ar[:L]
    idx = starts[k][:, None] + a[None, :]                       # (B, L)
    valid = a[None, :] < lens[k][:, None]
    idx = torch.where(valid, idx, torch.zeros_like(idx))
    global G_IDX
    G_IDX = idx
    xr = jitter(Xg[idx], jit, gen) if jit > 0 else Xg[idx]
    if cf:
        xr = counterfact(xr, cf, gen, lens[k])
    if crop > 0:                                                # CROP: train on a random sub-corridor (>= crop of the links)
        n = lens[k]; m = (n.float() * (crop + (1 - crop) * torch.rand(n.shape, device=dev))).long().clamp(min=1)
        lo = (torch.rand(n.shape, device=dev) * (n - m + 1).float()).long()
        valid = valid & (a[None, :] >= lo[:, None]) & (a[None, :] < (lo + m)[:, None])
    nm = Mg[idx]
    if fdrop > 0:                                               # augmentation: extra random missing inputs
        nm = nm | (torch.rand(nm.shape, device=dev) < fdrop)
    if sdrop > 0:                                               # hide the static link fingerprint for a whole window
        nm = nm | ((torch.rand(nm.shape[0], 1, 1, device=dev) < sdrop) & STATMASK)
    xb = encode(xr, nm, mu, sd, valid) * valid[:, :, None]
    if RAW:
        xb = torch.cat([xb, Rg[idx].float().flatten(2) * valid[:, :, None]], dim=-1)
    yb = Yg[idx].float() * valid[:, :, None]; eb = Eg[idx].float() * valid[:, :, None]
    if HOLD >= 0:
        eb = eb * (BLK[idx] != HOLD)[:, :, None]
    if NEARM is not None:                                       # OGNEAR: far ongoing cells leave the loss
        eb = eb * NEARM[idx][:, :, None]
    return xb, yb, eb, valid.float()


QW = np.add.reduceat((D["Y"] & D["EL"]).sum(1), OFF[:-1])       # queued eligible cells per window (loss weight)


def gather_rows(idx, valid, mu, sd, fdrop=0.0, sdrop=0.0, jit=0.0):
    """packed rows: idx/valid (R, LP) on the GPU; gaps are masked like padding."""
    xr = jitter(Xg[idx], jit) if jit > 0 else Xg[idx]
    nm = Mg[idx]
    if fdrop > 0:
        nm = nm | (torch.rand(nm.shape, device=dev) < fdrop)
    if sdrop > 0:                                               # packed rows: per row (several windows share it)
        nm = nm | ((torch.rand(nm.shape[0], 1, 1, device=dev) < sdrop) & STATMASK)
    v = valid[:, :, None]; e = Eg[idx].float() * v
    if NEARM is not None:
        e = e * NEARM[idx][:, :, None]
    return encode(xr, nm, mu, sd, valid) * v, Yg[idx].float() * v, e, valid.float()


NEARM = None                                                   # per-run (OGNEAR): 0 = far cell of an ongoing window
COLMASK = torch.ones(F, device=dev)                            # per-run: zero location-identifying columns
STATMASK = torch.zeros(F, dtype=torch.bool, device=dev)          # static link columns (for SDROP)
for _c in ("pos", "vf", "capl", "lanes", "length", "w"):
    if _c in names:
        STATMASK[names.index(_c)] = True
BS = int(os.environ.get("NN_BLOCK", "30"))                      # spatial-block CV: link blocks of BS, 4 folds round-robin
BLK = torch.from_numpy(np.concatenate([(np.arange(n) // BS) % 4 for n in np.diff(OFF)]).astype(np.int8)).to(dev)
HOLD = -1                                                      # block fold whose labels are hidden in training
PANEL = D["panel"] if "panel" in D else None


G_IDX = None; AUXG = None


def aux_loss(pred, idx, valid):
    """masked MSE of the auxiliary head against future v/vf (E:/tfb_nn/aux_vr.npy, aligned with nn_data rows)."""
    global AUXG
    if AUXG is None:
        AUXG = torch.from_numpy(np.load(f"{ROOT_E}tfb_nn/aux_vr.npy")).to(dev)
    ya = AUXG[idx].float(); ok = torch.isfinite(ya) & (valid[:, :, None] > 0)
    if AUXD:                                                    # 10/31 P7
        ya = torch.cat([ya[..., :1], 10.0 * (ya[..., 1:] - ya[..., :-1])], dim=-1); ok = torch.cat([torch.zeros_like(ok[..., :1]), ok[..., 1:] & ok[..., :-1]], dim=-1)
    d = torch.where(ok, pred.float() - torch.nan_to_num(ya), torch.zeros_like(pred, dtype=torch.float32))
    return (d ** 2).sum() / ok.sum().clamp(min=1)


def focal_w(logit, y, g):
    if g <= 0:
        return 1.0
    p = torch.sigmoid(logit.float().detach()); pt = torch.where(y > 0.5, p, 1 - p)
    return (1 - pt) ** g


def stats(tr):
    rows = torch.cat([torch.arange(OFF[k], OFF[k + 1], device=dev) for k in tr[::7]])
    x = Xg[rows]; m = ~Mg[rows]
    if ENC["sent"]:                                             # normalise the real distances only
        m[:, SENTC] &= x[:, SENTC] < 98.5
    cnt = m.sum(0).clamp(min=1); mu = (x * m).sum(0) / cnt
    sd = (((x - mu) ** 2) * m).sum(0).div(cnt).sqrt() + 1e-6
    return mu, sd


ONS = torch.from_numpy(np.asarray(cond) == "queue_onset").to(dev)   # onset flag per window (LISTW)


def listwise_onset(logit, yb, eb, kidx):
    """10/14: onset site choice is a within-window competition. Sum over the batch's onset windows of the cross-entropy
    between softmax over eligible links of the T+30 logits and the true queued links (equal shares)."""
    on = ONS[kidx]
    if not bool(on.any()):
        return logit.new_zeros(())
    z = logit[on][:, :, 5].float(); y5 = yb[on][:, :, 5]; e5 = eb[on][:, :, 5] > 0
    z = z.masked_fill(~e5, -1e4); tgt = y5 * e5; ts = tgt.sum(1)
    ce = -((tgt / ts.clamp(min=1)[:, None]) * torch.log_softmax(z, dim=1)).sum(1)
    return (ce * (ts > 0)).sum()


LOV = 0.0; TAILN = 0; AUXD = 0


def lovasz_ongoing(logit, yb, eb, kidx):
    """10/31 P5: Lovasz hinge (Berman et al. 2018) per ongoing window over its eligible cells, summed over the batch; vectorised:
    masked cells sort last and carry no weight"""
    og = ~ONS[kidx]
    if not bool(og.any()):
        return logit.new_zeros(())
    z = logit[og].float().flatten(1); y = yb[og].flatten(1); e = eb[og].flatten(1) > 0
    err = (1.0 - z * (2.0 * y - 1.0)).masked_fill(~e, -1e9); es, ix = torch.sort(err, dim=1, descending=True)
    gs = torch.gather(y * e, 1, ix); vs = torch.gather(e.float(), 1, ix); gts = gs.sum(1, keepdim=True)
    jac = 1.0 - (gts - gs.cumsum(1)) / (gts + ((1.0 - gs) * vs).cumsum(1)).clamp(min=1.0)
    grad = torch.cat([jac[:, :1], jac[:, 1:] - jac[:, :-1]], dim=1)
    return (es.clamp(min=0.0) * vs * grad.detach()).sum()


def run(TAG, H, B, LR, EPOCHS, SEED, FINAL, WD=1e-4, DEPTH=10, K=5, DROP=0.0, FDROP=0.0, WPOW=1.0,
        BUCKET=BUCKET, COMPILE=COMPILE, FOLD_SEL=FOLD_SEL, GLOBAL=0, ATTN=0, CV="month", ZERO="", SDROP=0.0, COND="", FOCAL=0.0, HEADS=4, AUX=0.0, GBAL=0, TJIT=0.0, EJIT=(), OGNEAR=0, ECF=(), RSTAT=0, TOK=0, CROP=0.0, SENT=0, JMODE="v", LISTW=0.0, SIMW=1.0, DOM=0, MW=1.0, EMA=0.0):
    torch.manual_seed(SEED)
    global JCOLS, JNB, JWIN
    JCOLS, JNB = {"v": (VCOL, NBV), "q": (QCOL, NBQ), "w": (QCOL, NBQ), "vq": (VCOL + QCOL, NBV + NBQ)}[JMODE]; JWIN = JMODE == "w"
    assert not (CROP and BUCKET == 2), "CROP needs one window per row (BUCKET 0)"
    assert not (LISTW and BUCKET == 2), "LISTW is implemented for BUCKET 0/1/3 only (audit F)"
    assert not (DOM and BUCKET == 2), "DOM is implemented for BUCKET 0/1/3 only"
    DOMt = torch.from_numpy(np.array([str(x)[:4] >= "2040" for x in date], np.float32)).to(dev) if DOM else None
    assert not (BUCKET in (3, 4) and (OGNEAR or CV != "month")), "BUCKET 3/4 denominators assume OGNEAR=0 and month CV (audit F)"
    ENC.update(rstat=RSTAT, tok=TOK, edges=None, sent=SENT)
    global NEARM
    NEARM = None
    if OGNEAR:          # 10/09: inference drops ongoing predictions far from a queue at T (t2_far_suppress), so the loss does too
        ci = names.index("last_queue"); lq = ((Xg[:, ci] == 1) & ~Mg[:, ci]).cpu().numpy(); keep = np.ones(len(lq), np.float32)   # Xn is freed after the GPU load
        for k in np.flatnonzero(cond == "queue_ongoing"):
            a, b = OFF[k], OFF[k + 1]; near = np.zeros(b - a, bool)
            for j in np.flatnonzero(lq[a:b]):
                near[max(0, j - OGNEAR):j + OGNEAR + 1] = True
            keep[a:b] = near
        NEARM = torch.from_numpy(keep).to(dev)
    FOLDS = [("NONE",)] if FINAL else [("2030-06", "2030-07", "2030-08"), ("2030-09", "2030-10", "2030-11"), ("2030-12", "2031-01", "2031-02")]
    global COLMASK
    COLMASK = torch.ones(F, device=dev)
    for c in [c for c in ZERO.split("+") if c]:
        if c == "ALL":
            COLMASK[:] = 0.0                                    # 10/31 X3a: the raw history alone
            continue
        COLMASK[names.index(c)] = 0.0
    grp = mon
    global HOLD
    if CV == "block" and not FINAL:                              # spatial blocks: labels hidden in one link-block set per fold
        assert BUCKET != 2, "block CV uses per-window loss weights (BUCKET 0)"
        FOLDS = [(f"B{i}",) for i in range(4)]; grp = None
    if CV == "panel" and not FINAL:                              # leave-panels-out: no location memory can help
        ps = sorted(set(PANEL)); FOLDS = [tuple(ps[i::4]) for i in range(4)]; grp = PANEL
    if FOLD_SEL and not FINAL:
        FOLDS = [FOLDS[int(i)] for i in FOLD_SEL.split(",")]
    oof = {}; oofj = {}; t1 = time.time(); diverged = False
    for months in FOLDS:
        if CV == "block" and not FINAL:
            HOLD = int(months[0][1:]); tr = np.flatnonzero(~is_sel); te = np.flatnonzero(is_eval | is_sel)
        else:
            HOLD = -1
            tr = np.flatnonzero(~np.isin(grp, months) & ~is_sel)
            te = np.flatnonzero(np.isin(grp, months) & (is_eval | is_sel))
        if COND:                                                # condition-specific model
            tr = tr[cond[tr] == COND]; te = te[cond[te] == COND]
        mu, sd = stats(tr); emean = float(ELW[tr].mean())
        if TOK:                                                 # bin edges = quantiles of the normalised training values
            rows = torch.cat([torch.arange(OFF[k], OFF[k + 1], device=dev) for k in tr[::29]])
            xs = torch.where(Mg[rows], torch.full_like(Xg[rows], float("nan")), (Xg[rows] - mu) / sd).clamp(-10, 10)
            qs = torch.linspace(0, 1, TOK + 1, device=dev)
            ENC["edges"] = torch.stack([torch.nanquantile(xs[:, j][torch.randperm(len(xs), device=dev)[:200000]], qs) if torch.isfinite(xs[:, j]).any()
                                        else torch.linspace(-1, 1, TOK + 1, device=dev) for j in range(F)])
        GW = np.ones(len(cond), np.float32)
        if GBAL:                                                # the metric counts every panel x condition group equally (R1 audit 2-2)
            assert PANEL is not None, "GBAL needs the panel column"
            key = cond[tr].astype(str) if GBAL == 3 else np.char.add(np.char.add(PANEL[tr].astype(str), "|"), cond[tr].astype(str))
            u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
            if GBAL == 1:                                       # equal window counts per group (10/08: overshoots, onset 89% of the loss)
                GW[tr] = (len(tr) / (len(u) * cnt))[inv]
            else:                                               # 2: equal loss mass per panel x condition, 3: per condition only
                mass = 100.0 * np.maximum(QW[tr], 3).astype(np.float64) ** (-WPOW) * ELW[tr]
                gm = np.bincount(inv, weights=mass)
                GW[tr] = (mass.sum() / (len(u) * gm))[inv]      # every group carries 1/len(u) of the loss, total unchanged
        if SIMW != 1.0:                                         # 10/14: down-weight synthetic windows (dates in the twin years >= 2040)
            sim = np.array([str(d)[:4] >= "2040" for d in date])
            GW[tr] = GW[tr] * np.where(sim[tr], SIMW, 1.0)
            GW[tr] = GW[tr] * (len(tr) / GW[tr].sum())        # mean training weight stays 1 (same effective step size)
            print(f"SIMW={SIMW}: synthetic training windows {int(sim[tr].sum())} of {len(tr)}, weight share {GW[tr][sim[tr]].sum() / GW[tr].sum():.3f}", flush=True)
        if MW != 1.0:                                           # 10/22 E4: weight of the March (validation-month) training windows
            mar = np.array([str(d)[:7] == "2031-03" for d in date])
            GW[tr] = GW[tr] * np.where(mar[tr], MW, 1.0)
            GW[tr] = GW[tr] * (len(tr) / GW[tr].sum())        # mean training weight stays 1
            print(f"MW={MW}: March training windows {int(mar[tr].sum())} of {len(tr)}, weight share {GW[tr][mar[tr]].sum() / GW[tr].sum():.3f}", flush=True)
        if RW_ != 1.0:                                          # 11/03 ledger #193: weight of the rare-birth training windows (g226)
            rr = np.load(os.environ["NN_RAREFILE"]); assert len(rr) == len(GW)
            GW[tr] = GW[tr] * np.where(rr[tr], RW_, 1.0)
            GW[tr] = GW[tr] * (len(tr) / GW[tr].sum())        # mean training weight stays 1
            on_ = np.array([str(c) == "queue_onset" for c in cond])[tr]
            print(f"RW={RW_}: rare-birth training windows {int(rr[tr].sum())} of {int(on_.sum())} onset, onset weight share {GW[tr][rr[tr]].sum() / GW[tr][on_].sum():.3f}", flush=True)
        GWt = torch.from_numpy(GW).to(dev)
        assert not ((GLOBAL or ATTN) and BUCKET == 2), "GLOBAL/ATTN need one window per row"
        net = Net(enc_width(), H, DEPTH, K, DROP, GLOBAL, ATTN, HEADS, AUX > 0, DOM, RAW).to(dev)
        fwd = torch.compile(net, dynamic=(COMPILE == 2) if BUCKET != 4 else False) if (COMPILE and (BUCKET != 3 or COMPILE == 2)) else net   # audit F: BUCKET 3 group shapes vary; 10/22: COMPILE=2 = dynamic shapes (for BUCKET 3)
        try:
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD, fused=True)
        except Exception:
            opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        if EMA > 0:                                             # 10/22 (vii): exponential moving average of the weights,
            ema_p = [q for q in net.parameters()]              # updated after every optimizer step (all batching paths),
            ema_s = [q.detach().clone() for q in ema_p]; ema_n = [0]   # copied into the net before evaluation / saving
            def _ema_hook(o, a, k):
                ema_n[0] += 1; dcy = min(EMA, (1 + ema_n[0]) / (10 + ema_n[0]))
                with torch.no_grad():
                    torch._foreach_lerp_(ema_s, [q.detach() for q in ema_p], 1 - dcy)
            opt.register_step_post_hook(_ema_hook)
        rng = np.random.default_rng(SEED)

        def epoch_batches():
            if BUCKET == 4:                                     # 10/22 speed: mixed batches with the same number of short (<= LCUT)
                s_ = rng.permutation(tr[LENS[tr] <= LCUT]); l_ = rng.permutation(tr[LENS[tr] > LCUT])   # and long windows in
                ns = int(round(B * len(s_) / len(tr))); nl = B - ns                                         # every batch (stratified
                assert ns > 0 and nl > 0, "BUCKET 4 needs windows on both sides of LCUT"                     # sampling): 2 static shapes
                return [(np.concatenate([s_[i * ns:(i + 1) * ns], l_[i * nl:(i + 1) * nl]]), LMAX) for i in range(min(len(s_) // ns, len(l_) // nl))]
            if not BUCKET or BUCKET == 3:                       # BUCKET 3: the same mixed batches as 0 (split only for compute)
                p = rng.permutation(tr)
                return [(p[i:i + B], LMAX) for i in range(0, len(p) - B + 1, B)]
            out = []
            for Lg in np.unique(LENS[tr]):
                g = rng.permutation(tr[LENS[tr] == Lg])
                out += [(g[i:i + B], int(Lg)) for i in range(0, len(g), B) if len(g) - i >= B // 4]
            return [out[o] for o in rng.permutation(len(out))]

        if BUCKET == 2:                                         # pack mixed-panel windows into rows, gap >= conv reach
            GAP, LP = (K // 2) * 16, 512
            WB = (100.0 * np.maximum(QW, 3).astype(np.float64) ** (-WPOW) * GW).astype(np.float32)

            def pack():                                         # first-fit over the most recent open rows (~90% fill)
                rows, rem = [], np.zeros(len(tr), np.int64)
                for w in rng.permutation(tr):
                    n = int(LENS[w]); lo = max(0, len(rows) - 256)
                    hit = np.flatnonzero(rem[lo:len(rows)] >= n + GAP)
                    if len(hit):
                        r = lo + hit[0]; o = LP - rem[r] + GAP
                    else:
                        r = len(rows); rows.append([]); o = 0; rem[r] = LP
                    rows[r].append((w, o)); rem[r] = LP - (o + n)
                R = len(rows); IDX = np.zeros((R, LP), np.int64); VAL = np.zeros((R, LP), bool); TW = np.zeros((R, LP), np.float32)
                for r, row in enumerate(rows):
                    for w, o in row:
                        n = LENS[w]; IDX[r, o:o + n] = np.arange(OFF[w], OFF[w] + n); VAL[r, o:o + n] = True; TW[r, o:o + n] = WB[w]
                return IDX, VAL, TW, np.array([len(r) for r in rows])

            I0, _, _, NW0 = pack(); RB = max(1, int(round(B / NW0.mean())))   # rows per step ~ B windows
            nb = len(I0) // RB; rng = np.random.default_rng(SEED)
        else:
            nb = len(epoch_batches()); rng = np.random.default_rng(SEED)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * nb); tail_c = [0, None]
        for ep in range(EPOCHS):
            tot = torch.zeros((), device=dev)
            if BUCKET == 2:
                IDX, VAL, TW, NWR = pack()
                IDXt, VALt, TWt = (torch.from_numpy(a).to(dev) for a in (IDX, VAL, TW))
                for s in range(nb):
                    sl = slice(s * RB, (s + 1) * RB)
                    xb, yb, eb, mb = gather_rows(IDXt[sl], VALt[sl], mu, sd, FDROP, SDROP, TJIT)
                    with torch.autocast(**AC):
                        logit = fwd(xb, mb)
                    if AUX > 0:
                        ap, logit = logit[..., 6:], logit[..., :6]
                    l = nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * eb * focal_w(logit, yb, FOCAL)
                    loss = (l.sum(-1) * TWt[sl]).sum() / (float(NWR[sl].sum()) * emean + 1)
                    if AUX > 0:
                        loss = loss + AUX * aux_loss(ap, IDXt[sl], VALt[sl].float())
                    opt.zero_grad(set_to_none=True); SCALER.scale(loss).backward(); SCALER.step(opt); SCALER.update(); sched.step(); tot += loss.detach()
                print(f"{TAG} fold {months[0]} epoch {ep} loss {tot.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
                continue
            bl = epoch_batches()
            allk = torch.as_tensor(np.concatenate([b for b, _ in bl]), device=dev)   # one upload per epoch, no per-step sync
            pos = 0
            if BUCKET in (3, 4):                                # 10/14 audit D: same batches, split by window length (<= 8 groups)
                assert AUX == 0 and CROP == 0, "BUCKET 3/4 support AUX=0, CROP=0"  # gradients summed over the groups, one step:
                for kb, Lg in bl:                               # mathematically the BUCKET 0 update without padding work
                    pos += len(kb); den = float(ELW[kb].sum()) + 1.0        # = eb.sum() + 1 of the padded batch
                    groups = ([(kb[LENS[kb] == Lu], int(Lu)) for Lu in np.unique(LENS[kb])] if BUCKET == 3 else
                              [(kb[LENS[kb] <= LCUT], LCUT), (kb[LENS[kb] > LCUT], LMAX)])   # BUCKET 4: 2 pad lengths, fixed counts
                    if BUCKET == 4 and CHECK4 and ep == 0 and pos == len(kb):
                        check4(net, kb, groups, den, mu, sd, GWt, DOMt if DOM else None, WPOW, FOCAL)
                    opt.zero_grad(set_to_none=True)
                    for kk, Lu in groups:
                        kg = torch.as_tensor(kk, device=dev)
                        xb, yb, eb, mb = gather(kg, mu, sd, int(Lu), FDROP, SDROP, TJIT)
                        wb = 100.0 * torch.clamp((yb * eb).sum(dim=(1, 2)), min=3) ** (-WPOW) * GWt[kg]
                        with torch.autocast(**AC):
                            logit = fwd(xb, mb, DOMt[kg] if DOM else None)
                        l = nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * eb * focal_w(logit, yb, FOCAL)
                        loss = (l.sum(dim=(1, 2)) * wb).sum() / den
                        if LISTW > 0:
                            loss = loss + LISTW * 100.0 * listwise_onset(logit, yb, eb, kg) / den
                        SCALER.scale(loss).backward(); tot += loss.detach()
                    SCALER.step(opt); SCALER.update(); sched.step()
                print(f"{TAG} fold {months[0]} epoch {ep} loss {tot.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
                continue
            for kb, Lg in bl:
                ks = allk[pos:pos + len(kb)]; pos += len(kb)
                xb, yb, eb, mb = gather(ks, mu, sd, Lg, FDROP, SDROP, TJIT, crop=CROP)
                wb = 100.0 * torch.clamp((yb * eb).sum(dim=(1, 2)), min=3) ** (-WPOW) * GWt[ks]
                with torch.autocast(**AC):
                    logit = fwd(xb, mb, DOMt[ks] if DOM else None)
                if AUX > 0:
                    ap, logit = logit[..., 6:], logit[..., :6]
                l = nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * eb * focal_w(logit, yb, FOCAL)
                den = (len(kb) * emean + 1) if BUCKET else (eb.sum() + 1)
                loss = (l.sum(dim=(1, 2)) * wb).sum() / den
                if LISTW > 0:
                    loss = loss + LISTW * 100.0 * listwise_onset(logit, yb, eb, ks) / den
                if LOV > 0:
                    loss = loss + LOV * 100.0 * lovasz_ongoing(logit, yb, eb, ks) / den
                if AUX > 0:
                    loss = loss + AUX * aux_loss(ap, G_IDX, mb)
                opt.zero_grad(set_to_none=True); SCALER.scale(loss).backward(); SCALER.step(opt); SCALER.update(); sched.step(); tot += loss.detach()
            print(f"{TAG} fold {months[0]} epoch {ep} loss {tot.item():.3f} ({time.time()-t0:.0f}s)", flush=True)
            if TAILN > 0 and ep >= EPOCHS - TAILN:                  # 10/31 P8
                with torch.no_grad():
                    cur = ema_s if EMA > 0 else [q.detach() for q in net.parameters()]; tail_c[0] += 1
                    tail_c[1] = [q.clone() for q in cur] if tail_c[1] is None else [a + (b - a) / tail_c[0] for a, b in zip(tail_c[1], cur)]
        diverged |= not np.isfinite(tot.item())                  # M5 audit M2: every fold, not only the last
        if EMA > 0:
            with torch.no_grad():
                torch._foreach_copy_([q.data for q in ema_p], ema_s)
            print(f"{TAG}: EMA {EMA} weights after {ema_n[0]} steps", flush=True)
        if TAILN > 0:
            with torch.no_grad():
                torch._foreach_copy_([q.data for q in net.parameters()], tail_c[1])
            print(f"{TAG}: mean of the last {tail_c[0]} epochs", flush=True)
        net.eval()
        if SAVEF and not FINAL:                                 # 10/23 instrument v3 §2.1: the fold model (out-of-month predictions)
            os.makedirs(MODELS, exist_ok=True)
            torch.save({k: v.cpu() for k, v in net.state_dict().items()} | {"_mu": mu.cpu(), "_sd": sd.cpu(), "_colmask": COLMASK.cpu(),
                        "_months": list(months), "_cfg": dict(H=H, DEPTH=DEPTH, K=K, GLOBAL=GLOBAL, ATTN=ATTN, HEADS=HEADS, AUX=AUX, ZERO=ZERO,
                        RSTAT=RSTAT, TOK=TOK, SENT=SENT, NN_DATA=os.environ.get("NN_DATA", "nn_data"), NN_TEST=os.environ.get("NN_TEST", "nn_test"))},
                       f"{MODELS}/{TAG}_fold{months[0]}.pt")
            print(f"{TAG}: saved fold model {months[0]}", flush=True)
        tes = [(te[LENS[te] == Lg], int(Lg)) for Lg in np.unique(LENS[te])] if BUCKET else [(te, LMAX)]
        variants = [("", 0.0, "")] + [(f"j{round(j * 100)}", j, "") for j in EJIT] + [(c, 0.0, c) for c in ECF]
        for vn, jl, cfk in variants:                            # stress tests: same model, perturbed inputs (EJIT, ECF)
          gen = torch.Generator(device=dev); gen.manual_seed(1234)
          dst = oof if vn == "" else oofj.setdefault(vn, {})
          with torch.no_grad(), torch.autocast(**AC):
           for tg, Lg in tes:
            for i in range(0, len(tg), 256):
                ks = tg[i:i + 256]; xb, yb, eb, mb = gather(ks, mu, sd, Lg, jit=jl, gen=gen, cf=cfk)
                p = torch.sigmoid(net(xb, mb)[..., :6].float()).cpu().numpy()
                for j, k in enumerate(ks):
                    n = OFF[k + 1] - OFF[k]; pk = p[j, :n].T.copy()
                    if cond[k] == "queue_onset":
                        pk[:5] = 0.0
                    if HOLD >= 0:                                       # keep only the hidden block's cells from this fold
                        hb = (np.arange(n) // BS) % 4 == HOLD
                        dst.setdefault(int(k), np.zeros_like(pk))[:, hb] = pk[:, hb]
                    else:
                        dst[int(k)] = pk
    if diverged:                                                 # a diverged run must not leave probs/oof behind (10/10)
        raise FloatingPointError(f"{TAG}: training loss is not finite")
    assert not (RAW and FINAL), "P2(c): FINAL with the raw history needs the test R.npy path (added only if the probe passes)"
    if FINAL:
        os.makedirs(MODELS, exist_ok=True)                                   # weights: a few MB
        torch.save({k: v.cpu() for k, v in net.state_dict().items()} | {"_mu": mu.cpu(), "_sd": sd.cpu(), "_colmask": COLMASK.cpu(),   # M5 audit M1: what re-inference needs
                    "_cfg": dict(H=H, DEPTH=DEPTH, K=K, GLOBAL=GLOBAL, ATTN=ATTN, HEADS=HEADS, AUX=AUX, ZERO=ZERO, RSTAT=RSTAT, TOK=TOK, SENT=SENT,
                                 NN_DATA=os.environ.get("NN_DATA", "nn_data"), NN_TEST=os.environ.get("NN_TEST", "nn_test"))}
                   | ({"_edges": ENC["edges"].cpu()} if ENC["tok"] else {}),
                   f"{MODELS}/{TAG}.pt")
        Dt = npdir(os.environ.get("NN_TEST", "nn_test"))
        Xt, offt, wid, condt = np.array(Dt["X"]), Dt["off"], Dt["window_id"], Dt["cond"]   # np.array: writable even when NN_MMAP maps it
        for c in ("prior_on", "prior_og"):
            Xt[:, names.index(c)] = 0.0
        probs = {}
        with torch.no_grad():
            for i in range(len(wid)):                           # same encode() as training: no separate test-time path
                a, b = offt[i], offt[i + 1]; n = b - a
                v = torch.from_numpy(np.asarray(Xt[a:b], dtype=np.float32)).to(dev)[None]; nm = ~torch.isfinite(v)
                v = torch.where(nm, torch.zeros_like(v), v)
                xb = torch.zeros((1, LMAX, enc_width()), device=dev); mb = torch.zeros((1, LMAX), device=dev)
                xb[0, :n] = encode(v, nm, mu, sd, torch.ones((1, n), dtype=torch.bool, device=dev))[0]; mb[0, :n] = 1
                with torch.autocast(**AC):
                    p = torch.sigmoid(net(xb, mb)[..., :6].float()).cpu().numpy()[0, :n].T.copy()
                if condt[i] == "queue_onset":
                    p[:5] = 0.0
                probs[str(wid[i])] = p
        pickle.dump(probs, open(f"{OUT}probs_{TAG}.pkl", "wb"))
    else:
        for vn, o in oofj.items():                              # oof_<tag>_<variant>: evaluators read it as tag "<tag>_<variant>"
            pickle.dump(o, open(f"{OUT}oof_{TAG}_{vn}.pkl", "wb"))
        pickle.dump(oof, open(f"{OUT}oof_{TAG}.pkl", "wb"))
    HOLD = -1
    print("done", TAG, len(oof), f"{time.time()-t1:.0f}s", flush=True)


def check4(net, kb, groups, den, mu, sd, GWt, DOMt, wpow, focal):
    """BUCKET 4 identity check (NN_B4_CHECK=1, first step): the loss gradient of the batch padded to LMAX (BUCKET 0) and the
    sum over the 2 length groups, fp32, no input dropout; prints the largest difference relative to the largest gradient."""
    gs = []
    for parts in ([(kb, LMAX)], groups):
        net.zero_grad(set_to_none=True)
        for kk, Lp in parts:
            kg = torch.as_tensor(kk, device=dev); xb, yb, eb, mb = gather(kg, mu, sd, int(Lp))
            wb = 100.0 * torch.clamp((yb * eb).sum(dim=(1, 2)), min=3) ** (-wpow) * GWt[kg]
            logit = net(xb, mb, DOMt[kg] if DOMt is not None else None)
            l = nn.functional.binary_cross_entropy_with_logits(logit.float(), yb, reduction="none") * eb * focal_w(logit, yb, focal)
            ((l.sum(dim=(1, 2)) * wb).sum() / den).backward()
        gs.append([q.grad.detach().clone() for q in net.parameters() if q.grad is not None])
    d = max((a - b).abs().max().item() for a, b in zip(*gs)); s_ = max(a.abs().max().item() for a in gs[0])
    print(f"BUCKET 4 check: max |grad diff| {d:.3e}, max |grad| {s_:.3e}, relative {d / s_:.2e}, groups {[len(k) for k, _ in groups]} x {[L for _, L in groups]}", flush=True)
    net.zero_grad(set_to_none=True)


# config line: "tag,H=128,B=256,LR=4e-3,EP=12,SEED=0,WD=1e-4,DEPTH=10,K=5,DROP=0,FDROP=0,WPOW=1,FINAL=0,BUCKET=1,COMPILE=0,FOLDS=0"
# (omitted keys take these defaults / the env flags)
def run_line(c):
    f = c.split(","); kv = dict(x.split("=") for x in f[1:])
    KNOWN = {"LOV", "TAIL", "AUXD", "H", "B", "LR", "EP", "SEED", "FINAL", "WD", "DEPTH", "K", "DROP", "FDROP", "WPOW", "BUCKET", "COMPILE", "FOLDS",
             "GLOBAL", "ATTN", "CV", "ZERO", "SDROP", "COND", "FOCAL", "HEADS", "AUX", "GBAL", "TJIT", "EJIT", "OGNEAR", "ECF", "RSTAT", "TOK", "CROP", "SENT", "JMODE", "LISTW", "SIMW", "DOM", "MW", "EMA", "RW"}
    bad = set(kv) - KNOWN                                      # fail fast: an unknown key used to be ignored silently
    if bad:
        raise ValueError(f"unknown config keys {sorted(bad)} in {f[0]}")
    g = lambda k, d: type(d)(kv.get(k, d))
    global LOV, TAILN, AUXD, RW_; RW_ = float(kv.get("RW", 1)); AUXD = int(kv.get("AUXD", 0)); LOV = float(kv.get("LOV", 0)); TAILN = int(kv.get("TAIL", 0))
    run(f[0], g("H", 128), g("B", 256), g("LR", 4e-3), g("EP", 12), g("SEED", 0), g("FINAL", 0) == 1,
        g("WD", 1e-4), g("DEPTH", 10), g("K", 5), g("DROP", 0.0), g("FDROP", 0.0), g("WPOW", 1.0),
        g("BUCKET", BUCKET), g("COMPILE", int(COMPILE)), kv.get("FOLDS", FOLD_SEL).replace("+", ","), g("GLOBAL", 0), g("ATTN", 0), kv.get("CV", "month"), kv.get("ZERO", ""),
        g("SDROP", 0.0), kv.get("COND", "").replace("on", "queue_onset").replace("og", "queue_ongoing") if kv.get("COND") in ("on", "og") else "", g("FOCAL", 0.0), g("HEADS", 4), g("AUX", 0.0), g("GBAL", 0),
        g("TJIT", 0.0) / 100, tuple(float(x) / 100 for x in kv.get("EJIT", "").split("+") if x), g("OGNEAR", 0),
        tuple(c for c in kv.get("ECF", "").split("+") if c),   # ECF=t+r+tr: memory stress variants
        g("RSTAT", 0), g("TOK", 0), g("CROP", 0.0), g("SENT", 0), kv.get("JMODE", "v"), g("LISTW", 0.0), g("SIMW", 1.0), g("DOM", 0), g("MW", 1.0), g("EMA", 0.0))   # TJIT/EJIT in percent, EJIT=1+3


QUEUE = os.environ.get("NN_QUEUE", "")      # daemon: data + compiled graphs stay on the GPU; append lines to the file
cfgs = os.environ.get("NN_CONFIGS", "")
if QUEUE:
    seen = set()
    while True:
        try:
            lines = [l.strip() for l in open(QUEUE) if l.strip() and not l.startswith("#")]
        except OSError:
            lines = []
        done_f = lambda tg: any(os.path.exists(f"{OUT}{p}_{tg}.pkl") for p in ("oof", "probs"))
        seen |= {l.split(",")[0] for l in lines if l != "STOP" and done_f(l.split(",")[0])}   # restart-safe
        todo = [l for l in lines if l.split(",")[0] not in seen]
        if not todo:
            time.sleep(5); continue
        if todo[0] == "STOP":
            break
        seen.add(todo[0].split(",")[0])
        try:
            run_line(todo[0])
        except Exception as e:                                  # a bad config must not kill the daemon
            print("FAILED", todo[0].split(",")[0], type(e).__name__, str(e)[:300], flush=True)
            if dev.type == "cuda":
                torch.cuda.empty_cache()
    print("ALLDONE", f"{time.time()-t0:.0f}s", flush=True)
elif cfgs:
    for c in cfgs.split(";"):
        run_line(c)
    print("ALLDONE", f"{time.time()-t0:.0f}s", flush=True)
else:
    run(TAG, H, B, LR, EPOCHS, SEED, FINAL)