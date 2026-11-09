"""CPU re-inference of saved Task 2 NN models (t2_nn2.py FINAL runs save weights + mu/sd/colmask/cfg, audit M1), so the
twin can be scored on new test families while the GPU is busy. Same encode() as t2_nn2.py for the configurations used
here (no TOK / RSTAT / SENT; refuses others). GPU runs use bf16 autocast, so outputs match to ~1e-2, not bit-exactly:
the identity check compares against the GPU probs of the same model on the same data.
usage: python t2_nn_cpu_infer.py <model tag>[,tag...] <nn_test dir glob under E:/tfb_nn> <out name>
       writes work/t2/probs_<out>__<tag>.pkl per model (window_id -> (6, n) probabilities)"""
import sys, glob, pickle, time, numpy as np, torch, torch.nn as nn
from common import NN, ROOT, WORK                                        # 10/22: data root off the failing E: (TFB_NN)

torch.set_num_threads(16)


class DilConv(nn.Module):
    def __init__(self, h, d, k=5):
        super().__init__(); self.d, self.k = d, k; self.lin = nn.Linear(h * k, h)

    def forward(self, z):
        L = z.shape[1]; pad = (self.k // 2) * self.d
        zp = nn.functional.pad(z, (0, 0, pad, pad))
        return self.lin(torch.cat([zp[:, i * self.d:i * self.d + L] for i in range(self.k)], dim=-1))


class Net(nn.Module):                                              # copied from t2_nn2.py (keep in sync)
    def __init__(self, fin, h, depth=10, k=5, drop=0.0, glob=0, attn=0, heads=4, aux=0):
        super().__init__()
        self.auxh = nn.Linear(h, 6) if aux else None
        self.attn = nn.ModuleList([nn.TransformerEncoderLayer(h, heads, 2 * h, drop, activation="gelu", batch_first=True, norm_first=True)
                                   for _ in range(attn)])
        self.glob = glob
        self.gl = nn.ModuleList([nn.Linear(h, h) for _ in range(depth // 5)]) if glob else None
        self.inp = nn.Sequential(nn.Linear(fin, h), nn.GELU(), nn.Linear(h, h))
        ds = [(1, 2, 4, 8, 16)[i % 5] for i in range(depth)]
        self.convs = nn.ModuleList([DilConv(h, d, k) for d in ds])
        self.mlps = nn.ModuleList([nn.Sequential(nn.GELU(), nn.Dropout(drop), nn.Linear(h, h)) for _ in ds])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in ds])
        self.out = nn.Linear(h, 6)

    def forward(self, x, m):
        z = self.inp(x); mm = m[:, :, None]
        for i, (c, f, n) in enumerate(zip(self.convs, self.mlps, self.norms)):
            z = z + f(c(n(z) * mm))
            if self.glob and i % 5 == 4:
                zn = n(z) * mm
                g = zn.sum(1) / mm.sum(1).clamp(min=1.0)
                z = z + self.gl[i // 5](g)[:, None, :]
        for a in self.attn:
            z = a(z, src_key_padding_mask=m < 0.5)
        return torch.cat([self.out(z), self.auxh(z)], dim=-1) if self.auxh is not None else self.out(z)


def legacy_cfg(tag, d):
    """10/22: models saved before the audit (no _cfg / _colmask, e.g. q15f/q19f/q24f): the config comes from the queue
    line that trained them (NN/queue*.txt or repro/nn_configs/production_runs.txt), checked against the weights."""
    import glob as _g
    line = None
    for f in sorted(_g.glob(f"{NN}/queue*.txt")) + [str(ROOT / "repro" / "nn_configs" / "production_runs.txt")]:
        for l in open(f, encoding="utf-8", errors="ignore"):
            if l.split(",")[0].strip() == tag:
                line = l.strip()
    if line is None:
        sys.exit(f"{tag}: no _cfg and no training config line found")
    kv = dict(x.split("=", 1) for x in line.split(",")[1:])
    cfg = dict(H=int(kv.get("H", 128)), DEPTH=int(kv.get("DEPTH", 10)), K=int(kv.get("K", 5)), GLOBAL=int(kv.get("GLOBAL", 0)),
               ATTN=int(kv.get("ATTN", 0)), HEADS=int(kv.get("HEADS", 4)), AUX=float(kv.get("AUX", 0)), ZERO=kv.get("ZERO", ""),
               TOK=int(kv.get("TOK", 0)), RSTAT=int(kv.get("RSTAT", 0)), SENT=int(kv.get("SENT", 0)))
    assert d["inp.0.weight"].shape[0] == cfg["H"], (tag, "H")
    assert bool(cfg["GLOBAL"]) == any(k.startswith("gl.") for k in d), (tag, "GLOBAL")
    assert cfg["ATTN"] == len({k.split(".")[1] for k in d if k.startswith("attn.")}), (tag, "ATTN")
    assert cfg["DEPTH"] == len({k.split(".")[1] for k in d if k.startswith("convs.")}), (tag, "DEPTH")
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
    cm = torch.ones(len(names))
    for c in [c for c in cfg["ZERO"].split("+") if c]:
        cm[names.index(c)] = 0.0
    return cfg, cm


def load_model(tag):
    d = torch.load(f"{NN}/models/{tag}.pt", map_location="cpu", weights_only=False)
    if "_cfg" not in d:
        cfg, cm0 = legacy_cfg(tag, d); d["_cfg"] = cfg; d["_colmask"] = cm0
    cfg = d["_cfg"]
    if cfg.get("TOK") or cfg.get("RSTAT") or cfg.get("SENT"):
        sys.exit(f"{tag}: TOK/RSTAT/SENT encodings are not implemented here")
    mu, sd, cm = d["_mu"].float(), d["_sd"].float(), d["_colmask"].float(); F = mu.shape[0]
    net = Net(2 * F, cfg["H"], cfg["DEPTH"], cfg["K"], 0.0, cfg["GLOBAL"], cfg["ATTN"], cfg["HEADS"], cfg["AUX"] > 0)
    net.load_state_dict({k: v for k, v in d.items() if not k.startswith("_") and k != "dom"}); net.eval()   # DOM shift: training only
    return net, mu, sd, cm


def infer(tag, dirs):
    net, mu, sd, cm = load_model(tag); probs = {}
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
    zero = [names.index(c) for c in ("prior_on", "prior_og")]
    for d in dirs:
        X = np.load(f"{d}/X.npy", mmap_mode="r"); off = np.load(f"{d}/off.npy"); wid = np.load(f"{d}/window_id.npy", allow_pickle=True)
        cond = np.load(f"{d}/cond.npy", allow_pickle=True)
        with torch.no_grad():
            for i in range(len(wid)):
                a, b = off[i], off[i + 1]
                v = torch.from_numpy(np.array(X[a:b], dtype=np.float32))[None]; v[..., zero] = 0.0
                nm = ~torch.isfinite(v); v = torch.where(nm, torch.zeros_like(v), v)
                nm = nm & (cm > 0)
                x = torch.where(nm, torch.zeros_like(v), (((v - mu) / sd) * cm).clamp(-10, 10))
                p = torch.sigmoid(net(torch.cat([x, nm.float()], dim=-1), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.copy()
                if cond[i] == "queue_onset":
                    p[:5] = 0.0
                probs[str(wid[i])] = p
    return probs


if __name__ == "__main__":
    tags, pat, out = sys.argv[1].split(","), sys.argv[2], sys.argv[3]
    dirs = sorted(glob.glob(f"{NN}/{pat}"))
    for tag in tags:
        t0 = time.time(); pr = infer(tag, dirs)
        pickle.dump(pr, open(f"{WORK}/t2/probs_{out}__{tag}.pkl", "wb"))
        print(tag, len(pr), "windows", f"{time.time()-t0:.0f}s", flush=True)
