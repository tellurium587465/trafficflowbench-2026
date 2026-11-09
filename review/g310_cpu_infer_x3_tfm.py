"""G310 (11/05): CPU-only inference of the two trained network families that t2_nn_cpu_infer.py cannot load:
  x3af_s0/1  raw-history networks (Kaggle tfb-t2-x3f, trainer C:/tfb_data/kaggle_out/x3f/t2_nn3f.py, NN_RAW=1, ZERO=ALL, COND=og):
             input = encode() of X (all columns masked by ZERO=ALL) + R.npy rows T-12..T-1 (36 raw values, unnormalised) -> TEnc
  tfm4_s0-4  attention backbone (Kaggle tfb-t2-tfm1 / v2, trainer src/t2_nn2.py with DEPTH=2, ATTN=4, HEADS=4): the plain Net
Net / DilConv / TEnc are copied verbatim from t2_nn3f.py (self-contained: does not import src/). Weights + _mu/_sd/_colmask/_cfg
from NN/models/<tag>.pt (bit-identical to the Kaggle copies). Onset windows: rows 0-4 zeroed, as in the trainers.
usage: py g310_cpu_infer_x3_tfm.py <input dir under NN or absolute> <prefix> [tag,tag,...]
       writes C:/Users/user/tfb/work/t2/probs_<prefix>__<tag>.pkl = {window_id: (6, n_links) float32}"""
import os, sys, time, pickle, numpy as np, torch, torch.nn as nn

torch.set_num_threads(4)
NN = "C:/tfb_data/tfb_nn"; WORK = "C:/Users/user/tfb/work/t2"
TAGS = ["x3af_s0", "x3af_s1"] + [f"tfm4_s{s}" for s in range(5)]


class DilConv(nn.Module):
    def __init__(self, h, d, k=5):
        super().__init__(); self.d, self.k = d, k; self.lin = nn.Linear(h * k, h)

    def forward(self, z):
        L = z.shape[1]; pad = (self.k // 2) * self.d
        zp = nn.functional.pad(z, (0, 0, pad, pad))
        return self.lin(torch.cat([zp[:, i * self.d:i * self.d + L] for i in range(self.k)], dim=-1))


class TEnc(nn.Module):
    def __init__(self, h, c=48):
        super().__init__()
        self.i = nn.Linear(3, c); self.b = nn.ModuleList([DilConv(c, d, 3) for d in (1, 2, 4)]); self.n = nn.ModuleList([nn.LayerNorm(c) for _ in range(3)])
        self.o = nn.Sequential(nn.Linear(2 * c, h), nn.GELU(), nn.Linear(h, h))

    def forward(self, r):
        B, L, _ = r.shape; z = self.i(r.view(B * L, 12, 3))
        for b, n in zip(self.b, self.n):
            z = z + nn.functional.gelu(b(n(z)))
        return self.o(torch.cat([z[:, -1], z.mean(1)], dim=-1)).view(B, L, -1)


class Net(nn.Module):                                              # t2_nn3f.py Net (dom: training only, not built here)
    def __init__(self, fin, h, depth=10, k=5, drop=0.0, glob=0, attn=0, heads=4, aux=0, raw=0):
        super().__init__()
        self.fin = fin; self.tenc = TEnc(h) if raw else None
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
        z = self.inp(x[..., :self.fin]); mm = m[:, :, None]
        if self.tenc is not None:
            z = z + self.tenc(x[..., self.fin:])
        for i, (c, f, n) in enumerate(zip(self.convs, self.mlps, self.norms)):
            z = z + f(c(n(z) * mm))
            if self.glob and i % 5 == 4:
                zn = n(z) * mm
                g = zn.sum(1) / mm.sum(1).clamp(min=1.0)
                z = z + self.gl[i // 5](g)[:, None, :]
        for a in self.attn:
            z = a(z, src_key_padding_mask=m < 0.5)
        return torch.cat([self.out(z), self.auxh(z)], dim=-1) if self.auxh is not None else self.out(z)


def load(tag):
    d = torch.load(f"{NN}/models/{tag}.pt", map_location="cpu", weights_only=False); cfg = d["_cfg"]
    if cfg.get("TOK") or cfg.get("RSTAT") or cfg.get("SENT"):
        sys.exit(f"{tag}: TOK/RSTAT/SENT encodings are not implemented here")
    mu, sd, cm = d["_mu"].float(), d["_sd"].float(), d["_colmask"].float(); F = mu.shape[0]
    raw = any(k.startswith("tenc.") for k in d)
    net = Net(2 * F, cfg["H"], cfg["DEPTH"], cfg["K"], 0.0, cfg["GLOBAL"], cfg["ATTN"], cfg["HEADS"], cfg["AUX"] > 0, raw)
    net.load_state_dict({k: v for k, v in d.items() if not k.startswith("_") and k != "dom"}); net.eval()   # strict
    assert torch.isfinite(sd).all() and bool((sd[cm > 0] > 0).all()), (tag, "sd")
    sd = torch.where(cm > 0, sd, torch.ones_like(sd))            # masked columns: x is multiplied by cm=0 anyway; avoid 0*inf
    return net, mu, sd, cm, raw


def infer(tag, d):
    net, mu, sd, cm, raw = load(tag); probs = {}
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
    zero = [names.index(c) for c in ("prior_on", "prior_og")]
    X = np.load(f"{d}/X.npy", mmap_mode="r"); off = np.load(f"{d}/off.npy"); wid = np.load(f"{d}/window_id.npy", allow_pickle=True)
    cond = np.load(f"{d}/cond.npy", allow_pickle=True); Rw = np.load(f"{d}/R.npy", mmap_mode="r") if raw else None
    if raw:
        assert Rw.shape[0] == X.shape[0] and Rw.shape[1:] == (13, 3), ("R.npy shape", Rw.shape)
    with torch.no_grad():
        for i in range(len(wid)):
            a, b = off[i], off[i + 1]
            v = torch.from_numpy(np.array(X[a:b], dtype=np.float32))[None]; v[..., zero] = 0.0
            nm = ~torch.isfinite(v); v = torch.where(nm, torch.zeros_like(v), v); nm = nm & (cm > 0)
            x = torch.where(nm, torch.zeros_like(v), (((v - mu) / sd) * cm).clamp(-10, 10))
            parts = [x, nm.float()]
            if raw:                                               # trainer: R[:, :12].astype(float16) -> .float() -> flatten
                r = np.array(Rw[a:b, :12]).astype(np.float16).astype(np.float32)
                assert np.isfinite(r).all(), (tag, str(wid[i]), "R has non-finite values")
                parts.append(torch.from_numpy(r).reshape(1, b - a, 36))
            p = torch.sigmoid(net(torch.cat(parts, dim=-1), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.astype(np.float32)
            if cond[i] == "queue_onset":
                p[:5] = 0.0
            probs[str(wid[i])] = p
    return probs


if __name__ == "__main__":
    d = sys.argv[1] if os.path.isabs(sys.argv[1]) else f"{NN}/{sys.argv[1]}"; pre = sys.argv[2]
    tags = sys.argv[3].split(",") if len(sys.argv) > 3 else TAGS
    T0 = time.time()
    for tag in tags:
        t0 = time.time(); pr = infer(tag, d)
        pickle.dump(pr, open(f"{WORK}/probs_{pre}__{tag}.pkl", "wb"))
        print(tag, len(pr), "windows", f"{time.time() - t0:.1f}s", flush=True)
    print("total", f"{time.time() - T0:.1f}s", flush=True)
