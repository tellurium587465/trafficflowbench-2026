"""P2(c) (10/31, ledger #86): CPU inference of the raw-history fold models (review/t2_nn3.py) on M3 v3: the encode() of
t2_nn_cpu_infer.py, the raw history (R.npy rows T-12..T-1) appended, TEnc as in t2_nn3.py.
usage: py t2_nn3_infer.py <tag>[,tag...] [out name=m3v3cap]   (models in NN/models, writes work/t2/probs_<out>__<tag>.pkl)"""
import sys, pickle, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, r"C:/Users/user/tfb/src")
import t2_nn_cpu_infer as I
from common import NN, WORK


class TEnc(nn.Module):                                              # copied from t2_nn3.py (keep in sync)
    def __init__(self, h, c=48):
        super().__init__()
        self.i = nn.Linear(3, c); self.b = nn.ModuleList([I.DilConv(c, d, 3) for d in (1, 2, 4)]); self.n = nn.ModuleList([nn.LayerNorm(c) for _ in range(3)])
        self.o = nn.Sequential(nn.Linear(2 * c, h), nn.GELU(), nn.Linear(h, h))

    def forward(self, r):
        B, L, _ = r.shape; z = self.i(r.view(B * L, 12, 3))
        for b, n in zip(self.b, self.n):
            z = z + nn.functional.gelu(b(n(z)))
        return self.o(torch.cat([z[:, -1], z.mean(1)], dim=-1)).view(B, L, -1)


def load(tag):
    d = torch.load(f"{NN}/models/{tag}.pt", map_location="cpu", weights_only=False); cfg = d["_cfg"]
    mu, sd, cm = d["_mu"].float(), d["_sd"].float(), d["_colmask"].float(); F = mu.shape[0]; fin = 2 * F
    net = I.Net(fin, cfg["H"], cfg["DEPTH"], cfg["K"], 0.0, cfg["GLOBAL"], cfg["ATTN"], cfg["HEADS"], cfg["AUX"] > 0); te = TEnc(cfg["H"])
    net.load_state_dict({k: v for k, v in d.items() if not k.startswith("_") and k != "dom" and not k.startswith("tenc.")})
    te.load_state_dict({k[5:]: v for k, v in d.items() if k.startswith("tenc.")}); net.eval(); te.eval()
    inp0 = net.inp; del net._modules["inp"]; net.inp = lambda x: inp0(x[..., :fin]) + te(x[..., fin:])
    return net, mu, sd, cm


def infer(tag, d):
    net, mu, sd, cm = load(tag); probs = {}
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; zero = [names.index(c) for c in ("prior_on", "prior_og")]
    X = np.load(f"{d}/X.npy", mmap_mode="r"); Rw = np.load(f"{d}/R.npy", mmap_mode="r"); off = np.load(f"{d}/off.npy"); wid = np.load(f"{d}/window_id.npy", allow_pickle=True); cond = np.load(f"{d}/cond.npy", allow_pickle=True)
    with torch.no_grad():
        for i in range(len(wid)):
            a, b = off[i], off[i + 1]
            v = torch.from_numpy(np.array(X[a:b], dtype=np.float32))[None]; v[..., zero] = 0.0
            nm = ~torch.isfinite(v); v = torch.where(nm, torch.zeros_like(v), v); nm = nm & (cm > 0)
            x = torch.where(nm, torch.zeros_like(v), (((v - mu) / sd) * cm).clamp(-10, 10))
            rr = torch.from_numpy(np.array(Rw[a:b, :12], dtype=np.float32)).flatten(1)[None]
            p = torch.sigmoid(net(torch.cat([x, nm.float(), rr], dim=-1), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.copy()
            if cond[i] == "queue_onset":
                p[:5] = 0.0
            probs[str(wid[i])] = p
    return probs


if __name__ == "__main__":
    tags = sys.argv[1].split(","); out = sys.argv[2] if len(sys.argv) > 2 else "m3v3cap"
    for tag in tags:
        t0 = time.time(); pr = infer(tag, f"{NN}/nn_test_m3v3"); pickle.dump(pr, open(f"{WORK}/t2/probs_{out}__{tag}.pkl", "wb")); print(tag, len(pr), "windows", f"{time.time() - t0:.0f}s", flush=True)
