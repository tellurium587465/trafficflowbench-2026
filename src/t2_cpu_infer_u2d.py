"""11/05 ledger #246: CPU inference of the 2D U-Net u2d_s0 / u2d_s1 (review/t2_unet2d.py, ledger #133; weights NN/models/u2d_s<seed>.pt =
the Kaggle fold files bit for bit). Res / Net / inp are copied verbatim from t2_unet2d.py; the inference is review/g193's: rows T-12 .. T-1 of
R.npy, onset windows rows 0-4 zeroed. Checked: identical to the shipped probs_u2d_s*.pkl on April's D7_I10_E windows (max |d| 0.0).
usage: python t2_cpu_infer_u2d.py <input dir under NN> <prefix or "-" for none> [seed,...]  -> WORK/t2/probs_<prefix>__u2d_s<seed>.pkl"""
import os, sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from common import NN, WORK
torch.set_num_threads(int(os.environ.get("U2D_THREADS", "2")))


class Res(nn.Module):
    def __init__(s, c, dl=1):
        super().__init__(); s.a = nn.Conv2d(c, c, 3, padding=(1, dl), dilation=(1, dl)); s.b = nn.Conv2d(c, c, 3, padding=(1, dl), dilation=(1, dl)); s.n1 = nn.GroupNorm(8, c); s.n2 = nn.GroupNorm(8, c)

    def forward(s, x):
        return x + s.b(Fn.gelu(s.n2(s.a(Fn.gelu(s.n1(x))))))


class Net(nn.Module):
    def __init__(s, b):
        super().__init__(); s.inp = nn.Conv2d(4, b, 3, padding=1); s.e1 = nn.Sequential(Res(b), Res(b)); s.d1 = nn.Conv2d(b, 2 * b, 3, stride=(1, 2), padding=1); s.e2 = nn.Sequential(Res(2 * b), Res(2 * b)); s.d2 = nn.Conv2d(2 * b, 4 * b, 3, stride=(1, 2), padding=1)
        s.mid = nn.Sequential(Res(4 * b, 1), Res(4 * b, 2), Res(4 * b, 4), Res(4 * b, 8)); s.u2 = nn.ConvTranspose2d(4 * b, 2 * b, (1, 2), stride=(1, 2)); s.c2 = nn.Sequential(nn.Conv2d(4 * b, 2 * b, 1), Res(2 * b)); s.u1 = nn.ConvTranspose2d(2 * b, b, (1, 2), stride=(1, 2)); s.c1 = nn.Sequential(nn.Conv2d(2 * b, b, 1), Res(b)); s.h = nn.Sequential(nn.Conv1d(2 * b, 2 * b, 3, padding=1), nn.GELU(), nn.Conv1d(2 * b, 6, 1)); nn.init.constant_(s.h[-1].bias, -3.0)

    def forward(s, x):
        z1 = s.e1(s.inp(x)); z2 = s.e2(s.d1(z1)); z = s.mid(s.d2(z2)); z = s.c2(torch.cat([s.u2(z), z2], 1)); z = s.c1(torch.cat([s.u1(z), z1], 1)); return s.h(torch.cat([z[:, :, -1], z.mean(2)], 1))


def inp(r):
    """r: (B, L, 12, 3) float -> (B, 4, 12, L + pad to a multiple of 4)"""
    v, q, o = r[..., 0], r[..., 1], r[..., 2]; x = torch.stack([v, q, o, ((v <= 1) & (o > 0)).float()], 1).permute(0, 1, 3, 2); p = (4 - x.shape[-1] % 4) % 4; return (Fn.pad(x, (0, p, 0, 0), mode="replicate") if p else x)




def infer(seed, d):
    net = Net(int(os.environ.get("U2D_B", "64"))); net.load_state_dict(torch.load(f"{NN}/models/u2d_s{seed}.pt", map_location="cpu")); net.eval()
    wid = [str(w) for w in np.load(f"{d}/window_id.npy", allow_pickle=True)]; cond = np.load(f"{d}/cond.npy", allow_pickle=True); R = np.load(f"{d}/R.npy", mmap_mode="r"); off = np.load(f"{d}/off.npy"); out = {}
    with torch.no_grad():
        for i, w in enumerate(wid):
            a, b = int(off[i]), int(off[i + 1]); r = torch.from_numpy(np.array(R[a:b, :12], dtype=np.float32))[None]; p = torch.sigmoid(net(inp(r))[..., :b - a]).numpy()[0].copy()
            if cond[i] == "queue_onset":
                p[:5] = 0.0
            out[w] = p
    return out


if __name__ == "__main__":
    d = sys.argv[1] if os.path.isabs(sys.argv[1]) else f"{NN}/{sys.argv[1]}"; pre = sys.argv[2]; seeds = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0", "1"])]
    for s_ in seeds:
        r = infer(s_, d); f = f"{WORK}/t2/probs_{'' if pre == '-' else pre + '__'}u2d_s{s_}.pkl"; pickle.dump(r, open(f, "wb")); print(f"u2d_s{s_}", len(r), "windows ->", f, flush=True)
