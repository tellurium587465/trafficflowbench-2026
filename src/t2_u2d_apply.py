"""2D U-Net of ledger #133 (a Task 2 network of another family, trained on the raw history only: rows T-12 .. T-1 of v / vcut, q / cap and the
observed flag, plus a queue flag; fold 2 weights = the six training months before 2030-12, Kaggle tfb-t2-u2d) applied on the CPU to the
windows of a test directory's R.npy (t2_raw_hist.py) -> work/t2/probs_<tag>.pkl, onset rows T+5 .. T+25 zero (as t2_nn_cpu_infer.py).
11/03 ledger #201: mixed into A's ongoing windows at 0.2 right after the raw-history networks (t2_final_spec TFB_T2_OGU2D). Weights NN/models/<tag>.pt.
The model code is copied from review/t2_unet2d.py (keep in sync).
usage: python t2_u2d_apply.py [tag,...=u2d_s0,u2d_s1] [test dir=NN/nn_test_afdem] [out prefix=""]"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from common import NN, WORK
B0 = 64


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


def infer(tag, d):
    net = Net(B0); net.load_state_dict(torch.load(f"{NN}/models/{tag}.pt", map_location="cpu")); net.eval()
    R_ = np.load(f"{d}/R.npy", mmap_mode="r"); off_ = np.load(f"{d}/off.npy"); wid = [str(w) for w in np.load(f"{d}/window_id.npy", allow_pickle=True)]
    cond = np.load(f"{d}/cond.npy", allow_pickle=True); out = {}
    with torch.no_grad():
        for k in range(len(wid)):
            a, b = int(off_[k]), int(off_[k + 1]); r = torch.from_numpy(np.array(R_[a:b, :12], dtype=np.float32))[None]
            p = torch.sigmoid(net(inp(r))[..., :b - a]).numpy()[0].copy()
            if cond[k] == "queue_onset":
                p[:5] = 0.0
            out[wid[k]] = p
    return out


if __name__ == "__main__":
    torch.set_num_threads(2)
    tags = (sys.argv[1] if len(sys.argv) > 1 else "u2d_s0,u2d_s1").split(","); d = sys.argv[2] if len(sys.argv) > 2 else f"{NN}/nn_test_afdem"; pre = sys.argv[3] if len(sys.argv) > 3 else ""
    for tag in tags:
        pr = infer(tag, d); pickle.dump(pr, open(f"{WORK}/t2/probs_{pre}{tag}.pkl", "wb")); print(tag, len(pr), "windows ->", f"probs_{pre}{tag}.pkl", flush=True)
