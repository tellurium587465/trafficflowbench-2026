"""G193 (11/01, ledger #133): the 2D U-Net of tfb-t2-u2d (fold 2 weights) on the CPU: M3 windows -> work/t2/probs_m3v3__u2d_s<seed>.pkl, January and February -> work/t2/oof_u2d_s<seed>.pkl.
usage: py -B -X utf8 g193_u2d_infer.py <seed>"""
import os, sys, time, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SEED = int(sys.argv[1]); torch.set_num_threads(2); T0 = time.time(); NN = "C:/tfb_data/tfb_nn/"; W = "C:/Users/user/tfb/work/t2/"
s = open("C:/Users/user/tfb/review/t2_unet2d.py", encoding="utf-8").read(); exec(s[s.index("class Res(nn.Module)"):s.index("tr = np.flatnonzero(MON")])
net = Net(int(os.environ.get("U2D_B", "64"))); net.load_state_dict(torch.load(f"C:/tfb_data/kaggle_out/u2d/u2d_s{SEED}_fold.pt", map_location="cpu")); net.eval()


def run(R_, off_, ks, onset):
    out = {}
    with torch.no_grad():
        for k in ks:
            a, b = int(off_[k]), int(off_[k + 1]); r = torch.from_numpy(np.array(R_[a:b, :12], dtype=np.float32))[None]; p = torch.sigmoid(net(inp(r))[..., :b - a]).numpy()[0].copy()
            if onset[k]:
                p[:5] = 0.0
            out[k] = p
    return out


d = NN + "nn_test_m3v3/"; wid = [str(w) for w in np.load(d + "window_id.npy", allow_pickle=True)]; cond = np.load(d + "cond.npy", allow_pickle=True); r = run(np.load(d + "R.npy", mmap_mode="r"), np.load(d + "off.npy"), range(len(wid)), cond == "queue_onset"); pickle.dump({wid[k]: v for k, v in r.items()}, open(W + f"probs_m3v3__u2d_s{SEED}.pkl", "wb")); print("m3", len(r), f"{time.time() - T0:.0f}s", flush=True)
d = NN + "nn_data_afdem/"; date = np.load(d + "date.npy", allow_pickle=True); cond = np.load(d + "cond.npy", allow_pickle=True); isel = np.load(d + "is_sel.npy"); mon = np.array([str(x)[:7] for x in date]); ks = np.flatnonzero(isel & np.isin(mon, ("2031-01", "2031-02"))); r = run(np.load(d + "R.npy", mmap_mode="r"), np.load(d + "off.npy"), ks, cond == "queue_onset"); pickle.dump(r, open(W + f"oof_u2d_s{SEED}.pkl", "wb")); print("ALLDONE", len(r), f"{time.time() - T0:.0f}s", flush=True)
