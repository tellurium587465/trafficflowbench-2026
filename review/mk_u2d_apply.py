"""11/03 ledger #201: writes tfb_a8/src/t2_u2d_apply.py, the CPU inference of the 2D U-Net (#133) on a test directory's raw history, with the
model code copied verbatim from review/t2_unet2d.py (the class Res / Net and inp()). usage: py mk_u2d_apply.py"""
s = open(r"C:/Users/user/tfb/review/t2_unet2d.py", encoding="utf-8").read()
code = s[s.index("class Res(nn.Module)"):s.index("tr = np.flatnonzero(MON")].rstrip() + "\n"
head = '''"""2D U-Net of ledger #133 (a Task 2 network of another family, trained on the raw history only: rows T-12 .. T-1 of v / vcut, q / cap and the
observed flag, plus a queue flag; fold 2 weights = the six training months before 2030-12, Kaggle tfb-t2-u2d) applied on the CPU to the
windows of a test directory's R.npy (t2_raw_hist.py) -> work/t2/probs_<tag>.pkl, onset rows T+5 .. T+25 zero (as t2_nn_cpu_infer.py).
11/03 ledger #201: mixed into A's ongoing windows at 0.2 right after the raw-history networks (t2_final_spec TFB_T2_OGU2D). Weights NN/models/<tag>.pt.
The model code is copied from review/t2_unet2d.py (keep in sync).
usage: python t2_u2d_apply.py [tag,...=u2d_s0,u2d_s1] [test dir=NN/nn_test_afdem] [out prefix=""]"""
import sys, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from common import NN, WORK
B0 = 64


'''
tail = '''

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
'''
open(r"C:/Users/user/tfb_a8/src/t2_u2d_apply.py", "w", encoding="utf-8").write(head + code + tail); print("written")
