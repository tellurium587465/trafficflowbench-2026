"""X1 under M3 v2 (strategy §11 (b)): for each March day D with M3 windows and each k = 1..min(6, D-1), fine-tune the
saved model on the March windows of days D-k .. D-1 (t2_nn_finetune.py, CPU) EXCLUDING the windows whose rows T-12..T+6
touch the selector-simulated blanks of their panel (m3v2.sim_blank_rows(panel, D, k, start of day D)), and predict all
M3 windows of day D. Writes work/t2/probs_m3v2x1<arm>__<tag>.pkl = {(window_id, k): (6, n)} (k = 0: the saved model).
usage: python t2_m3v2_x1.py <tag[,tag...]> [lr=2e-4] [arm=""] [epochs=4] [replay dir=""] [data dir] [month]
V9 placebo (10/22): data dir nn_data_afdem + month 2031-02 fine-tunes on the same days of February, subsampled to the
number of March windows the real arm would use (February has ~2.5x more windows per day)."""
import sys, time, pickle, numpy as np
import torch
import t2_nn_finetune as FT
import m3v2
from common import NN

tags = sys.argv[1].split(","); LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4
ARM = sys.argv[3] if len(sys.argv) > 3 else ""; EP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
REPLAY = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
DATA = sys.argv[6] if len(sys.argv) > 6 else "nn_data_march"; MONTH = sys.argv[7] if len(sys.argv) > 7 else None   # V9 placebo: nn_data_afdem 2031-02
torch.set_num_threads(12)
W = "C:/Users/user/tfb/work/t2/"
import os
VER = os.environ.get("TFB_M3VER", "")                         # 10/23: "v3" -> the v3 window set
KMIN = int(os.environ.get("X1_KMIN", "1"))                    # 10/23 X1 onset rescue: fine-tune only when k >= KMIN (else the saved model)
L2SP = float(os.environ.get("X1_L2SP", "0"))                  # 10/23: L2-SP strength (t2_nn_finetune.finetune l2sp)
META = pickle.load(open(W + f"m3{VER}_meta.pkl", "rb"))["meta"]
byday = {}
for m in META:
    byday.setdefault(m["day"], []).append(m["window_id"])
MD = FT.data("nn_data_march"); MP = np.load(f"{NN}/nn_data_march/panel.npy", allow_pickle=True); MT = np.load(f"{NN}/nn_data_march/t.npy")


def excluded(D0, k):
    """indices of March training windows inside the simulated blanks for day D0 (0-based) with k earlier days"""
    ex = set()
    for p in np.unique(MP):
        hid = m3v2.sim_blank_rows(str(p), D0, k, D0 * 288)
        if not hid:
            continue
        for i in np.flatnonzero(MP == p):
            if any(r in hid for r in range(int(MT[i]) - 12, int(MT[i]) + 7)):
                ex.add(int(i))
    return ex


for tag in tags:
    t0 = time.time(); base = pickle.load(open(W + f"probs_m3{VER}__{tag}.pkl", "rb")); out = {}; nft = 0; nwin = 0
    for D0, wids in sorted(byday.items()):
        for w in wids:
            out[(w, 0)] = base[w]
        for k in range(1, min(6, D0) + 1):
            if k < KMIN:
                for w in wids:
                    out[(w, k)] = base[w]
                continue
            D1 = D0 + 1                                             # 1-based day of month
            ex = excluded(D0, k)
            if DATA == "nn_data_march":
                net, mu, sd, cm, n = FT.finetune(tag, DATA, D1 - k, D1 - 1, lr=LR, epochs=EP, replay_dir=REPLAY, exclude=ex, l2sp=L2SP)
            else:                                                   # placebo: the same number of windows as March would give
                nm = sum(1 for i in np.flatnonzero((MD["day"] >= D1 - k) & (MD["day"] <= D1 - 1)) if int(i) not in ex)
                net, mu, sd, cm, n = FT.finetune(tag, DATA, D1 - k, D1 - 1, lr=LR, epochs=EP, replay_dir=REPLAY, month=MONTH, max_n=nm)
            pr = FT.predict(net, mu, sd, cm, f"nn_test_m3{VER}", wids); nft += 1; nwin += n
            for w in wids:
                out[(w, k)] = pr[w]
    pickle.dump(out, open(W + f"probs_m3{VER}v2x1{ARM}__{tag}.pkl" if VER else W + f"probs_m3v2x1{ARM}__{tag}.pkl", "wb"))
    print(f"{tag}: {nft} fine-tunes, {nwin} windows seen in total, {time.time() - t0:.0f}s", flush=True)
