"""G351 (11/05, ledger #279 stage 1): fine-tune cvf_attnglob s0 / s1 fold 2030-06 (trained without 2030-06 .. 08) on the first-origin
windows of 2030-09 .. 2031-02 (G350's nn_data_first, both conditions, every day), t2_nn_finetune.finetune as X1 (lr 2e-4, 4 epochs,
batch 32, wd 1e-3, the model's WPOW, feature drop 0.05, seed 0, no replay), CPU. Predict the 80 official June windows (nn_test_jun80) and
the June replica selector windows (nn_test_junmh0) -> work/t2/probs_jun80ft__<tag>.pkl, probs_junmh0ft__<tag>.pkl.
usage: [TFB_FT_DATA=nn_data_v6first TFB_FT_SUF=v6ft] py -B -X utf8 g351_first_origin_ft.py"""
import sys, time, pickle
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
import numpy as np
import t2_nn_finetune as FT
import os
W = "C:/Users/user/tfb/work/t2/"; T0 = time.time(); DD = os.environ.get("TFB_FT_DATA", "nn_data_first"); SUF = os.environ.get("TFB_FT_SUF", "ft")   # #279b: nn_data_v6first / v6ft
for s in (0, 1):
    tag = f"cvf_attnglob_s{s}_fold2030-06"; wp = FT.wpow(tag)
    net, mu, sd, cm, n = FT.finetune(tag, DD, 1, 31, lr=2e-4, epochs=4, wpow=wp)
    print(f"{tag}: fine-tuned on {n} windows (WPOW {wp}) {time.time() - T0:.0f}s", flush=True)
    for td, out in (("nn_test_jun80", "jun80" + SUF), ("nn_test_junmh0", "junmh0" + SUF)):
        wids = [str(w) for w in np.load(f"C:/tfb_data/tfb_nn/{td}/window_id.npy", allow_pickle=True)]
        pr = FT.predict(net, mu, sd, cm, td, wids); pickle.dump(pr, open(W + f"probs_{out}__{tag}.pkl", "wb"))
        print(f"  {td}: {len(pr)} windows -> probs_{out}__{tag}.pkl ({time.time() - T0:.0f}s)", flush=True)
