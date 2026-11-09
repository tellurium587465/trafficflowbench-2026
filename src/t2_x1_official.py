"""X1 for the OFFICIAL Task 2 windows (10/22; used only if X1 is adopted on M3, t2_m3_x1.py): for each model and each
day D that has official windows, fine-tune the saved FINAL model on the same month's windows of days 1 .. D-1
(nn_data_march for the validation month, nn_data_april for the private month; t2_nn_finetune.py, CPU, seed 0) and predict
that day's official windows (nn_test_afdem). Day-1 windows keep the saved model's prediction. The window loss weight uses
the model's own WPOW (training config line). Nothing at or after day D is read.
Writes work/t2/probs_x1off<arm>__<tag>.pkl (all 160 windows) and, with the same fine-tuned nets, the 8 TTA draws
(nn_test_afdem_tta<k>, t2_tta_official_build.py) as work/t2/probs_x1tta<k><arm>__<tag>.pkl (day-1 windows: the saved
model's TTA probabilities probs_afdemtta<k>__<tag>.pkl).
usage: python t2_x1_official.py <tag[,tag...]> [lr=2e-4] [arm=""] [epochs=4]"""
import sys, time, glob, pickle, collections, numpy as np, pandas as pd
import torch
torch.set_num_threads(12)
import t2_nn_finetune as FT
from common import REL, NN, WORK
from cachelib import T2_PANELS

tags = sys.argv[1].split(","); LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4
ARM = sys.argv[3] if len(sys.argv) > 3 else ""; EP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
W = WORK + "/t2/"
DATA = {"validation": "nn_data_march", "private": "nn_data_april"}
import os
SPLITS = os.environ.get("X1_SPLITS", "validation,private").split(",")   # 10/22: e.g. validation only for the s76 LB probe
NT = int(os.environ.get("X1_TTA", "8"))                               # 10/23: 0 = no TTA draws (the finals have no TTA)
groups = collections.defaultdict(list)
for sp in ("validation", "private"):
    for p in T2_PANELS:
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            groups[(sp, int(str(r.date)[8:10]))].append(r.window_id)


wpow = FT.wpow


for tag in tags:
    t0 = time.time(); base = pickle.load(open(W + f"probs_{tag}.pkl", "rb")); out = {}; nft = 0; wp = wpow(tag)
    tbase = {k: pickle.load(open(W + f"probs_afdemtta{k}__{tag}.pkl", "rb")) for k in range(1, NT + 1)}; tout = {k: {} for k in range(1, NT + 1)}
    for (sp, D), wids in sorted(groups.items()):
        if D == 1 or sp not in SPLITS:
            out.update({w: base[w] for w in wids})
            for k in tout:
                tout[k].update({w: tbase[k][w] for w in wids})
            continue
        net, mu, sd, cm, n = FT.finetune(tag, DATA[sp], 1, D - 1, lr=LR, epochs=EP, wpow=wp)
        out.update(FT.predict(net, mu, sd, cm, "nn_test_afdem", wids)); nft += 1
        for k in tout:
            tout[k].update(FT.predict(net, mu, sd, cm, f"nn_test_afdem_tta{k}", wids))
    assert set(out) == set(base), tag
    pickle.dump(out, open(W + f"probs_x1off{ARM}__{tag}.pkl", "wb"))
    for k in tout:
        pickle.dump(tout[k], open(W + f"probs_x1tta{k}{ARM}__{tag}.pkl", "wb"))
    print(f"{tag}: WPOW {wp}, {nft} fine-tunes, {time.time() - t0:.0f}s", flush=True)
