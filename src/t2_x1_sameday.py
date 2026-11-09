"""Same-day X1 for the official ongoing windows (11/01, ledger #125; M3 arm sd of review/g141_x1_arms.py, ledger #101). As t2_x1_official.py, but each fine-tune also takes the same day's windows that end before the
window's hour starts: for a window at row T of day D, the month's pool windows of days 1 .. D - 1 plus those of day D with T' + 6 <= c, c = the first row of T's hour (c <= T). Labels and inputs of the pool read rows <= c. One fine-tune per
(split, day, hour). An empty pool keeps the saved model, as day 1 in t2_x1_official.py. Writes work/t2/probs_x1off<arm>__<tag>.pkl (the ongoing windows only: X1 is not used on onset).
11/03 ledger #189 (A13): X1_ARM names the arm (default sd), X1_POOL_PRIV the April pool (default nn_data_april = origins every 6 rows;
the dense arm sdd3 = nn_data_april_sp3, twin/build_march.py 3 private with TFB_MONTH_OUT), X1_FILL an arm whose file gives the windows
of the splits not run (sdd3: March stays sd).
usage: python t2_x1_sameday.py <tag[,tag...]> [lr=2e-4] [epochs=4]   (env X1_SPLITS as t2_x1_official.py)"""
import os, sys, time, pickle, collections, numpy as np, pandas as pd
import torch
import t2_nn_finetune as FT
from common import REL, NN, WORK
from cachelib import T2_PANELS, load

torch.set_num_threads(int(os.environ.get("X1_THREADS", "4")))     # the thread count is part of the bit-for-bit result (2e-7 between counts)
tags = sys.argv[1].split(","); LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4; EP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ARM, FILL = os.environ.get("X1_ARM", "sd"), os.environ.get("X1_FILL", "")
W = WORK + "/t2/"; DATA = {"validation": "nn_data_march", "private": os.environ.get("X1_POOL_PRIV", "nn_data_april")}; SPLITS = os.environ.get("X1_SPLITS", "validation,private").split(",")
groups = collections.defaultdict(list)
for sp in SPLITS:
    for p in T2_PANELS:
        ti = {s: i for i, s in enumerate(load(p, sp).times)}
        for r in pd.read_csv(REL / "task2" / p / sp / "window_index.csv").itertuples():
            if r.condition == "queue_ongoing":
                groups[(sp, int(str(r.date)[8:10]), ti[r.forecast_origin] // 12 * 12)].append(r.window_id)
        load.cache_clear()
MT = {sp: np.load(f"{NN}/{DATA[sp]}/t.npy") for sp in SPLITS}
DATA0, TEST = DATA, "nn_test_afdem"
for tag in tags:
    if tag.startswith("q24t"):                                  # 11/02 ledger #164: these networks read the row at T
        DATA, TEST = {k: v + "t" for k, v in DATA0.items()}, "nn_test_afdemt"
    else:
        DATA, TEST = DATA0, "nn_test_afdem"
    t0 = time.time(); base = pickle.load(open(W + f"probs_{tag}.pkl", "rb")); out = {}; nft = 0; wp = FT.wpow(tag)
    for (sp, D, c), wids in sorted(groups.items()):
        Dd = FT.data(DATA[sp]); late = set(np.flatnonzero((Dd["day"] == D) & (MT[sp] + 6 > c)).tolist())
        net, mu, sd, cm, n = FT.finetune(tag, DATA[sp], 1, D, lr=LR, epochs=EP, wpow=wp, exclude=late)
        if n == 0:
            out.update({w: base[w] for w in wids})
        else:
            out.update(FT.predict(net, mu, sd, cm, TEST, wids)); nft += 1
    if FILL:
        out = {**pickle.load(open(W + f"probs_x1off{FILL}__{tag}.pkl", "rb")), **out}
    pickle.dump(out, open(W + f"probs_x1off{ARM}__{tag}.pkl", "wb")); print(f"{tag}: WPOW {wp}, {len(groups)} groups, {nft} fine-tunes, {time.time() - t0:.0f}s", flush=True)
