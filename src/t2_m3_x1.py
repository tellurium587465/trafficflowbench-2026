"""X1 on M3 (STATE.md 10/22 pre-registration): per M3 window (day D, usable earlier days k = min(k, D - 1)), fine-tune the
saved FINAL model on the March windows of days D-k .. D-1 (t2_nn_finetune.py), predict that window; k = 0 -> the saved
model's prediction. One fine-tune per distinct (D, k). Writes work/t2/probs_m3x1<arm>__<tag>.pkl for every M3 window.
usage: python t2_m3_x1.py <tag[,tag...]> [lr=2e-4] [arm name=""] [epochs=4] [replay dir=""]"""
import sys, time, pickle, collections, numpy as np
import torch
torch.set_num_threads(12)
import t2_nn_finetune as FT
import t2_nn_cpu_infer as CI

tags = sys.argv[1].split(","); LR = float(sys.argv[2]) if len(sys.argv) > 2 else 2e-4
ARM = sys.argv[3] if len(sys.argv) > 3 else ""; EP = int(sys.argv[4]) if len(sys.argv) > 4 else 4
REPLAY = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None
W = "C:/Users/user/tfb/work/t2/"
META = pickle.load(open(W + "m3_meta.pkl", "rb"))["meta"]
groups = collections.defaultdict(list)
for m in META:
    D = m["day"] + 1; k = min(m["k"], D - 1); groups[(D, k)].append(m["window_id"])
for tag in tags:
    t0 = time.time(); base = pickle.load(open(W + f"probs_m3__{tag}.pkl", "rb")); out = {}; nwin = []
    for (D, k), wids in sorted(groups.items()):
        if k == 0:
            out.update({w: base[w] for w in wids}); continue
        net, mu, sd, cm, n = FT.finetune(tag, "nn_data_march", D - k, D - 1, lr=LR, epochs=EP, replay_dir=REPLAY)
        out.update(FT.predict(net, mu, sd, cm, "nn_test_m3", wids)); nwin.append(n)
    assert set(out) == set(base)
    pickle.dump(out, open(W + f"probs_m3x1{ARM}__{tag}.pkl", "wb"))
    print(f"{tag}: {len(groups)} (day, k) groups, fine-tunes {len(nwin)}, training windows per fine-tune median {int(np.median(nwin))}, {time.time() - t0:.0f}s", flush=True)
