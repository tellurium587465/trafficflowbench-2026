"""G116 (10/29, ledger #67, record only): the listwise onset loss (t2_nn2.py LISTW, softmax over the eligible links of the
T+30 logits in each onset window) on the attention backbone, Kaggle tfb-t2-tfm1 v3: tfm4 seed 0 with LISTW 0.3 / 1.0
against tfm4_s0 (same seed and configuration, BCE only). Each network alone, on M3 v3 dev:
  onset  : its probabilities -> physics onset mix (month-adaptive a) -> evidence rule -> decoder (as G110's platform check)
  ongoing: its raw probabilities -> cone -> decoder
Paired window differences vs tfm4_s0 with the window SE; the seed spread of the same family is in G110's platform line.
usage: TFB_M3VER=v3 py g116_listw_onset.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))
TAGS = {"base": "tfm4_s0", "L03": "tfm4L03_s0", "L10": "tfm4L10_s0"}
PT = {c: ld(f"probs_{PRE}__{t}.pkl") for c, t in TAGS.items()}
cols = {c: {} for c in TAGS}
for i, w in enumerate(G.WID):
    m = G.META[w]; Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); kd = m3v2.kdist(m["day"])
    if m["cond"] == "queue_ongoing":
        for c in TAGS:
            cols[c][w] = sum(pr * G.iou(og_pred(PT[c][w] * cone_f(m, k), m, k, None, None), m) for k, pr in kd.items())
    else:
        ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
        for c in TAGS:
            p = PT[c][w].copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
            cols[c][w] = sum(pr * G.iou(V.onset_pred(p, m, "cur", k), m) for k, pr in kd.items())
D = V.frame(cols); D.to_csv(W + "g116_listw.csv", index=False); dv = ~m3v2.is_holdout(D.day)
for cond in ("queue_onset", "queue_ongoing"):
    d = D[dv & (D.cond == cond)]
    out = [f"{cond} dev {len(d)}: tfm4_s0 alone {d.base.mean():.4f}"]
    for c in ("L03", "L10"):
        x = (d[c] - d.base).to_numpy(); out.append(f"LISTW {c[1:]}: {d[c].mean():.4f} (diff {x.mean():+.4f}, SE {x.std(ddof=1) / np.sqrt(len(x)):.4f})")
    print(" | ".join(out))
