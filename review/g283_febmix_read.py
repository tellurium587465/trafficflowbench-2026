"""G283 (11/05, ledger #229 stage 1, the read declared 11/04 22:05): G282's arms against the same-day X1 (arm sd) on M3 v3 dev ongoing
windows, G142's ongoing flow (A column: 0.8 (0.8 X1 + 0.2 tfm4 5 seeds) + 0.2 LightGBM -> cone -> og_pred), the mean of c24f_attnglob
s0 / s1, k expectation with the simulated blanks; hold-out days not read. m3v2.summary (private weights, cluster bootstrap) on all windows
and on the top third of history coverage (nn_test_m3v3/R.npy[:, :12, 2], G233's instrument).
usage: TFB_M3VER=v3 py -B -X utf8 g283_febmix_read.py odd | even <r>"""
import os, sys, glob, pickle
os.environ.setdefault("TFB_M3VER", "v3")
PAR = sys.argv[1]; RSEL = sys.argv[2] if len(sys.argv) > 2 else None; sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g142_x1_arms_score.py", encoding="utf-8").read().split("\nAR = {n: arm(n)")[0].split('sys.argv = [sys.argv[0]]\n', 1)[1]
exec(compile(src, "g142", "exec"))


def parts(name, days):
    PR = []
    for t in TAGS:
        d = {}
        for D0 in days:
            f = OUT + f"parts/{name}__{t}__{D0}.pkl"; assert os.path.exists(f), f; d.update(pickle.load(open(f, "rb")))
        PR.append(d)
    return {key: (PR[0][key] + PR[1][key]) / 2 for key in PR[0]}


DEVD = sorted({G.META[w]["day"] for w in G.WID if G.META[w]["cond"] == "queue_ongoing" and not m3v2.is_holdout(np.array([G.META[w]["day"]]))[0]})
DAYS = [d for d in DEVD if (d + 1) % 2 == (1 if PAR == "odd" else 0)]
ARMS = ["mix25", "mix50"] if PAR == "odd" else [f"mix{int(round(100 * float(RSEL)))}"]
AR = {n: parts(n, DAYS) for n in ["sd"] + ARMS}; print("days", [d + 1 for d in DAYS], "arms", list(AR), flush=True)
d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; R_ = np.load(d_ + "R.npy", mmap_mode="r"); o_ = np.load(d_ + "off.npy"); w_ = [str(x) for x in np.load(d_ + "window_id.npy", allow_pickle=True)]
COV = {x: float(np.asarray(R_[o_[i]:o_[i + 1], :12, 2], float).mean()) for i, x in enumerate(w_)}
cols = {n: {} for n in AR}
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_ongoing" or m["day"] not in DAYS or m3v2.is_holdout(np.array([m["day"]]))[0]:
        continue
    kd = m3v2.kdist(m["day"]); Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); pl = FS.lgb_ongoing(Xw); cf = {k: cone_f(m, k) for k in kd}
    for nm, P in AR.items():
        cols[nm][w] = sum(pr * G.iou(og_pred((0.8 * (0.8 * P[(w, k)] + 0.2 * PT[w]) + 0.2 * pl) * cf[k], m, k, None, None), m) for k, pr in kd.items())
D = V.frame(cols).dropna(subset=list(cols)).reset_index(drop=True); D["cov"] = D.w.astype(str).map(COV); assert D["cov"].notna().all(); D.to_csv(OUT + f"g283_{PAR}.csv", index=False)
top = D[D["cov"] > D["cov"].quantile(2 / 3)].reset_index(drop=True)
for b in ARMS:
    full = m3v2.summary(D, "sd", b, "private"); s3 = m3v2.summary(top, "sd", b, "private")
    zf = full["t2_weighted"] / max(full["se"], 1e-12); z3 = s3["t2_weighted"] / max(s3["se"], 1e-12)
    print(f"{PAR} days, {b} - sd: all {len(D)} windows {full['t2_weighted']:+.5f} (SE {full['se']:.5f}, z {zf:+.2f}, changed {int((~np.isclose(D[b], D.sd)).sum())}) | "
          f"top third {len(top)} {s3['t2_weighted']:+.5f} (SE {s3['se']:.5f}, z {z3:+.2f})" + (f" | stage-2 line (all > 0 and z >= +1, top >= -1 SE): {full['t2_weighted'] > 0 and zf >= 1 and z3 >= -1}" if PAR == "even" else ""), flush=True)
