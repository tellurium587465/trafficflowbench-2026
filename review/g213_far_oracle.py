"""G213 (11/02, ledger #143, first stage): how much is there in the order of the far region of ongoing windows? Far = links more than 3 from the queue at T - 5. January, February (OOF q25_nomem s3, s4, G119 path, mask visibility): the probabilities of the far cells are kept as a set, and reassigned so that the true positives hold the largest ones (the oracle knows the order, not the amount). best_set on them against best_set on p. Line: the oracle's gain in S (0.15 x IoU) >= +0.002, then a ranker is worth building.
usage: TFB_M3VER=v3 py -B -X utf8 g213_far_oracle.py"""
import sys, pickle
R = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True
src = open(R + "g119_month_postproc.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
sys.argv = [sys.argv[0], "run"]; exec(compile(src, "g119", "exec"), globals())
O = [pickle.load(open(W + f"oof_q25_nomem_s{s}.pkl", "rb")) for s in (3, 4)]; rows = []
for mo in MONTHS:
    ks = np.flatnonzero(T.META["is_sel"] & (MON == mo) & (T.META["cond"] == "queue_ongoing"))
    for k in ks:
        k = int(k); lab, el = T.LAB[k]; y = lab.astype(bool); e = el.astype(bool); p_ = (O[0][k] + O[1][k]) / 2; lq = np.asarray(XN[OFFN[k]:OFFN[k + 1]][:, G.CI], float) == 1; n = len(lq); near = G.dil(lq, 3); far = np.tile(~near, (6, 1)) & e; q = p_.copy(); idx = np.flatnonzero(far.ravel())
        if len(idx):
            v = np.sort(q.ravel()[idx])[::-1]; o = np.argsort(-(y.ravel()[idx].astype(float) + 1e-6 * q.ravel()[idx])); x = q.ravel().copy(); x[idx[o]] = v; q = x.reshape(q.shape)
        a = iou(decode(p_), y, e, "queue_ongoing"); b = iou(decode(q), y, e, "queue_ongoing"); rows.append(dict(month=mo, panel=str(T.META["panel"][k]), a=a, b=b, ypos=int((y & far).sum())))
D = pd.DataFrame(rows); D["d"] = D.b - D.a; print("windows", len(D), "with far positives", int((D.ypos > 0).sum()), "base IoU", round(D.a.mean(), 4), "oracle gain IoU", round(D.d.mean(), 4), "S", round(D.d.mean() * 0.15, 5), "on the windows with far positives", round(D.d[D.ypos > 0].mean(), 4)); print(D.groupby("month").d.mean().round(4).to_dict())
