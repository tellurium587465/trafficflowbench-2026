"""How much of a test month's congestion sits where the train months never queue? (10/14; aggregate check on visible
cells only, no labels) Per block (structural, train months), the train share of congested cells; then the validation and
private months' visible congested cells (all day) by the block's train congestion rate. If a large
share of the test months' congestion is in blocks that were (almost) never congested in train, no site memory can
predict it - that part of the Task 2 gap needs a law, not a map. Also the reverse: train-congested blocks that go quiet.
usage: python new_sites.py [panels=ALL T2 panels]"""
import sys, numpy as np, pandas as pd, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.path.insert(0, r"C:/Users/user/tfb/twin")
from cachelib import load, T2_PANELS
from onset_shape3_lib import raw_matrices, blocks, split_cells

out = []
for p in (sys.argv[1].split(",") if len(sys.argv) > 1 else T2_PANELS):
    P = load(p, "train"); V, Q, _ = raw_matrices(P); bid = blocks(V, P); nb = bid.max() + 1
    ok = np.isfinite(V); cg = ok & (V <= P.vcut[None, :])
    rate_tr = np.array([cg[:, bid == j].sum() / max(ok[:, bid == j].sum(), 1) for j in range(nb)])
    li = {str(l): i for i, l in enumerate(P.links)}
    for sp in ("validation", "private"):
        c2, v2, q2 = split_cells(P, sp); m = v2 <= P.vcut[c2]
        b2 = bid[c2]; n_ok = np.bincount(b2, minlength=nb); n_cg = np.bincount(b2[m], minlength=nb)
        rate_sp = n_cg / np.maximum(n_ok, 1)
        tot = n_cg.sum()
        for lo, hi, nm in ((0, 0.001, "never(<0.1%)"), (0.001, 0.01, "rare(0.1-1%)"), (0.01, 1.01, "usual(>=1%)")):
            sel = (rate_tr >= lo) & (rate_tr < hi)
            out.append(dict(panel=p, split=sp, train_rate=nm, blocks=int(sel.sum()), share_of_test_congestion=round(float(n_cg[sel].sum() / max(tot, 1)), 3),
                            test_rate_in_these=round(float(n_cg[sel].sum() / max(n_ok[sel].sum(), 1)), 4), train_rate_in_these=round(float(rate_tr[sel].mean()) if sel.any() else np.nan, 4)))
        # rank agreement of block congestion rates (train vs test)
        act = (rate_tr > 0) | (rate_sp > 0)
        rho = pd.Series(rate_tr[act]).corr(pd.Series(rate_sp[act]), method="spearman")
        print(f"{p} {sp}: test congested cells {tot}, spearman(train rate, test rate) over active blocks {rho:.2f}", flush=True)
    load.cache_clear(); del V, Q
R = pd.DataFrame(out); R.to_csv(r"C:/Users/user/tfb/work/twin_new_sites.csv", index=False)
print(R.to_string(index=False))
print(R.groupby(["split", "train_rate"]).share_of_test_congestion.mean().round(3).to_string())
