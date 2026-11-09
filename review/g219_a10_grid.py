"""G219 (11/02, ledger #185): the constants of the ongoing flow, read again on A10's flow (row-T networks, same-day X1). One constant at a time, the two values around the production one: tfm 0.1, 0.3 (0.2); LightGBM 0.1, 0.3 (0.2); X3a 0.1, 0.3 (0.2); cone factor 0.35, 0.7 (0.5); seen 0.75, 0.95 (0.85); temperature 1.0, 1.4 (1.2); far gate D 2, 4 (3). Each constant's value is chosen on odd days, read on even days (judge's April composition). Dev windows only; G175's source with these columns, nothing of G175 changes.
usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g219_a10_grid.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g175_bundle_read.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


GRID = dict(tw=(0.1, 0.3), la=(0.1, 0.3), xw=(0.1, 0.3), cn=(0.35, 0.7), se=(0.75, 0.95), te=(1.0, 1.4), dd=(2, 4))
ARMS = [(k, v) for k, vs in GRID.items() for v in vs]; NAMES = [f"{k}{v}" for k, v in ARMS]
sub('SD = {}\n', 'SD = {}; SDT = {}\n')
sub('    CC = ("A", "bundle", "B", "Bc", "B2", "Bc2"); cols', '    for s in (0, 1):\n        for f in glob.glob(W + f"x1_arms/{parts}/sd__q24t_attnglob_s{s}__*.pkl"):\n            for key, v in pickle.load(open(f, "rb")).items():\n                SDT.setdefault(key, []).append(v)\n    CC = ("A", "bundle", "B", "Bc", "B2", "Bc2", "a10") + tuple(%r); cols' % (NAMES,))
sub('sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)', '''xt = sum(SDT[(w, k)]) / 2; nrs = g132.nears(lq, P.n)
            def flow(tw=0.2, la=0.2, xw=0.2, cn=0.5, se=0.85, te=1.2, dd=3):
                cf_ = np.where(ex[None, :], 1.0, np.where(cone < 0.999, cn, 1.0)); sw_ = np.where(G.dil(s, 1), 1.0, se)[None, :] if nd else 1.0; pb = ((1 - la) * ((1 - xw) * ((1 - tw) * xt + tw * pt) + xw * px) + la * pl) * cf_ * sw_; pb = sg(lg(pb) / te); return G.iou(g132.gate(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), ex, nrs[dd]), m)
            sc["a10"] += pr * flow()
            for k_, v_ in %r:
                sc[f"{k_}{v_}"] += pr * flow(**{k_: v_})
            sc["bundle"] += pr * G.iou(dec(tmp((0.8 * ((1 - X3) * (0.8 * xs + 0.2 * pt) + X3 * px) + 0.2 * pl) * cf * sw)), m)''' % (ARMS,))
sub('"g175_dev.csv"', '"g219_dev.csv"', 2)
sub('for a, b, nm in (("A", "bundle", "A side"), ("B", "Bc", "B side"), ("B2", "Bc2", "B side with X1 beta 0.25")):\n        s = m3v2.summary(D, a, b, "private"); sm =', '''odd = D.day.to_numpy() %% 2 == 0; Do, De = D[odd].reset_index(drop=True), D[~odd].reset_index(drop=True)
    for k_, vs in %r.items():
        o = {v_: m3v2.summary(Do, "a10", f"{k_}{v_}", "private", n_boot=500, n_draw=500)["t2_weighted"] for v_ in vs}; bv = max(o, key=o.get); e_ = m3v2.summary(De, "a10", f"{k_}{bv}", "private", n_boot=1000, n_draw=500); a_ = m3v2.summary(D, "a10", f"{k_}{bv}", "private", n_boot=1000, n_draw=500); print(k_, "odd", {v_: round(x_, 4) for v_, x_ in o.items()}, f"chosen {bv} | even {e_['t2_weighted']:+.4f} (SE {e_['se']:.4f}) | all {a_['t2_weighted']:+.4f} (SE {a_['se']:.4f})", flush=True)
    sys.exit()
    for a, b, nm in (("A", "bundle", "A side"),):
        s = m3v2.summary(D, a, b, "private"); sm =''' % (GRID,))
sys.argv = [sys.argv[0], "dev"]; exec(compile(s, "g175g", "exec"))
