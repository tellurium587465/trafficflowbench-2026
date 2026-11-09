"""G342b (11/05, ledger #267b, measurement only): runs an existing June script on the 80 official June windows instead of the 330 replica
selector windows. The evaluator T (t2_eval_oof, cache 2_noT_af) gets the 80 windows appended (k = NWIN + i, META / labels from G342a's
g342_jun80.pkl), is_sel is cleared on every other June window and set on the 80; np.load of nn_data_afdem X / off returns the base arrays
with nn_test_jun80 appended. Text substitutions: junmh0 -> jun80 (the cvf probabilities of G342a), output names g321_jun / g329_jun ->
g342_jun80 (no existing output is overwritten). usage: py -B -X utf8 g342b_jun80_run.py <g321 | g329 | g328> (g321 first: g329's identity reads its onset csv)"""
import os, sys, pickle
os.environ["TFB_T2CACHE"] = "2_noT_af"; os.environ.setdefault("TFB_M3VER", "v3"); sys.dont_write_bytecode = True
import numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src"); R0 = "C:/Users/user/tfb/"; W = R0 + "work/t2/"; NN = "C:/tfb_data/tfb_nn/"
WHICH = sys.argv[1]; sys.argv = sys.argv[:1]
A = pickle.load(open(W + "g342_jun80.pkl", "rb")); rows = A["rows"]
_orig = np.load; XNEW = _orig(NN + "nn_test_jun80/X.npy"); ONEW = _orig(NN + "nn_test_jun80/off.npy")


class AugX:
    def __init__(s, b):
        s.b = b; s.n = b.shape[0]; s.shape = (b.shape[0] + XNEW.shape[0],) + b.shape[1:]

    def __getitem__(s, ix):
        if isinstance(ix, slice) and ix.start is not None and ix.start >= s.n:
            return XNEW[ix.start - s.n:ix.stop - s.n]
        return s.b[ix]


def _load(f, *a, **k):
    f_ = str(f).replace("\\", "/")
    if f_.endswith("nn_data_afdem/X.npy"):
        return AugX(_orig(f, *a, **k))
    if f_.endswith("nn_data_afdem/off.npy"):
        o = _orig(f, *a, **k); return np.concatenate([o, o[-1] + ONEW[1:]])
    return _orig(f, *a, **k)


np.load = _load
import t2_eval_oof as EO
T = EO.T; assert T.NWIN == A["NWIN"], (T.NWIN, A["NWIN"])
mon = np.array([str(d)[:7] for d in T.META["date"]]); T.META["is_sel"] = T.META["is_sel"] & (mon != "2030-06")
for key, val in (("panel", [r["panel"] for r in rows]), ("t", [r["t"] for r in rows]), ("cond", [r["cond"] for r in rows]), ("date", [r["date"] for r in rows]), ("is_eval", [False] * len(rows)), ("is_sel", [True] * len(rows))):
    T.META[key] = np.concatenate([T.META[key], np.array(val)])
T.LAB.extend([(r["lab"].astype(T.LAB[0][0].dtype), r["el"].astype(T.LAB[0][1].dtype)) for r in rows]); T.NWIN += len(rows)
print(f"evaluator augmented: NWIN {T.NWIN}, June is_sel {int((T.META['is_sel'] & np.char.startswith(T.META['date'].astype(str), '2030-06')).sum())}", flush=True)
# identity of the inputs / probabilities on the windows that match the evaluator
P0 = [pickle.load(open(W + f"probs_junmh0__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]; P1 = [pickle.load(open(W + f"probs_jun80__cvf_attnglob_s{s_}_fold2030-06.pkl", "rb")) for s_ in (0, 1)]
dd = [float(np.abs(P1[j][str(kn)] - P0[j][str(ko)]).max()) for kn, ko in A["match"].items() for j in (0, 1) if str(ko) in P0[j]]
print(f"identity: cvf probabilities on {len(dd) // 2} matched windows, max |diff| {max(dd) if dd else float('nan'):.2e}", flush=True)
src = open(R0 + "review/" + {"g321": "g321_june_signs", "g329": "g329_june_onset_dump", "g328": "g328_oracle_map"}[WHICH] + ".py", encoding="utf-8").read()
PRB_ = os.environ.get("TFB_JUN80_PROBS", "jun80")   # 11/05 #279: the fine-tuned probabilities (jun80ft) -> outputs g342_<that>_*
src = src.replace("junmh0", PRB_).replace("g321_jun_", f"g342_{PRB_}_").replace("g329_jun_onset_rows", f"g342_{PRB_}_onset_rows")
if WHICH == "g328":
    a = '(("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",)))'; assert src.count(a) == 1; src = src.replace(a, '(("June 2030", ("2030-06",)),)')
    a = "    D = pd.DataFrame(rows); b = D.base;"; assert src.count(a) == 1; src = src.replace(a, "    D = pd.DataFrame(rows); D.to_csv(W + 'g342_' + lab.split(' (')[0].replace(' / ', '_').replace(' ', '_') + '.csv', index=False); b = D.base;")
exec(compile(src, f"{WHICH}_jun80", "exec"))
