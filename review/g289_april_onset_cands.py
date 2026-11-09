"""G289 (11/05, ledger #235 stage 0 (a)): A16's April onset T+30 marginals and their place candidates. The probabilities are rebuilt
from the saved blend of the A16 build (work/t2/probs_repro_finA16tcone.pkl, the input of t2_onset_evidence.py) with the evidence rule as
there (B = 0.25, SKIP 0: links +-2 without queue in T-30 .. T+60 min on up to 3 earlier days, x B at T+30); identity: best_set of the
rebuilt probabilities equals the T+30 row of A16's submitted queue on every April onset window. Candidates: maximal runs of consecutive
links with p > 0.05 on the T+30 row (link order of the panel), mass = sum of p over the run. A's place = the run(s) holding A's selected
links. Counted: A's mass share < 0.65 and another run with share >= 0.25. Writes work/t2/g289_april.pkl. usage: py -B -X utf8 g289_april_onset_cands.py"""
import sys, pickle
import numpy as np, pandas as pd
sys.path.insert(0, r"C:/Users/user/tfb_a8/src")
from cachelib import load, T2_PANELS, asof
from common import REL
from t2_model import best_set
W = "C:/Users/user/tfb/work/t2/"; B = 0.25; TH = 0.05


def runs(p, th=TH):
    out, j = [], 0
    while j < len(p):
        if p[j] > th:
            e = j
            while e + 1 < len(p) and p[e + 1] > th:
                e += 1
            out.append((j, e)); j = e + 1
        else:
            j += 1
    return out


def cands(p_row, sel_row):
    """runs with mass and share; A's runs = those overlapping sel_row; returns (runs, masses, shares, a_share, best other index or None)"""
    rs = runs(p_row); ms = np.array([p_row[a:b + 1].sum() for a, b in rs]); tot = ms.sum()
    if not rs or tot <= 0:
        return rs, ms, ms, 0.0, None
    sh = ms / tot; mine = np.array([sel_row[a:b + 1].any() for a, b in rs]); a_sh = float(sh[mine].sum())
    oth = [i for i in range(len(rs)) if not mine[i]]; bo = max(oth, key=lambda i: sh[i]) if oth else None
    return rs, ms, sh, a_sh, bo


if __name__ == "__main__":
    probs = pickle.load(open(W + "probs_repro_finA16tcone.pkl", "rb")); Q = pd.read_parquet(W + "queue_repro_finA16t.parquet"); QI = Q.groupby("window_id", sort=False).indices
    out = {}; bad = 0
    for p in T2_PANELS:
        P = load(p, "private"); ti = {s: i for i, s in enumerate(P.times)}
        for r in pd.read_csv(REL / "task2" / p / "private" / "window_index.csv").itertuples():
            if r.condition != "queue_onset":
                continue
            t = ti[r.forecast_origin]; pb = probs[r.window_id].copy(); Pt = asof(P, t); q = (Pt.V <= Pt.vcut[None, :]) & np.isfinite(Pt.V)
            seen = np.zeros(P.n, bool); nd = 0
            for back in (1, 2, 3):
                dd = t - back * 288
                if dd - 6 >= 0:
                    seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
            if nd:
                ev = seen.copy()
                for o in (1, 2):
                    ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
                pb[-1] = pb[-1] * np.where(ev, 1.0, B)
            sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape); qs = Q.queue_pred.to_numpy()[QI[r.window_id]].reshape(pb.shape[0], P.n).astype(bool)
            bad += int(not np.array_equal(sel[-1], qs[-1])); rs, ms, sh, a_sh, bo = cands(pb[-1], sel[-1])
            out[r.window_id] = dict(p=pb[-1], sel=sel[-1], runs=rs, mass=ms, share=sh, a_share=a_sh, other=bo, flag=bool(a_sh < 0.65 and bo is not None and sh[bo] >= 0.25))
    pickle.dump(out, open(W + "g289_april.pkl", "wb")); fl = [w for w, v in out.items() if v["flag"]]
    print(f"April onset windows {len(out)} | identity best_set T+30 against A16's queue: {len(out) - bad} / {len(out)} equal | flagged (A share < 0.65, other >= 0.25): {len(fl)}", flush=True)
    for w in fl:
        v = out[w]; print(f"  {w}: A share {v['a_share']:.2f}, other share {v['share'][v['other']]:.2f}, runs {len(v['runs'])}, total mass {v['mass'].sum():.2f}, A selected {int(v['sel'].sum())} links", flush=True)
