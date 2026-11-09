"""G183 (11/01, ledger #142 stage 0, strategy T2_DEEP_REVIEW 18): the look-back of the evidence rule stops at the first day of the split, and the official windows sit on the first days. Rows of the month before are released (748789). Count: births on days 1-3 covered by the last 3 days of the month before (same clock slot T-30 .. T+55, links +-2)
against births on days 4-6 covered by days 1-3 of the same month. Births as G119's month_events. March reads dev days only.
usage: py -B -X utf8 g183_month_edge.py"""
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
import numpy as np, pandas as pd
from cachelib import load, T2_PANELS

MO = {"2031-01": ("train", "2030-12"), "2031-02": ("train", "2031-01"), "2031-03": ("validation", "2031-02"), "2031-04": ("private", "2031-03")}; rows = []


def dil(s):
    e = s.copy()
    for o in (1, 2):
        e[o:] |= s[:-o]; e[:-o] |= s[o:]
    return e


def month_q(p, mo):
    sp = "train" if mo < "2031-03" else ("validation" if mo == "2031-03" else "private"); P = load(p, sp); m = np.char.startswith(P.times.astype(str), mo); V = P.V[m]; V = np.where(P.blank[m], np.nan, V) if sp == "train" else V; return (V <= P.vcut[None, :]) & np.isfinite(V)


for p in T2_PANELS:
    Q = {mo: month_q(p, mo) for mo in ("2030-12", "2031-01", "2031-02", "2031-03", "2031-04")}
    for mo, (_, prev) in MO.items():
        q = Q[mo]; anyq = q.any(axis=1); last = -10 ** 9; D = q.shape[0] // 288; Dp = Q[prev].shape[0] // 288
        for t in range(17, q.shape[0]):
            if anyq[t] and not anyq[t - 17:t].any() and t - last >= 24:
                last = t; d = t // 288; s_ = t % 288 - 6; Y = np.flatnonzero(q[t])
                if mo == "2031-03" and (d + 1) % 3 == 0:
                    continue
                def cov(M, days):
                    e = np.zeros(q.shape[1], bool)
                    for dd in days:
                        a = dd * 288 + s_; e |= M[max(a, 0):a + 18].any(axis=0)
                    return bool(dil(e)[Y].any())
                if d <= 2:
                    rows.append(dict(panel=p, month=mo, kind="edge", cov=cov(Q[prev], range(Dp - 3, Dp))))
                elif d <= 5:
                    rows.append(dict(panel=p, month=mo, kind="same", cov=cov(q, range(0, 3))))
    load.cache_clear(); print(p, len(rows), flush=True)
D = pd.DataFrame(rows); print(D.pivot_table(index="month", columns="kind", values="cov", aggfunc=["mean", "size"]).round(3).to_string()); a = D[D.kind == "edge"]["cov"].mean(); b = D[D.kind == "same"]["cov"].mean(); print(f"all: edge {a:.3f} same {b:.3f} ratio {a / b:.2f}")
