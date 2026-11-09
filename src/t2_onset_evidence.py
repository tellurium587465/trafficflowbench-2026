"""Onset T+30 probabilities re-weighted by same-month causal evidence (10/10, CAUSAL_MAP P1; local check G4).
For an onset window with at least one earlier day in the month, links (+-2) with no queue in the same slot (T-30 .. T+60
min) on up to 3 earlier days get their T+30 probability multiplied by B; the others keep theirs. On the train months
this is neutral (B = 0.5..0.7: +-0.000, tfb/review/g4_onset_evidence.py) because the model's memory already matches the
month; in the test months the memory is stale and the month's own evidence is not. Ongoing windows are unchanged.
Then the expected-IoU set as in t2_blend_submit.

usage: python t2_onset_evidence.py <blended_probs_name> <out_queue_name> [B=0.5] [OGB=1.0] [SKIP=0]
       SKIP=1 (B1, 10/14): onset windows where NO link shows evidence (the release blanks hide the earlier days) are not
       scaled at all (the rule used to shrink every link by B there; twin v4 instrument: +0.0200, 3.0 SE)
       OGB: ongoing windows, links without same-month evidence (same slot T-90..T+60 on up to 3 earlier days, +-1 link)
       x OGB on all horizon steps (G11 nested CV: 0.85, +0.0007 on official-like ongoing windows, SE 0.0003)
       (reads work/t2/probs_<blended_probs_name>.pkl written by t2_blend_submit.py, writes work/t2/queue_<out>.parquet)
       env TFB_T2_WINDOWS=<id>[,<id>...] (10/22 E6, t2_window_entry.py): only those windows, and only their panel x split
       is loaded (the release may then hold that panel only, truncated after T)
"""
import os, sys, pickle, numpy as np, pandas as pd
from cachelib import load, T2_PANELS, asof
from common import REL, WORK
from t2_windows import F
from t2_model import best_set
import t2_final_spec as FS

W = WORK + "/"; OGTE = float(os.environ.get("TFB_T2_OGTE", "1")); PBS = {}
from t2_dex import DEX; OGDEX = os.environ.get("TFB_T2_OGDEX", "0") == "1"   # 11/04 ledger #216: the ongoing decode by the exact expected IoU (B12)
SELC = os.environ.get("TFB_T2_SELC", "0") == "1"; SELK = {}             # 11/05 ledger #283: TFB_T2_SELC=1 (A25 / B7c)
if SELC:
    import t2_selcond as SC
src, name = sys.argv[1], sys.argv[2]; B = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
OGB = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
SKIP = int(sys.argv[5]) if len(sys.argv) > 5 else 0; skipped = {"validation": 0, "private": 0}
probs = pickle.load(open(W + f"t2/probs_{src}.pkl", "rb"))
assert all(np.isfinite(v).all() for v in probs.values()), "non-finite blended probabilities"
WSEL = set(filter(None, os.environ.get("TFB_T2_WINDOWS", "").split(",")))
out = []; changed = {"validation": 0, "private": 0}; touched = {"validation": 0, "private": 0}
for split in ("validation", "private"):
    for p in T2_PANELS:
        if WSEL and not any(w[3:].startswith(f"{p}_{split}_") for w in WSEL):
            continue
        P = load(p, split); ti = {s: i for i, s in enumerate(P.times)}
        for r in pd.read_csv(REL / "task2" / p / split / "window_index.csv").itertuples():
            if WSEL and r.window_id not in WSEL:
                continue
            t = ti[r.forecast_origin]; pb = probs[r.window_id].copy()
            Pt = asof(P, t); q = (Pt.V <= Pt.vcut[None, :]) & np.isfinite(Pt.V)   # 10/22 E6: read boundary, rows < T only
            base = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
            if r.condition == "queue_onset":
                seen = np.zeros(P.n, bool); nd = 0
                for back in (1, 2, 3):
                    dd = t - back * 288
                    if dd - 6 >= 0:                                    # the split starts on the 1st: earlier days are this month
                        seen |= q[dd - 6:dd + 12].any(axis=0); nd += 1
                if nd and SKIP and not seen.any():
                    skipped[split] += 1                        # B1: no visible evidence at all -> leave the window as is
                elif nd:
                    ev = seen.copy()
                    for o in (1, 2):
                        ev[o:] |= seen[:-o]; ev[:-o] |= seen[o:]
                    pb[-1] = pb[-1] * np.where(ev, 1.0, B); touched[split] += 1
            elif r.condition == "queue_ongoing" and OGB != 1.0:
                seen = np.zeros(P.n, bool); nd = 0
                for back in (1, 2, 3):
                    dd = t - back * 288
                    if dd - 18 >= 0:
                        seen |= q[dd - 18:dd + 12].any(axis=0); nd += 1
                if nd:
                    ev = seen.copy(); ev[1:] |= seen[:-1]; ev[:-1] |= seen[1:]
                    pb = pb * np.where(ev, 1.0, OGB)[None, :]
            if r.condition == "queue_ongoing":                 # 11/01 ledger #125: TFB_T2_OGTE, default 1 = unchanged
                pb = FS.temper(pb, OGTE)
                if SELC:                                        # 11/05 ledger #283: conditioned on the official selection (t2_selcond, V11)
                    pb, kept = SC.apply(r.window_id, pb, p, split); SELK[r.window_id] = kept
                PBS[r.window_id] = pb
            sel = DEX(pb) if (OGDEX and r.condition == "queue_ongoing") else best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)   # 11/04 #216: B12
            changed[split] += int((sel != base).any())
            stamps = [P.times[t + s + 1] for s in range(F)]
            out.append(pd.DataFrame({"window_id": r.window_id, "timestamp": np.repeat(stamps, P.n),
                                     "link_id": np.tile(P.links, F), "queue_pred": sel.ravel().astype(int)}))
qq = pd.concat(out, ignore_index=True); qq.to_parquet(W + f"t2/queue_{name}.parquet", index=False)
if os.environ.get("TFB_T2_GATE2") == "1":                       # 11/01 ledger #146: the probabilities best_set saw, for t2_far_gate.py
    pickle.dump(PBS, open(W + f"t2/pbog_{name}.pkl", "wb"))
print(f"B={B} OGB={OGB} SKIP={SKIP} (skipped {skipped}): onset windows with earlier days {touched}, windows whose set changed {changed}, positives {int(qq.queue_pred.sum())}")
if SELC:
    print(f"selection-conditioned ongoing (#283): {len(SELK)} windows, kept samples median {np.median(list(SELK.values())) if SELK else 0:.0f} / 400, "
          f"< 10 kept (unchanged) {sum(v < 10 for v in SELK.values())}", flush=True)
