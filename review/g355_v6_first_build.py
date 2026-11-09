"""G355 (11/05, ledger #279b data): training windows at the first origin of each run of origins qualifying under the identified official
selector V6 (#280: visible observation (Task 1 blank hidden; queue = finite & speed <= vcut & eligible); coverage of T-12 .. T-1 >= 0.7 on finite &
eligible (deviation: the training months' blank leaves visible coverage ~0.6),
a true queue in T+1 .. T+6; onset = no visible queue in T-12 .. T-1, no true queue in T+1 .. T+5, a true queue at T+6; ongoing = visible
history queue sum >= 2 with a link >= 2 and IoU(per link the last visible observation of T-12 .. T-1 held, true T+1 .. T+6 on P.E) <= 0.9;
same calendar day T-12 .. T+6), no spacing limit, both conditions, 2030-09 .. 2031-02. Inputs / labels as G350 -> NN/nn_data_v6first.
usage: [TFB_V11P=1 (#279c: all on the pre-mask layer, ffill T-55 .. T -> nn_data_v11pfirst) | TFB_V11=1 (#279b with V11: persistence ffill over T-55 .. T, -> nn_data_v11first)] py -B -X utf8 g355_v6_first_build.py"""
import os, sys, time
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
import numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load, T2_PANELS
from t2_windows import queue_matrix, iou, H, F
V11P = os.environ.get("TFB_V11P", "0") == "1"; V11 = V11P or os.environ.get("TFB_V11", "0") == "1"; OUTD = "nn_data_v11pfirst" if V11P else "nn_data_v11first" if V11 else "nn_data_v6first"
T0 = time.time(); NN = "C:/tfb_data/tfb_nn/"; MONTHS = ("2030-09", "2030-10", "2030-11", "2030-12", "2031-01", "2031-02")


def qualify_v6(P, Qm, inm):
    Vv = np.where(P.blank, np.nan, P.V); vq = np.isfinite(Vv) & (np.nan_to_num(Vv, nan=1e9) <= P.vcut[None, :]) & P.E; obs = np.isfinite(Vv) & P.E
    if V11P:   # #279c (#284): every condition on the pre-mask layer
        vq = Qm.copy(); obs = np.isfinite(P.V) & P.E
    cobs = np.isfinite(P.V) & P.E   # 11/05: the coverage gate on finite & eligible (the training months' Task 1 blank leaves visible coverage ~0.6 < 0.7, no window would qualify)
    any_t = Qm.any(axis=1); any_h = vq.any(axis=1); cnt = np.concatenate([np.zeros((1, P.n), np.int32), np.cumsum(vq, axis=0, dtype=np.int32)])
    ocnt = np.concatenate([[0], np.cumsum(cobs.sum(axis=1))]); qt = np.concatenate([[0], np.cumsum(any_t)]); qh = np.concatenate([[0], np.cumsum(any_h)])
    on = np.zeros(P.T, bool); og = np.zeros(P.T, bool)
    for t in np.flatnonzero(inm):
        if t < H or t >= P.T - F or P.times[t - H][:10] != P.times[t + F][:10] or (ocnt[t] - ocnt[t - H]) / (H * P.n) < 0.7 or not qt[t + F + 1] - qt[t + 1] > 0:
            continue
        if qh[t] - qh[t - H] == 0 and qt[t + F] - qt[t] == 0 and any_t[t + F]:
            on[t] = True
        else:
            hq = cnt[t] - cnt[t - H]
            if hq.sum() >= 2 and hq.max() >= 2:
                a0 = t - H + 1 if V11 else t - H; a1 = t + 1 if V11 else t   # V11 (#280b): the ffill window T-55 .. T
                seen = obs[a0:a1]; last = np.where(seen.any(axis=0), H - 1 - np.argmax(seen[::-1], axis=0), -1); ps = np.where(last >= 0, vq[a0:a1][np.maximum(last, 0), np.arange(P.n)], False)
                og[t] = iou(np.broadcast_to(ps, (F, P.n)), Qm[t + 1:t + F + 1], P.E[t + 1:t + F + 1]) <= 0.9
    return on, og


items = []
for p in T2_PANELS:
    P = load(p, "train"); Qm = queue_matrix(P); inm = np.isin(np.array([s[:7] for s in P.times]), MONTHS); on, og = qualify_v6(P, Qm, inm)
    for cond, q in (("queue_onset", on), ("queue_ongoing", og)):
        first = q & ~np.concatenate([[False], q[:-1]]); items += [(p, int(t), cond) for t in np.flatnonzero(first)]
        print(f"{p} {cond}: qualifying {int(q.sum())}, first origins {int(first.sum())} ({time.time() - T0:.0f}s)", flush=True)
    load.cache_clear()
print(f"V6 first-origin windows: {len(items)} (onset {sum(c == 'queue_onset' for _, _, c in items)}, ongoing {sum(c == 'queue_ongoing' for _, _, c in items)}), {time.time() - T0:.0f}s", flush=True)
src = open(r"C:/Users/user/tfb/review/g350_first_origin_build.py", encoding="utf-8").read(); src = src[src.index("from t2_model import PanelInputs"):].replace('o = NN + "nn_data_first/"', f'o = NN + "{OUTD}/"').replace("nn_data_first:", f"{OUTD}:")
exec(compile(src, "g350_build_part", "exec"))
