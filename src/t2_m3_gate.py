"""M3 calibration gate (pre-registered in STATE.md 10/22 10:20): do the March pseudo windows reproduce the SIGN of five
known LB differences? Probabilities from t2_nn_cpu_infer.py on NN/nn_test_m3 (probs_m3__<tag>.pkl); post-processing
re-implemented from t2_onset_evidence.py / t2_far_gate.py with two instrument changes:
  - evidence uses at most k earlier days (k per window from the official distribution, m3_meta), on top of the
    production cap of 3 days;
  - trap 4: on each evidence day, every onset event of the panel within +-60 min of the window's clock time hides
    that day's rows e+1..e+18 (the official onset window's blank would have hidden them).
Scoring: IoU over observed eligible cells only. T2 = mean of the onset and ongoing condition means (equal_mean).
Pairs (sign expected from the LB):
  1 memory   : NN parts q15f+q19f (s40)  -> q24f (s42)          LB +0.0032  (T2 of the full pipeline)
  2 evidence : q24f onset, no rule        -> x0.25              LB +0.0032  (onset; s45 used x0.5)
  3 far gate : q24f ongoing, no removal   -> gate D=3 + contig  LB +0.0031  (ongoing; s34 + s40)
  4 sim      : q24f ongoing               -> q43f+q46f          LB +0.003..0.004 (ongoing; s60 -> s70)
  5 B1       : x0.25                       -> skip when no evidence  LB -0.0021 (onset; 3 windows)
Pass: >= 4 of 5 signs agree. Paired differences with cluster bootstrap SE (panel x day).
usage: python t2_m3_gate.py [seeds=0]"""
import os, sys, pickle, numpy as np, pandas as pd
from cachelib import load, T2_PANELS
from common import NN
from t2_model import best_set

SEEDS = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1 else "0").split(",")]
W = "C:/Users/user/tfb/work/t2/"
VER = os.environ.get("TFB_M3VER", "")                         # 10/23: "v3" -> the v3 window set (m3v3_meta.pkl, nn_test_m3v3)
PRE = f"m3{VER}"                                                # probability files: probs_<PRE>__<tag>.pkl
M = pickle.load(open(W + f"m3{VER}_meta.pkl", "rb")); META = {m["window_id"]: m for m in M["meta"]}; EVENTS = M["events"]
names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; CI = names.index("last_queue")
X = np.load(f"{NN}/nn_test_m3{VER}/X.npy", mmap_mode="r"); OFF = np.load(f"{NN}/nn_test_m3{VER}/off.npy")
WID = [str(w) for w in np.load(f"{NN}/nn_test_m3{VER}/window_id.npy", allow_pickle=True)]; POS = {w: i for i, w in enumerate(WID)}


def blend(spec):
    parts = []
    for fam, w in spec:
        for s in SEEDS:
            parts.append((w, pickle.load(open(W + f"probs_{PRE}__{fam}_s{s}.pkl", "rb"))))
    tw = sum(w for w, _ in parts)
    return {k: sum(w * pr[k] for w, pr in parts) / tw for k in parts[0][1]}


CFG5 = lambda f: [(f + "_attnglob", 2), (f + "_attnwp05", 2), (f + "_attn1", 2), (f + "_drop1", 1), (f + "_r13", 1)]
MEM = blend([("q15f_attnglob", 2), ("q15f_attnwp05", 2), ("q15f_attn1", 2), ("q19f_r13", 1), ("q19f_drop1", 1)])
NOMEM = blend(CFG5("q24f"))
SIM = blend(CFG5("q43f") + CFG5("q46f"))
NEG = blend(CFG5("q32f"))                                    # 10/22 negative control: s49 (time / weekday kept), LB -0.0051

PC = {}
def panel(p):
    if p not in PC:
        P = load(p, "validation"); PC[p] = (P, (P.V <= P.vcut[None, :]) & np.isfinite(P.V))
    return PC[p]


def seen_rows(p, t, k, lo, hi, cap=3):
    """links with an observed queue in rows dd+lo..dd+hi-1 on the earlier days back = 1..min(cap, k) (trap 4 applied)"""
    P, q = panel(p); s = np.zeros(P.n, bool); nd = 0
    for back in range(1, min(cap, k) + 1):
        dd = t - back * 288
        if dd + lo < 0:
            break
        blk = q[dd + lo:dd + hi].copy()
        for e in EVENTS[p][np.abs(EVENTS[p] - dd) <= 12]:          # trap 4: hide that day's official-like blank
            a, b = max(e + 1, dd + lo), min(e + 19, dd + hi)
            if a < b:
                blk[a - (dd + lo):b - (dd + lo)] = False
        s |= blk.any(axis=0); nd += 1
    return s, nd


def dil(s, r):
    e = s.copy()
    for o in range(1, r + 1):
        e[o:] |= s[:-o]; e[:-o] |= s[o:]
    return e


def onset_pred(p_, m, rule):
    pb = p_.copy()
    if rule != "none":
        s, nd = seen_rows(m["panel"], m["t"], m["k"], -6, 12)
        if nd and not (rule == "skip" and not s.any()):
            pb[-1] = pb[-1] * np.where(dil(s, 2), 1.0, 0.25)
    return best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)


def ongoing_pred(p_, m, gate):
    s, nd = seen_rows(m["panel"], m["t"], m["k"], -18, 12)
    pb = p_ * np.where(dil(s, 1), 1.0, 0.85)[None, :] if nd else p_.copy()
    sel = best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape)
    return far_gate(sel, m) if gate else sel


def far_gate(sel, m):
    """t2_far_gate.py (D=3, CONTIG=1) on a decoded ongoing set, with the k-day limit and trap 4 for earlier days"""
    P, q = panel(m["panel"]); t = m["t"]; d0 = t - (t % 288) + 60
    ev = q[d0:t].any(axis=0) if t > d0 else np.zeros(P.n, bool)
    s2, _ = seen_rows(m["panel"], t, m["k"], -18, 12)
    ev = dil(ev | s2, 2)
    i = POS[m["window_id"]]; lq = np.asarray(X[OFF[i]:OFF[i + 1], CI], dtype=float) == 1
    near = np.zeros(P.n, bool)
    for j in np.flatnonzero(lq):
        near[max(0, j - 3):j + 4] = True
    far = ~near; C = np.zeros_like(sel)
    for r_ in range(sel.shape[0]):                                   # contiguous growth of the visible queue is kept
        j = 0
        while j < sel.shape[1]:
            if sel[r_, j]:
                e = j
                while e + 1 < sel.shape[1] and sel[r_, e + 1]:
                    e += 1
                if near[j:e + 1].any():
                    C[r_, j:e + 1] = True
                j = e + 1
            else:
                j += 1
    drop = sel & far[None, :] & ~ev[None, :] & ~C
    return sel & ~drop


LABEL = os.environ.get("TFB_M3_LABEL", "L0")                   # 10/23 v3 §2.3: "L1" = the noise-reduced labels (y1, m1 in the meta)


def iou(sel, m):
    yk, mk_ = ("y1", "m1") if LABEL == "L1" and "y1" in m else ("y", "m")
    y, mk = m[yk].T.astype(bool), m[mk_].T.astype(bool)               # stored (n, 6) like build_nn2 labels
    if m["cond"] == "queue_onset":
        sel, y, mk = sel[-1:], y[-1:], mk[-1:]                       # onset is decoded and scored at T+30
    u = ((sel | y) & mk).sum(); return 1.0 if u == 0 else float((sel & y & mk).sum() / u)


if __name__ == "__main__":
    rows = []
    for w in WID:
        m = META[w]; r = dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], cl=f"{m['panel']}_{m['day']}", band="0.5-0.6" if m["cov"] < 0.6 else "0.6-0.7", morning=m["morning"])
        if m["cond"] == "queue_onset":
            r["mem_full"] = iou(onset_pred(MEM[w], m, "cur"), m); r["nomem_full"] = iou(onset_pred(NOMEM[w], m, "cur"), m)
            r["ev_none"] = iou(onset_pred(NOMEM[w], m, "none"), m); r["ev_cur"] = r["nomem_full"]
            r["b1_skip"] = iou(onset_pred(NOMEM[w], m, "skip"), m); r["neg_full"] = iou(onset_pred(NEG[w], m, "cur"), m)
        else:
            r["mem_full"] = iou(ongoing_pred(MEM[w], m, True), m); r["nomem_full"] = iou(ongoing_pred(NOMEM[w], m, True), m)
            r["gate_off"] = iou(ongoing_pred(NOMEM[w], m, False), m); r["gate_on"] = r["nomem_full"]
            r["sim_full"] = iou(ongoing_pred(SIM[w], m, True), m); r["neg_full"] = iou(ongoing_pred(NEG[w], m, True), m)
        rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(W + f"m3_gate_s{'-'.join(map(str, SEEDS))}.csv", index=False); rng = np.random.default_rng(0)


    def paired(g, a, b):
        u = g.cl.unique(); idx = {x: np.flatnonzero(g.cl.to_numpy() == x) for x in u}; d = (g[b] - g[a]).to_numpy()
        se = np.std([d[np.concatenate([idx[x] for x in rng.choice(u, len(u))])].mean() for _ in range(1000)])
        return d.mean(), se


    on, og = D[D.cond == "queue_onset"], D[D.cond == "queue_ongoing"]
    res = []
    # 1 memory: T2 = mean of the two condition means -> difference = 0.5 (onset diff + ongoing diff)
    d1o, s1o = paired(on, "mem_full", "nomem_full"); d1g, s1g = paired(og, "mem_full", "nomem_full")
    res.append(("1 memory q15f+q19f -> q24f (T2)", 0.5 * (d1o + d1g), 0.5 * np.hypot(s1o, s1g), +1))
    d6o, s6o = paired(on, "nomem_full", "neg_full"); d6g, s6g = paired(og, "nomem_full", "neg_full")
    for name, g, a, b, sign in (("2 onset evidence x0.25", on, "ev_none", "ev_cur", +1), ("3 far gate (ongoing)", og, "gate_off", "gate_on", +1),
                                ("4 sim ongoing q24f -> q43f+q46f", og, "nomem_full", "sim_full", +1), ("5 B1 skip (onset)", on, "ev_cur", "b1_skip", -1)):
        d, se = paired(g, a, b); res.append((name, d, se, sign))
    res.append(("6 NEG time back q24f -> q32f (T2)", 0.5 * (d6o + d6g), 0.5 * np.hypot(s6o, s6g), -1))
    print(f"M3 windows: onset {len(on)} (morning {int(on.morning.sum())}), ongoing {len(og)}; seeds {SEEDS}")
    agree = 0
    for name, d, se, sign in res:
        ok = np.sign(d) == sign; agree += int(ok)
        print(f"  {name:36s} {d:+.4f} (SE {se:.4f}, {d / max(se, 1e-9):+.1f} SE)  LB sign {'+' if sign > 0 else '-'}  -> {'agree' if ok else 'DISAGREE'}")
    print(f"signs agreeing: {agree}/6 -> {'PASS' if agree >= 5 else 'FAIL'} (10/22: 6 signs incl. the negative control, pass >= 5/6)")
    from m3_stats import summary, fmt                         # PLAN 2.4 items 3-4: production weights, P(B > A) over 80-window draws
    for name, sub, a, b in (("1 memory (T2)", D, "mem_full", "nomem_full"), ("2 evidence (onset)", on, "ev_none", "ev_cur"),
                            ("3 far gate (ongoing)", og, "gate_off", "gate_on"), ("4 sim (ongoing)", og, "nomem_full", "sim_full"),
                            ("5 B1 (onset)", on, "ev_cur", "b1_skip"), ("6 NEG (T2)", D, "nomem_full", "neg_full")):
        print("  " + fmt(name, summary(sub, a, b)), flush=True)
    for band, gb in D.groupby("band"):                        # effect by history-coverage band (M3 windows are all < 0.7)
        o_, g_ = gb[gb.cond == "queue_onset"], gb[gb.cond == "queue_ongoing"]
        print(f"  band {band}: onset {len(o_)} ongoing {len(g_)} | 2 ev {paired(o_, 'ev_none', 'ev_cur')[0]:+.4f} 3 gate {paired(g_, 'gate_off', 'gate_on')[0]:+.4f} "
              f"4 sim {paired(g_, 'nomem_full', 'sim_full')[0]:+.4f} 5 B1 {paired(o_, 'ev_cur', 'b1_skip')[0]:+.4f} "
              f"1 mem {0.5 * (paired(o_, 'mem_full', 'nomem_full')[0] + paired(g_, 'mem_full', 'nomem_full')[0]):+.4f} "
              f"6 neg {0.5 * (paired(o_, 'nomem_full', 'neg_full')[0] + paired(g_, 'nomem_full', 'neg_full')[0]):+.4f}")
