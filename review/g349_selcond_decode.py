"""G349 (11/05, ledger #278): ongoing decode conditioned on the official selection rule. G328's ongoing flows (M3 = A17's mix of G284 with
G132's gate; January / February = capB0 and the June 2030 replica = cvf through G119's flow with the far gate; mode jun80: the 80 official
June windows via G342b). Per window x k: from the marginals the decode sees (q = temperature 0.8 of the pipeline's probabilities, 6 x n)
NS = 400 joint samples: the box (lq = last visible queue) comonotone (one uniform per sample: cell on iff u < q), every other cell
independent. Keep the samples that satisfy the selection rule (organizer 745700 allows conditioning on it):
  (a) IoU(persistence of the T-5 state over the 6 horizon rows, sample) <= 0.9;
  (b) if the T-5 origin satisfies the other conditions (>= 360 min after the previous official window of the panel (official windows
      only; elsewhere taken as met), visible coverage of T-65 .. T-10 >= 0.7, a queue visible in T-65 .. T-10): IoU(persistence of the
      T-10 state over rows T .. T+25, [row T, sample rows T+5 .. T+25]) > 0.9. Row T: "cond" = the visible observation at T (745700: the T
      row may be used; its unobserved cells take the sample's T+5 value), "cond_noT" = the sample's T+5 row.
The decode on the kept samples' marginals (fewer than 10 kept -> the base). States are the visible queue (finite and <= vcut; the training
months' Task 1 blank hidden). Line (#278): official June points > 0 and z >= +1 (cluster bootstrap panel x day), M3 judge z >= -1.64, April
rows change (checked after). usage: TFB_M3VER=v3 [TFB_SC_BLOB=1 (#278b, outputs g349b_*) | TFB_SC_V6=1 (#278c, outputs g349c_*) | TFB_SC_V11=1 (#283, the production module t2_selcond (V6 since 17:11, NS 2000), outputs g349m<seed>_*) [TFB_SC_SEED] [TFB_SC_BAPR=1 (#289, B's April post-processing, outputs g349bapr_*)]] py -B -X utf8 g349_selcond_decode.py <all | jun80>"""
import os, sys, pickle
MODE = sys.argv[1]; R0 = "C:/Users/user/tfb/"; W = R0 + "work/t2/"
if MODE == "jun80":
    g = open(R0 + "review/g342b_jun80_run.py", encoding="utf-8").read(); g = g[:g.index("src = open(")]; sys.argv = ["g342b", "none"]; exec(compile(g, "g342b_head", "exec"))
sys.argv = sys.argv[:1]
import numpy as np
NS = 400; SEED_ = int(os.environ.get("TFB_SC_SEED", "278")); RNG = np.random.default_rng(SEED_)


def iou_b(S, P_):                     # S (ns, r, n) bool, P_ (r, n) bool -> (ns,)
    inter = (S & P_[None]).sum(axis=(1, 2)); uni = (S | P_[None]).sum(axis=(1, 2)); return np.where(uni == 0, 1.0, inter / np.maximum(uni, 1))


BLOB = os.environ.get("TFB_SC_BLOB", "0") == "1"   # #278b: outside the box, per row, runs of cells with q > 0.1 share one uniform


def blob_uniform(q, lq):
    n = q.shape[1]; U = RNG.random((NS, q.shape[0], n))
    for r in range(q.shape[0]):
        m = (q[r] > 0.1) & ~lq; j = 0
        while j < n:
            if m[j]:
                e_ = j
                while e_ + 1 < n and m[e_ + 1]:
                    e_ += 1
                U[:, r, j:e_ + 1] = RNG.random((NS, 1)); j = e_ + 1
            else:
                j += 1
    return U


V6 = os.environ.get("TFB_SC_V6", "0") == "1"   # #278c: the identified selector V6 (#280) as the condition; outputs g349c_*
C6_ = [None]   # (visible queue, visible eligible, t, lo) of the current window


def ffill6(vq, obs, s_):
    seen = obs[s_ - 12:s_]; last = np.where(seen.any(axis=0), 11 - np.argmax(seen[::-1], axis=0), -1)
    return np.where(last >= 0, vq[s_ - 12:s_][np.maximum(last, 0), np.arange(vq.shape[1])], False)


def cond_v6(q, lq):
    """(a) IoU(ffill persistence at T, sample T+5 .. T+30) <= 0.9; (b) every origin s in [max(lo, T-6), T-1] meeting V6's other conditions
    (visible coverage of s-12 .. s-1 >= 0.7, a visible history queue sum >= 2 with a link >= 2) must not qualify: its horizon s+1 .. s+6 =
    visible rows (< T; IoU on the visible eligible cells), row T drawn from the T+5 probabilities, rows > T the sample; qualifies = a queue in
    that horizon and IoU(ffill persistence at s, horizon) <= 0.9. Samples: the box comonotone, other runs with q > 0.1 one uniform per run
    and row (#278b); row T drawn with the same scheme. "cond_noT" carries the base q and the number of (b) origins checked."""
    vq, obs, t, lo = C6_[0]; n = q.shape[1]; q7 = np.vstack([q[:1], q]); u = RNG.random((NS, 1, 1)); ind = blob_uniform(q7, lq)
    S7 = np.where(lq[None, None, :], u < q7[None], ind < q7[None]); S = S7[:, 1:]
    keep = iou_b(S, np.broadcast_to(ffill6(vq, obs, t), (6, n))) <= 0.9; nb = 0
    for s_ in range(max(lo, t - 6), t):
        if obs[s_ - 12:s_].mean() < 0.7:
            continue
        hq = vq[s_ - 12:s_].sum(axis=0)
        if hq.sum() < 2 or hq.max() < 2:
            continue
        ps = ffill6(vq, obs, s_); inter = np.zeros(NS); uni = np.zeros(NS); anyq = np.zeros(NS, bool)
        for r in range(s_ + 1, s_ + 7):
            if r <= t - 1:
                e_ = obs[r]; y_ = vq[r] & e_; p_ = ps & e_; inter += (p_ & y_).sum(); uni += (p_ | y_).sum(); anyq |= bool(y_.any())
            else:
                y_ = S7[:, r - t]; inter += (ps[None] & y_).sum(axis=1); uni += (ps[None] | y_).sum(axis=1); anyq |= y_.any(axis=1)
        io = np.where(uni == 0, 1.0, inter / np.maximum(uni, 1)); keep &= ~(anyq & (io <= 0.9)); nb += 1
    return {"cond": (S[keep].mean(axis=0) if keep.sum() >= 10 else q, int(keep.sum())), "cond_noT": (q, nb)}


V11 = os.environ.get("TFB_SC_V11", "0") == "1"   # #283: the production module t2_selcond.condition (V11), outputs g349d_*


def cond_q(q, lq, st5, st10, rowT, obsT, b_applies):
    if V11:
        sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); import t2_selcond as SC_
        vq, obs, t, lo = C6_[0]; q2, k2 = SC_.condition(q, vq, obs, t, lo, RNG); return {"cond": (q2, k2), "cond_noT": (q, 0)}
    if V6:
        return cond_v6(q, lq)
    n = q.shape[1]; u = RNG.random((NS, 1, 1)); ind = blob_uniform(q, lq) if BLOB else RNG.random((NS, 6, n)); S = np.where(lq[None, None, :], u < q[None], ind < q[None])
    keep = iou_b(S, np.broadcast_to(st5, (6, n))) <= 0.9; out = {}
    for nm, use_T in (("cond", True), ("cond_noT", False)):
        k_ = keep.copy()
        if b_applies:
            T_ = np.where(obsT[None, :], rowT[None, :], S[:, 0]) if use_T else S[:, 0]; R_ = np.concatenate([T_[:, None, :], S[:, :5]], axis=1)
            k_ &= iou_b(R_, np.broadcast_to(st10, (6, n))) > 0.9
        out[nm] = (S[k_].mean(axis=0) if k_.sum() >= 10 else q, int(k_.sum()))
    return out


def states(fin, qd, t):
    st5, st10 = qd[t - 1], qd[t - 2]; cov = float(fin[t - 13:t - 1].mean()); hq = bool(qd[t - 13:t - 1].any()); return st5, st10, qd[t], fin[t], cov >= 0.7 and hq


s = open(R0 + "review/g328_oracle_map.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:70], s.count(a))
    s = s.replace(a, b)


sub('OG = ("O_box", "O_tail", "O_new", "O_box+tail", "O_map", "O_map_any");', 'OG = ("cond", "cond_noT", "kept", "kept_noT");')
sub('    y = (m["y"].T.astype(bool) & m["m"].T.astype(bool)); r = dict(w=w, base=0.0, **dict.fromkeys(OG, 0.0))',
    '    y = (m["y"].T.astype(bool) & m["m"].T.astype(bool)); r = dict(w=w, cond_="queue_ongoing", panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, base=0.0, **dict.fromkeys(OG, 0.0))\n'
    '    fin_, qd_ = VV_[m["panel"]]; st5, st10, rT, oT, bap = states(fin_, qd_, int(m["t"])); C6_[0] = VO_[m["panel"]] + (int(m["t"]), int(m["t"]) - 6)')
sub('        for nm, q in og_variants(p, y, sets, mp, mpa).items():\n            r[nm] += pr * G.iou(g132.gate(bs(tmp(q, 0.8)), ex, nr), m)\n',
    '        for nm, (q2, nk) in cond_q(tmp(p, 0.8), lq, st5, st10, rT, oT, bap).items():\n            r[nm] += pr * G.iou(g132.gate(bs(q2), ex, nr), m); r["kept" + nm[4:]] += pr * nk\n')
sub('    P = cload(p, "validation"); HM[p] = head_rows(P.V, P.vcut); cload.cache_clear()',
    '    P = cload(p, "validation"); HM[p] = head_rows(P.V, P.vcut); VV_[p] = (np.isfinite(P.V), np.isfinite(P.V) & (np.nan_to_num(P.V, nan=1e9) <= P.vcut[None, :])); Vb_ = np.where(P.blank, np.nan, P.V); VO_[p] = (np.isfinite(Vb_) & (np.nan_to_num(Vb_, nan=1e9) <= P.vcut[None, :]) & P.E, np.isfinite(Vb_) & P.E); cload.cache_clear()')
sub('HM = {}', 'HM = {}; VV_ = {}; VO_ = {}')
sub('                sets = og_sets(lq); mp, mpa = maps(Hm, (t - r0) // 288, t % 288); yt = y & e; r = dict(w=int(k), base=0.0, **dict.fromkeys(OG, 0.0))',
    '                sets = og_sets(lq); mp, mpa = maps(Hm, (t - r0) // 288, t % 288); yt = y & e; r = dict(w=int(k), panel=p, day=dom, base=0.0, **dict.fromkeys(OG, 0.0))\n'
    '                Vv_ = np.where(P.blank, np.nan, P.V); fin_ = np.isfinite(Vv_); qd_ = fin_ & (np.nan_to_num(Vv_, nan=1e9) <= P.vcut[None, :]); st5, st10, rT, oT, bap = states(fin_, qd_, t)\n'
    '                bap = bap and (t - 1 - PREV_.get(int(k), -10 ** 9) >= 72); C6_[0] = (qd_ & P.E, fin_ & P.E, t, max(PREV_.get(int(k), -10 ** 9) + 72, t - 6))')
sub('                    for nm, q in og_variants(pb, yt, sets, mp, mpa).items():\n                        r[nm] += pr * h.iou(h.far_gate(c, h.decode(tmp(q, 0.8)), t, kk, lq), y, e, "queue_ongoing")\n',
    '                    for nm, (q2, nk) in cond_q(tmp(pb, 0.8), lq, st5, st10, rT, oT, bap).items():\n                        r[nm] += pr * h.iou(h.far_gate(c, h.decode(q2), t, kk, lq), y, e, "queue_ongoing"); r["kept" + nm[4:]] += pr * nk\n')
sub("    D = pd.DataFrame(rows); b = D.base;", f"    D = pd.DataFrame(rows); D.to_csv(W + 'g349{("m" + str(SEED_) + os.environ.get("TFB_JUN80_PROBS", "jun80")[5:]) if V11 else ("c" if V6 else ("b" if BLOB else ""))}_{MODE}_' + lab.split(' (')[0].replace(' / ', '_').replace(' ', '_') + '.csv', index=False); b = D.base;")
PREV = {}
if MODE == "jun80":
    PRB_ = os.environ.get("TFB_JUN80_PROBS", "jun80")   # 11/05: e.g. jun80v6ft (the fine-tuned cvf, #279b stage 2 overlap)
    s = s.replace("junmh0", PRB_).replace("g321_jun_", "g342_jun80_").replace("g329_jun_onset_rows", "g342_jun80_onset_rows")
    sub('(("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",)))', '(("June 2030", ("2030-06",)),)')
    rws = A["rows"]
    for r_ in rws:   # the previous official window (either condition) of the same panel
        prv = [x["t"] for x in rws if x["panel"] == r_["panel"] and x["t"] < r_["t"]]; PREV[r_["k"]] = max(prv) if prv else -10 ** 9
s = s.replace("\nfor w, v in D8[\"M\"].items():", "\nPREV_ = PREV\nfor w, v in D8[\"M\"].items():", 1)
if os.environ.get("TFB_SC_BAPR", "0") == "1":   # #289: B's April post-processing (raw mixed probabilities, temperature 1.0, no cone / seen x0.85 / far gate)
    a_ = '                    r["base"] += pr * h.iou(h.far_gate(c, h.decode(tmp(pb, 0.8)), t, kk, lq), y, e, "queue_ongoing")\n'
    assert s.count(a_) == 1
    s = s.replace(a_, '                    s0_ = h.decode(p_); r["base"] += pr * h.iou(s0_, y, e, "queue_ongoing")\n')
    a_ = ('                    for nm, (q2, nk) in cond_q(tmp(pb, 0.8), lq, st5, st10, rT, oT, bap).items():\n'
          '                        r[nm] += pr * h.iou(h.far_gate(c, h.decode(q2), t, kk, lq), y, e, "queue_ongoing"); r["kept" + nm[4:]] += pr * nk\n')
    assert s.count(a_) == 1
    s = s.replace(a_, '                    for nm, (q2, nk) in cond_q(p_, lq, st5, st10, rT, oT, bap).items():\n'
                      '                        s1_ = h.decode(q2); r[nm] += pr * h.iou(s1_, y, e, "queue_ongoing"); r["kept" + nm[4:]] += pr * nk\n'
                      '                        if nm == "cond":\n'
                      '                            ad_ = s1_ & ~s0_; cj_ = np.flatnonzero(ad_.any(axis=0)); dl_ = [int(np.abs(np.flatnonzero(lq) - j).min()) if lq.any() else 999 for j in cj_]\n'
                      '                            r["add_far4"] = r.get("add_far4", 0.0) + pr * float(sum(ad_[:, j].sum() for j, d_ in zip(cj_, dl_) if d_ >= 4)); r["add_all"] = r.get("add_all", 0.0) + pr * float(ad_.sum())\n')
    s = s.replace("D.to_csv(W + 'g349", "D.to_csv(W + 'g349bapr_' + 'g349", 1)
exec(compile(s, "g328_g349", "exec"))
