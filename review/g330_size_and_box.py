"""G330 (11/05, ledgers #263 stage 0 and #264, measurement only), on G328's three instruments with G328's helpers.
#263 stage 0 (onset, the T+30 row): per window and k draw, k_bs = the scored cells best_set selects, k = the true scored cells; the
distribution of k_bs - k, and O_size's gain (top k among the scored cells) split by (a) k_bs < k / = / > k, (b) one true run / two or more,
(c) the network's first candidate (the highest scored cell) inside the true cells or not.
#264 (ongoing, O_box's gain split): the boxes = the last-queue runs; per box, the true end row E (first horizon row with no true queued cell
in the box's links from there on; 7 = it does not end). end-only = rows >= E of the box's links set to 0, rows < E as the network's;
space-only = on the rows where the truth still has a queued cell in the box, the box's cells set to the truth, the other rows as the
network's; O_box as G328. Windows by (b) the 30-minute outcome of their boxes (all box links queued at T+30 / none / partly) and (c) the
largest box's age (rows visible as queued, R) and size (links): small <= 3 links, young <= 3 rows. usage: TFB_M3VER=v3 py -B -X utf8 g330_size_and_box.py"""
import sys
src = open(r"C:/Users/user/tfb/review/g328_oracle_map.py", encoding="utf-8").read()
exec(compile(src.split("# --- M3 dev (A17)")[0], "g328h", "exec"))
from cachelib import load as cload, T2_PANELS


def runs(m):
    out, j = [], 0
    while j < len(m):
        if m[j]:
            e_ = j
            while e_ + 1 < len(m) and m[e_ + 1]:
                e_ += 1
            out.append((j, e_)); j = e_ + 1
        else:
            j += 1
    return out


def onset_parts(row, y, e):
    yt = y & e; k = int(yt.sum()); b = best_set(row, np.ones(row.size, bool)); kbs = int((b & e).sum())
    s_ = np.zeros(row.size, bool)
    if k:
        s_[np.argsort(-np.where(e, row, -1.0))[:k]] = True
    top1 = int(np.argmax(np.where(e, row, -1.0))); return dict(base=iou1(b, y, e), size=iou1(s_, y, e), d=kbs - k, nruns=len(runs(yt)), top1=bool(yt[top1]))


def onset_report(rows, lab):
    D = pd.DataFrame(rows); D["gain"] = D["size"] - D.base; print(f"--- {lab} onset, {len(D)} (window, k) draws weighted: k_bs - k: <0 {np.average(D.d < 0, weights=D.w):.3f}, =0 {np.average(D.d == 0, weights=D.w):.3f}, >0 {np.average(D.d > 0, weights=D.w):.3f}; mean {np.average(D.d, weights=D.w):+.2f}", flush=True)
    tot = np.average(D.gain, weights=D.w) * D.groupby("win").w.sum().shape[0]
    for nm, m in (("k_bs < k", D.d < 0), ("k_bs = k", D.d == 0), ("k_bs > k", D.d > 0), ("one true run", D.nruns == 1), ("two+ true runs", D.nruns >= 2), ("first candidate true", D.top1), ("first candidate false", ~D.top1)):
        g = float((D.gain * D.w)[m].sum() / D.w.sum()); print(f"  {nm:22s}: weight share {D.w[m].sum() / D.w.sum():.3f}, O_size gain contribution {g:+.4f}", flush=True)


def box_parts(p, y, lq, decode_iou):
    """p (6, n) pre-temperature, y (6, n) true scored cells; returns IoU of base / O_box / end-only / space-only through decode_iou"""
    out = {"base": decode_iou(p)}; q_box, q_end, q_sp = p.copy(), p.copy(), p.copy(); outc = []
    for a, b in runs(lq):
        yb = y[:, a:b + 1]; alive = yb.any(axis=1); E = 7
        for r in range(6):
            if not alive[r:].any():
                E = r; break
        q_box[:, a:b + 1] = yb; q_end[E:, a:b + 1] = 0.0
        for r in range(6):
            if alive[r]:
                q_sp[r, a:b + 1] = yb[r]
        outc.append("all" if yb[-1].all() else ("none" if not yb[-1].any() else "part"))
    out.update(O_box=decode_iou(q_box), end=decode_iou(q_end), space=decode_iou(q_sp)); return out, outc


def box_report(rows, lab):
    D = pd.DataFrame(rows); print(f"--- {lab} ongoing, {len(D)} windows: base {D.base.mean():.4f} | O_box {D.O_box.mean() - D.base.mean():+.4f}, end-only {D.end.mean() - D.base.mean():+.4f}, space-only {D.space.mean() - D.base.mean():+.4f}", flush=True)
    for col, vals in (("outcome", ("all", "none", "part", "mixed", "nobox")), ("stratum", ("young/small", "old/large", "nobox"))):
        for v in vals:
            m = D[col] == v
            if m.any():
                print(f"  {col}={v:11s}: windows {int(m.sum())}, O_box {(D.O_box - D.base)[m].sum() / len(D):+.4f}, end-only {(D.end - D.base)[m].sum() / len(D):+.4f}, space-only {(D.space - D.base)[m].sum() / len(D):+.4f} (contributions to the mean)", flush=True)


def wstrat(outc, lq, Rw):
    rs = runs(lq)
    if not rs:
        return "nobox", "nobox"
    oc = outc[0] if len(set(outc)) == 1 else "mixed"; a, b = max(rs, key=lambda x: x[1] - x[0]); n = b - a + 1
    v, o = Rw[:, :12, 0].astype(float), Rw[:, :12, 2].astype(float) > 0; q = o & (v <= 1.0); rows_ = np.flatnonzero(o.any(axis=0)); age = 0
    for r in rows_[::-1]:
        if q[a:b + 1, r].any():
            age += 1
        else:
            break
    return oc, ("young/small" if (n <= 3 or age <= 3) else "old/large")


# M3 dev
D4 = pickle.load(open(W + "g284_og_pretemp_dev.pkl", "rb")); d_ = "C:/tfb_data/tfb_nn/nn_test_m3v3/"; RM = np.load(d_ + "R.npy", mmap_mode="r"); off = np.load(d_ + "off.npy"); pos = {str(w): i for i, w in enumerate(np.load(d_ + "window_id.npy", allow_pickle=True))}; rows = []
for w, v in D4.items():
    m = G.META[w]; i_ = G.POS[w]; lq = np.asarray(G.X[G.OFF[i_]:G.OFF[i_ + 1], G.CI], float) == 1; y = m["y"].T.astype(bool) & m["m"].T.astype(bool); r = dict.fromkeys(("base", "O_box", "end", "space"), 0.0); outc = []
    for pr, p, ex, nr, _ in v["k"].values():
        o, outc = box_parts(p, y, lq, lambda q: G.iou(g132.gate(bs(tmp(q, 0.8)), ex, nr), m))
        for k_, v_ in o.items():
            r[k_] += pr * v_
    r["outcome"], r["stratum"] = wstrat(outc, lq, np.asarray(RM[off[pos[w]]:off[pos[w] + 1]])); rows.append(r)
box_report(rows, "M3 dev (A17)")
D8 = pickle.load(open(W + "g298_on_pretemp.pkl", "rb")); rows = []
for w, v in D8["M"].items():
    for pr, row in v["A"]:
        rows.append(dict(win=w, w=pr, **onset_parts(row, v["y"], v["e"])))
onset_report(rows, "M3 dev (A17)")
PRB = {"2031-01": [pickle.load(open(W + f"probs_jfmh0__capB0_s{s}_fold2030-12.pkl", "rb")) for s in range(4)], "2030-06": [pickle.load(open(W + f"probs_junmh0__cvf_attnglob_s{s}_fold2030-06.pkl", "rb")) for s in (0, 1)]}
PRB["2031-02"] = PRB["2031-01"]; A_ = "C:/tfb_data/tfb_nn/nn_data_afdem/"; RA = np.load(A_ + "R.npy", mmap_mode="r"); oa = np.load(A_ + "off.npy")
for lab, months in (("January / February", ("2031-01", "2031-02")), ("June 2030", ("2030-06",))):
    rows = []
    for mo in months:
        KS = [k for k in np.flatnonzero(T_.META["is_sel"] & (MON == mo)) if str(T_.META["cond"][k]) == "queue_ongoing" and str(k) in PRB[mo][0]]
        for pnl in sorted({str(T_.META["panel"][k]) for k in KS}):
            P = h.load(pnl, "train"); key = (mo, pnl, "mask"); h.CTX.clear(); h.sim_blank.cache_clear(); h.CTX[key] = h.Ctx(key, P, np.where(P.blank, np.nan, P.V), P.V); c = h.CTX[key]
            for k in [k for k in KS if str(T_.META["panel"][k]) == pnl]:
                t = int(T_.META["t"][k]); dom = int(P.times[t][8:10]) - 1; lab_, el = T_.LAB[k]; y, e = lab_.astype(bool), el.astype(bool); yt = y & e
                lq = np.asarray(h.XN[h.OFFN[k]:h.OFFN[k + 1]][:, h.G.CI], float) == 1; p_ = sum(d[str(k)] for d in PRB[mo]) / len(PRB[mo]); cone = h.cone_parts(c, t, lq, P.length); r = dict.fromkeys(("base", "O_box", "end", "space"), 0.0); outc = []
                for kk, pr in h.m3v2.kdist(dom, cap=3).items():
                    pc = p_ * h.cone_f(c, cone, t, kk); s_, nd = h.seen(c, t, kk, -18, 12); pb = pc * np.where(h.G.dil(s_, 1), 1.0, 0.85)[None, :] if nd else pc.copy()
                    o, outc = box_parts(pb, yt, lq, lambda q: h.iou(h.far_gate(c, h.decode(tmp(q, 0.8)), t, kk, lq), y, e, "queue_ongoing"))
                    for k_, v_ in o.items():
                        r[k_] += pr * v_
                r["outcome"], r["stratum"] = wstrat(outc, lq, np.asarray(RA[oa[k]:oa[k + 1]])); rows.append(r)
            h.load.cache_clear()
    box_report(rows, lab)
    srcd = D8["J"] if lab.startswith("January") else pickle.load(open(W + "g329_jun_onset_rows.pkl", "rb")); rows = []
    for k, v in srcd.items():
        for pr, row in v["C"]:
            rows.append(dict(win=k, w=pr, **onset_parts(row, v["y"], v["e"])))
    onset_report(rows, lab)
