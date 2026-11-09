"""The Task 2 composition shared by t2_build_variant.py (all 160 windows) and t2_window_entry.py (one window), 10/23:
the same env options give the same probabilities in both. Options (all off = the s76 composition):
  TFB_T2_ONSET_SEEDS=8 : the onset blend over q24f seeds 0-7 (E5)
  TFB_T2_APRIL=e4      : the April (private) windows use the E4 families e4f_* seeds 0-3 for both conditions; the March
                         windows keep the train-months models (PLAN §226)
  TFB_T2_E4_SEEDS      : the E4 seeds on the April windows (default 4; 8 once seeds 4-7 exist, work/e4s47_chain.sh)
  TFB_T2_OG_SEEDS_PRIV : the ongoing family's seeds on the April windows when E4 is off (default 4; the final B: 8)
  TFB_T2_TTA_SPLITS    : the splits whose windows get TTA where tta is on (default validation,private)
  TFB_T2_MIX_SPLITS    : the splits that get the LightGBM / E3' / physics mixes (default validation,private)
  TFB_T2_LGB_A         : no-memory LightGBM (t2_lgb_nomem_final.py) mixed into ongoing at a, before the cone
  TFB_T2_E3=1          : E3' / onset shift (t2_e3_official.py), on the conditions TFB_T2_E3_CONDS (default both; the final A:
                         queue_onset = the onset shift c = 0.5 only, METAGAME_REPLY §23)
  TFB_T2_OG_FAMILY     : the ongoing families (default q43f,q46f = the R world; the finals: c24f = the C world, §22 / §24)
  TFB_T2_CONE_SPLITS   : the splits t2_cone.py touches (default validation,private; read by t2_cone.py itself)
  TFB_T2_PHYS_A        : physics onset (t2_phys_onset.py) mixed at T+30: a number or "adaptive" (per window, physon_a.pkl)
  TFB_T2_TFM           : 10/29 lane 9 (ledger #55 / #56): the attention-backbone networks per split, "validation:t1+t2;private:t3+t4",
                         their mean mixed at TFB_T2_TFM_W (default 0.2) right after the network blend (before the LightGBM, E3'
                         and physics mixes), on the conditions TFB_T2_TFM_CONDS (default both); unset = off
  TFB_T2_OGRAW_SPLITS  : 10/30 ledger #72 (J8): the ongoing windows of these splits get the raw expected-IoU set of their
                         mixed probabilities (no cone, seen x0.85 or far gate), ograw(); unset = off
  TFB_T2_OGBASE_SPLITS : 10/31 ledger #97: the ongoing windows of these splits take the saved networks in place of the
                         choice's ongoing source (no X1); unset = off
  TFB_T2_X1BETA         : 11/01 ledger #136: on the splits TFB_T2_X1BETA_SPLITS the ongoing blend = base + beta (X1 - base) in logit, in place of the choice's source; unset = off
  TFB_T2_X1ARM          : 11/01 ledger #125: the X1 files probs_x1off<arm>__<tag>.pkl; sd = same-day X1 (t2_x1_sameday.py)
  TFB_T2_OGX3           : 11/01 ledger #125: raw-history networks mixed into ongoing at TFB_T2_OGX3_W (default 0.2), right after the tfm mix, "t1+t2"; unset = off
  TFB_T2_OGU2D          : 11/03 ledger #201: the 2D U-Net (#133, t2_u2d_apply.py) mixed into ongoing at TFB_T2_OGU2D_W (default 0.2), right after the raw-history mix, "u2d_s0+u2d_s1"; unset = off
  TFB_T2_ONPC           : 11/03 ledger #201 (#17 / #172b): the rank-input network (within-window percentile inputs, t2_pct.py) mixed into onset at TFB_T2_ONPC_W (default 0.2), just before the physics onset, "pcL_s0+pcL_s1"; unset = off
  TFB_T2_LQT            : 11/03 ledger #197: the far gate and the cone take the row T as the last queue where it is observed; unset = off
  TFB_T2_GATE2          : 11/01 ledger #146: t2_far_gate.py decodes again after the gate, the dropped cells at p = 0
  TFB_T2_OGTE           : 11/01 ledger #125: temperature on the ongoing probabilities just before best_set (t2_onset_evidence.py, ograw); default 1
  TFB_T2_SELC=1         : 11/05 ledger #283: after the temperature, the ongoing probabilities conditioned on the official window selection
                         (t2_selcond.apply: joint samples kept when the window is the first qualifying origin, V11); unset = off
window_probs(...) -> the window's (6, n) probabilities before the cone, from get(prefix, tag) = one model's (6, n)
probabilities of the window: prefix "" saved model, "afdemtta<k>__" its TTA draw k, "x1off__" X1 (fine-tuned on the month's
earlier days), "x1tta<k>__" X1's TTA draw k; get("lgbnm", "") / get(<physics name>, "") the mixes; get("physon_a", "") a;
get("e3stats", "") the window's earlier-day statistics (kq, ko, sp) or None."""
import os, json, numpy as np

SRC = os.path.dirname(os.path.abspath(__file__))
CF5 = (("attnglob", ":2"), ("attnwp05", ":2"), ("attn1", ":2"), ("drop1", ""), ("r13", ""))
lg = lambda x: np.log(np.clip(x, 1e-6, 1 - 1e-6) / (1 - np.clip(x, 1e-6, 1 - 1e-6)))


def config():
    e = os.environ
    spec = {"queue_onset": json.load(open(f"{SRC}/t2_specs.json"))["s37"]["spec"],
            "queue_ongoing": "+".join(f"{f}_{c}_s{s}{w}" for s in range(4) for f in e.get("TFB_T2_OG_FAMILY", "q43f,q46f").split(",") for c, w in CF5)}
    if e.get("TFB_T2_ONSET_SEEDS", "4") == "8":
        spec["queue_onset"] = "+".join(f"q24f_{c}_s{s}{w}" for s in range(8) for c, w in CF5)
    ne4 = int(e.get("TFB_T2_E4_SEEDS", "4"))                   # 10/23: 8 = E4 seeds 0-7 (A symmetric with the 8-seed onset)
    priv = {c: "+".join(f"e4f_{cc}_s{s}{w}" for s in range(ne4) for cc, w in CF5) for c in spec} if e.get("TFB_T2_APRIL", "") == "e4" else dict(spec)
    nogp = int(e.get("TFB_T2_OG_SEEDS_PRIV", "4"))             # 10/23: the April ongoing family over seeds 0..n-1 (B: 8; March stays 4)
    if nogp != 4 and e.get("TFB_T2_APRIL", "") != "e4":
        priv["queue_ongoing"] = "+".join(f"{f}_{c}_s{s}{w}" for s in range(nogp) for f in e.get("TFB_T2_OG_FAMILY", "q43f,q46f").split(",") for c, w in CF5)
    pf = e.get("TFB_T2_PRIV_FAMILY", "")                        # 11/06 ledger #252: the April windows (both conditions) from a family trained with March
    if pf:                                                      # (q24m), seeds 0..TFB_T2_PRIV_SEEDS-1 x CF5; March keeps its families
        ns_ = int(e.get("TFB_T2_PRIV_SEEDS", "8")); priv = {c: "+".join(f"{pf}_{cc}_s{s}{w}" for s in range(ns_) for cc, w in CF5) for c in spec}
    return dict(spec={"validation": spec, "private": priv},
                tta=set(e.get("TFB_T2_TTA_SPLITS", "validation,private").split(",")),
                mix=set(e.get("TFB_T2_MIX_SPLITS", "validation,private").split(",")),
                la=float(e.get("TFB_T2_LGB_A", "0")), e3=e.get("TFB_T2_E3", "0") == "1",
                e3c=set(e.get("TFB_T2_E3_CONDS", "queue_onset,queue_ongoing").split(",")),
                pa=e.get("TFB_T2_PHYS_A", "0"), phys=e.get("TFB_T2_PHYS", "physon"),
                tfm={k: v.split("+") for k, v in (x.split(":") for x in e.get("TFB_T2_TFM", "").split(";") if x)},
                tw=float(e.get("TFB_T2_TFM_W", "0.2")), tfmc=set(e.get("TFB_T2_TFM_CONDS", "queue_onset,queue_ongoing").split(",")),
                ograw=set(filter(None, e.get("TFB_T2_OGRAW_SPLITS", "").split(","))),
                ogbase=set(filter(None, e.get("TFB_T2_OGBASE_SPLITS", "").split(","))),
                x1b=float(e.get("TFB_T2_X1BETA", "0")), x1bs=set(filter(None, e.get("TFB_T2_X1BETA_SPLITS", "").split(","))), x1arm=e.get("TFB_T2_X1ARM", ""), x3=list(filter(None, e.get("TFB_T2_OGX3", "").split("+"))), xw=float(e.get("TFB_T2_OGX3_W", "0.2")), u2d=list(filter(None, e.get("TFB_T2_OGU2D", "").split("+"))), uw=float(e.get("TFB_T2_OGU2D_W", "0.2")), pc=list(filter(None, e.get("TFB_T2_ONPC", "").split("+"))), pw=float(e.get("TFB_T2_ONPC_W", "0.2")), te=float(e.get("TFB_T2_OGTE", "1")))


def parts(cfg, split, c):
    return [(x.split(":")[0], float(x.split(":")[1]) if ":" in x else 1.0) for x in cfg["spec"][split][c].split("+")]


def window_probs(wid, c, split, choice, cfg, get):
    """choice = (onset source, onset tta, ongoing source, ongoing tta); source base | x1 | half"""
    src, tta = (choice[0], choice[1]) if c == "queue_onset" else (choice[2], choice[3])
    if c == "queue_ongoing" and split in cfg["ogbase"]:           # 10/31 ledger #97
        src = "base"
    pt = parts(cfg, split, c); tw = sum(w for _, w in pt)
    blend = lambda pre: sum(w * get(pre, m) for m, w in pt) / tw
    pre = {"base": ("", [f"afdemtta{k}__" for k in range(1, 9)]), "x1": (f"x1off{cfg.get('x1arm', '')}__", [f"x1tta{k}__" for k in range(1, 9)])}

    def one(s):
        if not tta or split not in cfg["tta"]:
            return blend(pre[s][0])
        st = np.stack([blend(p_) for p_ in pre[s][1]])
        return np.where((st == 0).all(axis=0), 0.0, 1 / (1 + np.exp(-lg(st).mean(axis=0))))
    if c == "queue_ongoing" and split in cfg.get("x1bs", ()):    # 11/01 ledger #136: the middle of X1, on the blends
        a_ = one("base"); p = 1 / (1 + np.exp(-(lg(a_) + cfg["x1b"] * (lg(one("x1")) - lg(a_)))))
    else:
        p = (one("base") + one("x1")) / 2 if src == "half" else one(src)
    tt = cfg.get("tfm", {}).get(split, [])
    if tt and c in cfg["tfmc"]:                                   # 10/29 lane 9: the attention backbone's seed mean
        p = (1 - cfg["tw"]) * p + cfg["tw"] * sum(get("", t) for t in tt) / len(tt)
    if c == "queue_ongoing" and cfg.get("x3"):                    # 11/01 ledger #125: the raw-history networks' seed mean
        p = (1 - cfg["xw"]) * p + cfg["xw"] * sum(get("", t) for t in cfg["x3"]) / len(cfg["x3"])
    if c == "queue_ongoing" and cfg.get("u2d"):                   # 11/03 ledger #201: the 2D U-Net's seed mean
        p = (1 - cfg["uw"]) * p + cfg["uw"] * sum(get("", t) for t in cfg["u2d"]) / len(cfg["u2d"])
    if c == "queue_ongoing" and cfg["la"] > 0 and split in cfg["mix"]:
        p = (1 - cfg["la"]) * p + cfg["la"] * get("lgbnm", "")
    if cfg["e3"] and split in cfg["mix"] and c in cfg["e3c"]:
        import t2_e3_official as E3
        st = get("e3stats", "")
        if st is not None:
            p = E3.correct(p, *st, *E3.CONST[c])
    if c == "queue_onset" and cfg.get("pc"):                      # 11/03 ledger #201 (#17): the rank-input network's seed mean
        p = (1 - cfg["pw"]) * p + cfg["pw"] * sum(get("", t) for t in cfg["pc"]) / len(cfg["pc"])
    if c == "queue_onset" and cfg["pa"] != "0" and split in cfg["mix"]:
        a = get("physon_a", "") if cfg["pa"] == "adaptive" else float(cfg["pa"])
        p = p.copy(); p[5] = (1 - a) * p[5] + a * get(cfg["phys"], "")[5]
    return p


def temper(p, te):
    """11/01 ledger #125: the temperature on probabilities before best_set; te = 1 returns p unchanged"""
    return p if te == 1 else 1 / (1 + np.exp(-lg(p) / te))


def ograw(qpath, probs, cond, split, cfg):
    """10/30 (ledger #72, J8): the ongoing windows of the splits in cfg['ograw'] get the raw expected-IoU set of their mixed
    probabilities (window_probs: before the cone, the seen x0.85 and the far gate) in place of the post-processed one.
    Rewrites the queue parquet qpath in place; returns the number of windows whose set changed."""
    if not cfg["ograw"]:
        return 0
    import pandas as pd
    from t2_model import best_set
    q = pd.read_parquet(qpath); qp = q.queue_pred.to_numpy().copy(); n = 0
    for wid, ii in q.groupby("window_id", sort=False).indices.items():
        if cond[wid] == "queue_ongoing" and split[wid] in cfg["ograw"]:
            pb = temper(probs[wid], cfg.get("te", 1.0))
            if os.environ.get("TFB_T2_SELC", "0") == "1":              # 11/05 ledger #283: conditioned on the official selection (B7c's April rows)
                import t2_selcond; pb = t2_selcond.apply(wid, pb)[0]
            if os.environ.get("TFB_T2_OGDEX", "0") == "1":             # 11/04 ledger #216: B12, the exact expected IoU (t2_dex.DEX)
                from t2_dex import DEX; sel = DEX(pb).ravel().astype(qp.dtype)
            else:
                sel = best_set(pb.ravel(), np.ones(pb.size, bool)).astype(qp.dtype)
            assert len(ii) == sel.size and (np.diff(ii) == 1).all(), wid      # one contiguous block, rows (horizon, link)
            n += int((qp[ii] != sel).any()); qp[ii] = sel
    q["queue_pred"] = qp; q.to_parquet(qpath, index=False); return n


def _nomem_X(Xt):
    """the no-memory LightGBM's input for one window (t2_fast's priors zeroed + apply_drop on its feature names; the names
    from repro/featcache_names.json, so the 3 GB training cache is not needed)"""
    from common import WORK, ROOT
    fn = ROOT / "repro" / "featcache_names.json"                 # 10/23: the 3 GB cache's column names, shipped as a small file
    nm = json.load(open(fn)) if fn.exists() else [str(x) for x in np.load(WORK + "/t2/featcache2_noT_af.npz", allow_pickle=True)["names"]]
    ix = nm.index; drop = [ix(c) for c in "prior_on,prior_og,pos,vf,capl,lanes,length,w,tod,dow".split(",")]
    Xw = Xt.copy(); Xw[:, ix("prior_on")] = 0.0; Xw[:, ix("prior_og")] = 0.0; Xw[:, drop] = np.nan
    return Xw, ix


def lgb_ongoing(Xt, model=None):
    """t2_lgb_nomem_final.apply for one ongoing window (step blocks as t2_fast.step_block)"""
    import lightgbm as lgb
    from common import WORK
    Xw, ix = _nomem_X(Xt)

    def step_block(Xs, s):
        h = s * 5 / 60.0
        return np.column_stack([Xs, np.full(len(Xs), s, np.float32), Xs[:, ix("seg_tail")] + np.nan_to_num(Xs[:, ix("s_flow")]) * h,
                                Xs[:, ix("seg_tail")] + np.nan_to_num(Xs[:, ix("s_emp_tail")]) * h]).astype(np.float32)
    m = model or lgb.Booster(model_file=WORK + "/t2/lgbnm_queue_ongoing.txt")
    return m.predict(np.vstack([step_block(Xw, s + 1) for s in range(6)]), num_threads=14).reshape(6, len(Xw)).astype(np.float32)


def lgb_onset(Xt, model=None):
    """t2_lgb_nomem_final.apply for one onset window (the last step only)"""
    import lightgbm as lgb
    from common import WORK
    Xw, _ = _nomem_X(Xt); p = np.zeros((6, len(Xw)))
    p[-1] = (model or lgb.Booster(model_file=WORK + "/t2/lgbnm_queue_onset.txt")).predict(Xw, num_threads=14)
    return p.astype(np.float32)


# 10/23 the candidates by name (env, choice "onset source,tta,ongoing source,tta,cone"); A / B fixed in DECISION_LEDGER.md
# 10/23 06:44 (before the hold-out is read). t2_trunc_physical.py, repro.
# A1 = the 10/23 05:50 draft (finA1). A / B = METAGAME_REPLY §22-25 and the ledger (13:45): the March windows of both =
# the s80q flow (q24f 8-seed onset, q24f + X1 + cone); B's April = q24f + X1 (no cone, no synthetic); A's April = E4 + X1 +
# physics onset (adaptive a) + LightGBM 0.2 + cone; no TTA.
FINAL = {"s70": ({}, "base,0,base,0,0"),
         "A1": (dict(TFB_T2_APRIL="e4", TFB_T2_ONSET_SEEDS="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2"), "base,0,x1,1,1"),
         "A": (dict(TFB_T2_APRIL="e4", TFB_T2_E4_SEEDS="12", TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_MIX_SPLITS="private",
                    TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2"), "base,0,x1,0,1"),   # 07:00 no onset shift (#28); 13:45 March = s80q; 10/24 E4 12 seeds = finA5 (#38)
         "A4": (dict(TFB_T2_APRIL="e4", TFB_T2_E4_SEEDS="8", TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_MIX_SPLITS="private",
                     TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2"), "base,0,x1,0,1"),   # finA4 (E4 8 seeds), the final A until 10/24
         "A5": (dict(TFB_T2_APRIL="e4", TFB_T2_E4_SEEDS="12", TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_MIX_SPLITS="private",
                     TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2"), "base,0,x1,0,1"),   # = A (the name finA5 was built and checked under)
         "A3": (dict(TFB_T2_APRIL="e4", TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive",
                     TFB_T2_LGB_A="0.2"), "base,0,x1,0,1"),   # finA3 (E4 4 seeds), spare
         "B": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation"), "base,0,x1,0,1"),
         "B3": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation"), "base,0,x1,0,1"),   # finB3 (April ongoing 4 seeds), spare
         # 10/23 13:45 (ledger): B's ongoing = q24f + X1, no synthetic (E[max] of the two finals); the c24f versions kept by name
         "Ac24": (dict(TFB_T2_APRIL="e4", TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="c24f", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive",
                       TFB_T2_LGB_A="0.2"), "base,0,x1,0,1"),
         "Bc24": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="c24f", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation"), "base,0,x1,0,1"),
         # 10/29 lane 9 (ledger #55 / #56): the attention backbone (tfm4, Kaggle T4) mixed into ongoing only (onset -0.0027 on M3);
         # March (both) = tfm4 seeds 0-4 (training months), April A = the E4-type e4tfm4 seeds 0-3, April B unchanged
         "B5": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                     TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing"), "base,0,x1,0,1"),
         "A6": (dict(TFB_T2_APRIL="e4", TFB_T2_E4_SEEDS="12", TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_MIX_SPLITS="private",
                     TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2", TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5))
                     + ";private:" + "+".join(f"e4tfm4_s{s}" for s in range(4)), TFB_T2_TFM_CONDS="queue_ongoing"), "base,0,x1,0,1"),
         # 10/30 ledger #72 (J8): B6 = B5 with the April ongoing windows left raw (the far gate and seen x0.85 lose -0.013 / -0.015
         # in the train months, z <= -2 in both); the March rows = B5's
         "B6": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                     TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                     TFB_T2_OGRAW_SPLITS="private"), "base,0,x1,0,1"),
         # 10/31 ledger #95-#97: B7 = B6 without X1 on the April ongoing windows (X1 lost -0.0113 / -0.0177 on January / February, z <= -2
         # in both); the March rows = B6's
         "B7": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                     TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                     TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private"), "base,0,x1,0,1"),
         # B if the 11/4 read (#63) keeps finA5: finB4's March rows + B7's April rows; built and submitted only in that case
         "B8": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                     TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private"), "base,0,x1,0,1"),
         # 10/31 ledger #110: A7 = A6 without E4 and without synthetic data: the April windows take q24f x CF5 x 8 seeds (onset and ongoing) and the
         # training-month tfm4; X1, the LightGBM, the physics onset, the cone and the far gate as in A6; the March rows = A6's
         "A7": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                     TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing"), "base,0,x1,0,1"),
         # 11/01 ledger #125: the bundle of #121 on both months. A8 = A7 + same-day X1 + X3a 0.2 + temperature 1.2; B9 = B7 + X1 at logit beta 0.25 on April (#136) + X3a 0.2 + temperature 1.2
         "A8": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                     TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                     TFB_T2_X1ARM="sd", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1"), "base,0,x1,0,1"),
         "B9": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                     TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                     TFB_T2_OGRAW_SPLITS="private", TFB_T2_X1BETA="0.25", TFB_T2_X1BETA_SPLITS="private", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1"), "base,0,x1,0,1"),
         "A10": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24t", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",   # 11/02 ledger #164: A8 with the ongoing networks that read the row at T
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sd", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1"), "base,0,x1,0,1"),
         "B9s": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1"), "base,0,x1,0,1"),
         # 11/03 ledger #197: A8 / B9s with the row at T in the post-processing (far gate, cone); B9s's April ongoing is raw, so only its March rows move
         "A8r": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sd", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1"), "base,0,x1,0,1"),
         "B9sr": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                       TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                       TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1"), "base,0,x1,0,1"),
         # 11/03 ledgers #197, #201: A11 = A8r + the 2D U-Net at 0.2 on the ongoing windows of both months
         "A11": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sd", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1"), "base,0,x1,0,1"),
         # 11/03 ledger #201: B10 = B9s + the 2D U-Net at 0.2 on the ongoing windows of both months (B's line read on the top coverage third, the strategy's ruling)
         "B10": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1"), "base,0,x1,0,1"),
         # 11/03 ledger #201: A12 = A11 + the rank-input network at 0.2 on the onset windows of both months (#17 / #172b)
         "A12": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sd", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1", TFB_T2_ONPC="pcL_s0+pcL_s1"), "base,0,x1,0,1"),
         # 11/03 ledger #189: A13 = A12 with April's same-day X1 on origins every 3 rows (t2_x1_sameday.py X1_ARM=sdd3, X1_POOL_PRIV=nn_data_april_sp3); March stays sd
         "A13": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1", TFB_T2_ONPC="pcL_s0+pcL_s1"), "base,0,x1,0,1"),
         # 11/05 ledgers #230, #231: A15 = A13 with April's ongoing family over seeds 0-15 (dense same-day X1 sdd3 for seeds 8-15 by #230); March and onset unchanged
         "A15": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1", TFB_T2_ONPC="pcL_s0+pcL_s1"), "base,0,x1,0,1"),
         # 11/05 ledger #232: A16 = A15 with the rank-input onset network over seeds 0-3 (weight 0.2 unchanged); March and April onset change
         "A16": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1", TFB_T2_ONPC="pcL_s0+pcL_s1+pcL_s2+pcL_s3"), "base,0,x1,0,1"),
         # 11/05 ledger #234: A17 = A16 with the ongoing temperature 0.8 (chosen by the cell log loss) in place of 1.2
         "A17": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="0.8", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1", TFB_T2_ONPC="pcL_s0+pcL_s1+pcL_s2+pcL_s3"), "base,0,x1,0,1"),
         # 11/05 ledger #245: A21 = A17 without the 2D U-Net in the ongoing mix (March and April)
         "A21": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="0.8", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_ONPC="pcL_s0+pcL_s1+pcL_s2+pcL_s3"), "base,0,x1,0,1"),
         # 11/05 ledger #283: A25 = A21 with the ongoing probabilities conditioned on the official selection (t2_selcond, V11) before best_set (March and April)
         "A25": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="0.8", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_ONPC="pcL_s0+pcL_s1+pcL_s2+pcL_s3",
                      TFB_T2_SELC="1"), "base,0,x1,0,1"),
         # 11/06 ledger #252: A22 = A21 with the April windows (onset and ongoing networks, X1 sdd3 on them) from q24m (the training months + all of March) x 8 seeds;
         # March rows = A21's. A22c = A22 + the selection conditioning (#283)
         "A22": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="0.8", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_ONPC="pcL_s0+pcL_s1+pcL_s2+pcL_s3",
                      TFB_T2_PRIV_FAMILY="q24m", TFB_T2_PRIV_SEEDS="8"), "base,0,x1,0,1"),
         "A22c": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                       TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)) + ";private:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                       TFB_T2_X1ARM="sdd3", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="0.8", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_ONPC="pcL_s0+pcL_s1+pcL_s2+pcL_s3",
                       TFB_T2_PRIV_FAMILY="q24m", TFB_T2_PRIV_SEEDS="8", TFB_T2_SELC="1"), "base,0,x1,0,1"),
         # 11/05 ledger #283: B7c = B7 with the same conditioning (the evidence step on March, ograw on April)
         "B7c": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="8", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private", TFB_T2_SELC="1"), "base,0,x1,0,1"),
         # 11/04 ledgers #107, #179 6: B11 = B10 with April's ongoing family over seeds 0-15 (q24f seeds 8-15 trained by #107); March unchanged
         "B11": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1"), "base,0,x1,0,1"),
         # 11/04 ledgers #216, #179 6: B12 = B11 with the ongoing decode by the exact expected IoU (TFB_T2_OGDEX, t2_dex.DEX) in the evidence step,
         # the far gate's re-decode and ograw; March changes too
         "B12": (dict(TFB_T2_ONSET_SEEDS="8", TFB_T2_OG_FAMILY="q24f", TFB_T2_OG_SEEDS_PRIV="16", TFB_T2_MIX_SPLITS="none", TFB_T2_CONE_SPLITS="validation",
                      TFB_T2_TFM="validation:" + "+".join(f"tfm4_s{s}" for s in range(5)), TFB_T2_TFM_CONDS="queue_ongoing",
                      TFB_T2_OGRAW_SPLITS="private", TFB_T2_OGBASE_SPLITS="private", TFB_T2_OGX3="x3af_s0+x3af_s1", TFB_T2_OGTE="1.2", TFB_T2_GATE2="1", TFB_T2_LQT="1", TFB_T2_OGU2D="u2d_s0+u2d_s1",
                      TFB_T2_OGDEX="1"), "base,0,x1,0,1"),
         "A2pre": (dict(TFB_T2_APRIL="e4", TFB_T2_ONSET_SEEDS="8", TFB_T2_MIX_SPLITS="private", TFB_T2_PHYS_A="adaptive", TFB_T2_LGB_A="0.2",
                        TFB_T2_E3="1", TFB_T2_E3_CONDS="queue_onset"), "base,0,x1,0,1")}   # A on the R-world March ongoing until c24f exists (E6 code-path check only)
