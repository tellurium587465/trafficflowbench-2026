"""G242 (11/03, ledger #205, the strategy's 22:30 request (c)1, DATA_RULE_USAGE_AUDIT 1): occupancy, released in the history rows and
read by no Task 2 feature, tried on the physics onset ranker the way G214 (#180) tried the ramps: per link the occupancy over the
critical density at T-5 and over T-15..T-5, as window percentiles (op), and the same plus row T (ot, ruling 745700). G214's source with
its extra() and arms replaced; the ranker is retrained the same way (March: nine months; January, February: month left out, 400 rounds).
usage: TFB_M3VER=v3 py -B -X utf8 g242_occupancy.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


H = '''
from scipy.stats import rankdata
from t2_model import PanelInputs
BC = {}; RC = {}; RNG = np.random.default_rng(180)


def pct(v):
    P_ = np.full(len(v), 0.5); ok = np.isfinite(v)
    if ok.sum() >= 2 and np.nanstd(v) > 0:
        P_[ok] = (rankdata(v[ok]) - 0.5) / ok.sum()
    return P_


def births(p, sp):
    if (p, sp) not in BC:
        P = load(p, sp); PI = PanelInputs(P); V, Q = PI.mask_v, PI.mask_q; q = (V <= P.vcut[None, :]) & np.isfinite(V); b = q[3:] & ~q[2:-1] & ~q[1:-2] & ~q[:-3]; r, j = np.nonzero(b); r = r + 3; qb = np.array([np.nanmean(Q[x - 3:x, y]) if np.isfinite(Q[x - 3:x, y]).any() else np.nan for x, y in zip(r, j)]); ok = np.isfinite(qb); BC[(p, sp)] = (r[ok], j[ok], qb[ok], Q); on = np.zeros((P.T, P.n)); off = on.copy(); li = {l: i for i, l in enumerate(P.links)}
        for i_, (rt, rl) in enumerate(zip(P.ramp_type, P.ramp_link)):
            c_ = li.get(str(rl))
            if c_ is None:
                continue
            f = np.where(P.RE[:, i_], np.nan_to_num(P.R[:, i_]), 0.0)
            if str(rt).upper() in ("OR", "ON"):
                on[:, c_] += f
            else:
                off[:, c_] += f
        RC[(p, sp)] = (on, off, P.cap.copy(), P.n)
    return BC[(p, sp)]


def extra(p, sp, t, kk, t0):                                   # G242: occupancy over the critical density, window percentiles (rows <= T-5; and row T)
    P = load(p, sp); O = P.O; kc = np.where(P.kc > 0, P.kc, np.nan)
    with np.errstate(all="ignore"):
        o1 = O[t - 1] / kc; o3 = np.nanmean(O[t - 3:t], axis=0) / kc; oT = O[t] / kc
    return np.column_stack([pct(o1), pct(o3)]), np.column_stack([pct(oT)])


COLS_ = {"base": (), "op": (0,), "ot": (0, 1)}


def join(F0, ex, arm):
    return np.column_stack([F0] + [ex[i] for i in COLS_[arm]]).astype(np.float32)


def train_arm(FX, F0, mo, arm):
    tr = [k for k in F0 if MON[k] != mo]; X = np.vstack([join(F0[k], FX[k], arm)[T.LAB[k][1][5].astype(bool)] for k in tr]); y = np.concatenate([T.LAB[k][0][5][T.LAB[k][1][5].astype(bool)] for k in tr]).astype(int); return lgb.train(PO.PRM, lgb.Dataset(X, y), 400)


def sets(c, p_, ph, a, t, k, a2=None, PH=None):
'''
sub('def sets(c, p_, ph, a, t, k, a2=None):\n', H)
sub('"amt0.7", "hid", "hidf", "ac"); ND = 8', '"amt0.7", "hid", "hidf", "ac", "b0", "op", "ot"); ND = 8')
sub('"ac": rule(mix(a if a2 is None else a2), 0.25)}', '"ac": rule(mix(a if a2 is None else a2), 0.25), **{v_: rule((lambda q: (q.__setitem__(5, (1 - a) * p_[5] + a * PH[v_]), q)[1])(p_.copy()), 0.25) for v_ in ("b0", "op", "ot")}}')
sub('rows = []; print("births", len(ev)', 'rows = []; FF = phys_feats(); FX = fx(FF); MM = {v_: train_arm(FX, FF, "none", {"b0": "base"}.get(v_, v_)) for v_ in ("b0", "op", "ot")}; print("births", len(ev)')
sub('ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); y, e = m["y"]', 'F0_ = PO.feats(Xw); ph = phys.predict(F0_); kd_ = m3v2.kdist(m["day"], cap=3); ex_ = extra(m["panel"], "validation", m["t"], max(kd_, key=kd_.get), 0); PH = {v_: MM[v_].predict(join(F0_, ex_, {"b0": "base"}.get(v_, v_))) for v_ in MM}; a = PO.adaptive_a(ev, m["day"]); y, e = m["y"]')
sub('S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))', 'S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]), PH)')
sub('def march(limit=None):', '''def fx(F):
    out = {}
    for k in F:
        p = str(T.META["panel"][k]); P = load(p, "train"); t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; t0 = t // 288 * 288 - dom * 288; kd = m3v2.kdist(dom, cap=3); kk = int(RNG.choice(list(kd), p=np.array(list(kd.values())) / sum(kd.values()))); out[k] = extra(p, "train", t, kk, t0)
    print("extra features", len(out), flush=True); return out


def march(limit=None):''')
sub('mdl = phys_lomo(FF, mo) if FF is not None else lgb.Booster(model_file=W + "physon_base.txt"); F_m = rarity_lomo(mo);', 'mdl = phys_lomo(FF, mo) if FF is not None else lgb.Booster(model_file=W + "physon_base.txt"); FX_ = fx(FF) if "FX_" not in globals() else globals()["FX_"]; globals()["FX_"] = FX_; MM = {v_: train_arm(FX_, FF, mo, {"b0": "base"}.get(v_, v_)) for v_ in ("b0", "op", "ot")}; F_m = rarity_lomo(mo);')
sub('ph = mdl.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, dom)\n', 'F0_ = PO.feats(Xw); ph = mdl.predict(F0_); kd_ = m3v2.kdist(dom, cap=3); ex_ = extra(p, "train", t, max(kd_, key=kd_.get), t // 288 * 288 - dom * 288); PH = {v_: MM[v_].predict(join(F0_, ex_, {"b0": "base"}.get(v_, v_))) for v_ in MM}; a = PO.adaptive_a(ev, dom)\n')
sub('S = sets(c, p_, ph, a, t, kk)', 'S = sets(c, p_, ph, a, t, kk, None, PH)')
sub('"g170_m3.csv"', '"g242_m3.csv"'); sub('"g170_train.csv"', '"g242_train.csv"')
exec(compile(s, "g170r", "exec"))
