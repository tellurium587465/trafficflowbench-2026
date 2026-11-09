"""G214 (11/02, ledgers #180, #181): two sources the Task 2 features do not read, tried on the cheap model first: the physics onset ranker. #181 ramp flows at or before T: per link the on-ramp and off-ramp flows at T and the mean of the last three rows, over cap. #180 the month's own breakdown flow: per link, the median over the births on the earlier days (k days, +-1 link) of the flow in the 15 minutes before the birth, q_bd; features q(T - 5) / q_bd, mean of the last three rows / q_bd, (q_bd - q) / cap, a flag. All as within-window percentiles, as PO.feats, NaN at 0.5. Training windows draw k from m3v2.kdist(day, cap 3), seed 180; March reads the most likely k.
Arms on G170's flow: rp = ramp, cp = capacity, rc = both, against the current ranker, retrained the same way (March: on the nine months; January, February: month left out, 400 rounds, PO.PRM). Only row T and earlier are read.
usage: TFB_M3VER=v3 py -B -X utf8 g214_ramp_cap.py"""
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


def extra(p, sp, t, kk, t0):
    r, j, qb, Q = births(p, sp); on, off, cap, n = RC[(p, sp)]; d0 = max(t0, (t // 288 - kk) * 288); d1 = t // 288 * 288; m_ = (r >= d0) & (r < d1); bd = np.full(n, np.nan)
    if m_.any():
        jj, vv = j[m_], qb[m_]
        for l_ in np.unique(jj):
            for c_ in (l_ - 1, l_, l_ + 1):
                if 0 <= c_ < n and np.isnan(bd[c_]):
                    bd[c_] = np.median(vv[np.abs(jj - c_) <= 1])
    with np.errstate(all="ignore"):
        q1 = Q[t - 1]; q3 = np.nanmean(Q[t - 3:t], axis=0); R_ = np.column_stack([pct(on[t] / cap), pct(on[t - 3:t + 1].mean(axis=0) / cap), pct(off[t] / cap), pct(off[t - 3:t + 1].mean(axis=0) / cap)]); C_ = np.column_stack([pct(q1 / bd), pct(q3 / bd), pct((bd - q3) / cap), np.isfinite(bd).astype(float)])
    return R_, C_


COLS_ = {"base": (), "rp": (0,), "cp": (1,), "rc": (0, 1)}


def join(F0, ex, arm):
    return np.column_stack([F0] + [ex[i] for i in COLS_[arm]]).astype(np.float32)


def train_arm(FX, F0, mo, arm):
    tr = [k for k in F0 if MON[k] != mo]; X = np.vstack([join(F0[k], FX[k], arm)[T.LAB[k][1][5].astype(bool)] for k in tr]); y = np.concatenate([T.LAB[k][0][5][T.LAB[k][1][5].astype(bool)] for k in tr]).astype(int); return lgb.train(PO.PRM, lgb.Dataset(X, y), 400)


def sets(c, p_, ph, a, t, k, a2=None, PH=None):
'''
sub('def sets(c, p_, ph, a, t, k, a2=None):\n', H)
sub('"amt0.7", "hid", "hidf", "ac"); ND = 8', '"amt0.7", "hid", "hidf", "ac", "b0", "rp", "cp", "rc"); ND = 8')
sub('"ac": rule(mix(a if a2 is None else a2), 0.25)}', '"ac": rule(mix(a if a2 is None else a2), 0.25), **{v_: rule((lambda q: (q.__setitem__(5, (1 - a) * p_[5] + a * PH[v_]), q)[1])(p_.copy()), 0.25) for v_ in ("b0", "rp", "cp", "rc")}}')
sub('rows = []; print("births", len(ev)', 'rows = []; FF = phys_feats(); FX = fx(FF); MM = {v_: train_arm(FX, FF, "none", {"b0": "base"}.get(v_, v_)) for v_ in ("b0", "rp", "cp", "rc")}; print("births", len(ev)')
sub('ph = phys.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, m["day"]); y, e = m["y"]', 'F0_ = PO.feats(Xw); ph = phys.predict(F0_); kd_ = m3v2.kdist(m["day"], cap=3); ex_ = extra(m["panel"], "validation", m["t"], max(kd_, key=kd_.get), 0); PH = {v_: MM[v_].predict(join(F0_, ex_, {"b0": "base"}.get(v_, v_))) for v_ in MM}; a = PO.adaptive_a(ev, m["day"]); y, e = m["y"]')
sub('S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))', 'S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]), PH)')
sub('def march(limit=None):', '''def fx(F):
    out = {}
    for k in F:
        p = str(T.META["panel"][k]); P = load(p, "train"); t = int(T.META["t"][k]); dom = int(P.times[t][8:10]) - 1; t0 = t // 288 * 288 - dom * 288; kd = m3v2.kdist(dom, cap=3); kk = int(RNG.choice(list(kd), p=np.array(list(kd.values())) / sum(kd.values()))); out[k] = extra(p, "train", t, kk, t0)
    print("extra features", len(out), flush=True); return out


def march(limit=None):''')
sub('mdl = phys_lomo(FF, mo) if FF is not None else lgb.Booster(model_file=W + "physon_base.txt"); F_m = rarity_lomo(mo);', 'mdl = phys_lomo(FF, mo) if FF is not None else lgb.Booster(model_file=W + "physon_base.txt"); FX_ = fx(FF) if "FX_" not in globals() else globals()["FX_"]; globals()["FX_"] = FX_; MM = {v_: train_arm(FX_, FF, mo, {"b0": "base"}.get(v_, v_)) for v_ in ("b0", "rp", "cp", "rc")}; F_m = rarity_lomo(mo);')
sub('ph = mdl.predict(PO.feats(Xw)); a = PO.adaptive_a(ev, dom)\n', 'F0_ = PO.feats(Xw); ph = mdl.predict(F0_); kd_ = m3v2.kdist(dom, cap=3); ex_ = extra(p, "train", t, max(kd_, key=kd_.get), t // 288 * 288 - dom * 288); PH = {v_: MM[v_].predict(join(F0_, ex_, {"b0": "base"}.get(v_, v_))) for v_ in MM}; a = PO.adaptive_a(ev, dom)\n')
sub('S = sets(c, p_, ph, a, t, kk)', 'S = sets(c, p_, ph, a, t, kk, None, PH)')
sub('"g170_m3.csv"', '"g214_m3.csv"'); sub('"g170_train.csv"', '"g214_train.csv"')
exec(compile(s, "g170r", "exec"))
