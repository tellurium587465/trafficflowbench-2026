"""G241 (11/03, ledger #204, the strategy's 22:30 request (c), DATA_RULE_USAGE_AUDIT's unused-but-allowed data): G170's onset
instrument with three arms more, decode only (nothing retrained, the final A's onset flow):
  mk2, mk3 (c3, audit 5): the onset decode keeps at least 2 / 3 links at T+30 (the selector's requires_queue_in_horizon as a prior):
           when best_set picks fewer, the next links by probability are added
  o5   (c5, audit 8): another selected window of the same corridor and day with T' + 30 <= T (a queue was in its horizon, rulings
           742350 / 745700: conditioning on the selection procedure, data at or before T) relaxes the evidence rule's factor 0.25 -> 0.5.
           March dev: the official March windows (window_index.csv); January / February: the selector windows of T.META
  xm   (c4, audit 7): January / February windows on days 1-3 read the previous month's last days as their earlier days (k drawn as
           on day 4, the train split runs June-February without a gap): the proxy of April 1-3 reading March's end. Not on March dev
           (no February rows before March in M3)
Readings as G170: March dev (hold-out days not read) by judge; January / February L (eligible cells) and F (all finite cells).
usage: TFB_M3VER=v3 py -B -X utf8 g241_rule_margins.py [time | run]"""
import sys
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "mk2", "mk3", "o5"); ND = 8')
sub('def sets(c, p_, ph, a, t, k, a2=None):', 'def sets(c, p_, ph, a, t, k, a2=None, om=False):')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}\n',
    '''    pm["o5"] = rule(pa, 0.5) if om else b; out = {v: decode(x)[-1] for v, x in pm.items()}
    for K_ in (2, 3):
        d_ = out["base"].copy()
        if d_.sum() < K_:
            d_[[j for j in np.argsort(-b[-1], kind="stable") if not d_[j]][:K_ - int(d_.sum())]] = True
        out[f"mk{K_}"] = d_
    return out


OFF_M = {}


def om_march(p, t):                                             # an official March window of the corridor, same day, T' + 30 <= T
    if p not in OFF_M:
        from common import REL
        P_ = load(p, "validation"); ix = {x: i for i, x in enumerate(P_.times)}; wi = pd.read_csv(REL / "task2" / p / "validation" / "window_index.csv")
        OFF_M[p] = np.array([ix[x] for x in wi.forecast_origin])
    o_ = OFF_M[p]; return bool(((o_ + 6 <= t) & (o_ // 288 == t // 288)).any())


def om_train(k, p, t):                                          # another selector window of the corridor, same day, T' + 30 <= T
    if "PS" not in OFF_M:
        OFF_M["PS"] = np.array([str(x) for x in T.META["panel"]])
    tt = np.asarray(T.META["t"]); ok = np.asarray(T.META["is_sel"], bool) & (OFF_M["PS"] == p) & (tt + 6 <= t) & (tt // 288 == t // 288); ok[k] = False; return bool(ok.any())
''')
sub('            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))',
    '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]), om=om_march(m["panel"], m["t"]))')
sub('        rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, **sc));',
    '        rows.append(dict(w=w, cond=m["cond"], panel=m["panel"], day=m["day"], hour=(m["t"] % 288) // 12, om=om_march(m["panel"], m["t"]), **sc));')
sub('                for kk, pr in m3v2.kdist(dom, cap=3).items():\n                    S = sets(c, p_, ph, a, t, kk)\n',
    '                om = om_train(k, p, t); xL = xF = None\n                if dom < 3:\n                    xL = xF = 0.0\n                    for kk, pr in m3v2.kdist(3, cap=3).items():\n                        S = sets(c, p_, ph, a, t, kk, om=om); xL += pr * iou1(S["base"], yL, eL); xF += pr * iou1(S["base"], yF, eF)\n'
    '                for kk, pr in m3v2.kdist(dom, cap=3).items():\n                    S = sets(c, p_, ph, a, t, kk, om=om)\n')
sub('rows.append(dict(w=int(k), month=mo, panel=p, day=dom, nsel=int(sel.sum()), nY=int(yF.sum()), a=a, **sc,',
    'rows.append(dict(w=int(k), month=mo, panel=p, day=dom, nsel=int(sel.sum()), nY=int(yF.sum()), a=a, om=om, L_xm=sc["L_base"] if xL is None else xL, F_xm=sc["F_base"] if xF is None else xF, **sc,')
sub('Dm.to_csv(W + "g170_m3.csv", index=False); Dt.to_csv(W + "g170_train.csv", index=False)',
    'Dm.to_csv(W + "g241_m3.csv", index=False); Dt.to_csv(W + "g241_train.csv", index=False); print("o5 windows: March dev", int(Dm.om.sum()), "of", len(Dm), "| Jan, Feb", int(Dt.om.sum()), "of", len(Dt), "| xm windows (days 1-3)", int((Dt.day < 3).sum()), flush=True)\n'
    '    for v, lab in (("xm", "xm (previous month as earlier days)"),):\n'
    '        for mo in ("pool", "2031-01", "2031-02"):\n'
    '            D = Dt if mo == "pool" else Dt[Dt.month == mo]; lm, ls = ratio(D, (D[f"L_{v}"] - D.L_base).to_numpy(), np.ones(len(D))); fm, fs = ratio(D, (D[f"F_{v}"] - D.F_base).to_numpy(), np.ones(len(D)))\n'
    '            print(f"  {lab} {mo}: L {lm:+.4f} ({ls:.4f}) | F {fm:+.4f} ({fs:.4f}) | windows changed (F) {int((~np.isclose(D[f\'F_{v}\'], D.F_base)).sum())}", flush=True)')
exec(compile(s, "g170c", "exec"))
