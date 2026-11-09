"""G291 (11/05, ledger #236 stage 0 (b), (c)): M3 v3 dev onset 100 windows (hold-out days not read), G170's flow, k expectation:
  pc4  = A17's onset (= A16's: the onset networks 0.8 + pcL s0-3 0.2, the physics onset adaptive a, the evidence penalty 0.25)
  a0   = B7's onset (the onset networks only, no physics, no pcL, penalty 0.25)        [G170's column]
  a0n  = B7s's onset (as a0 without the penalty)
(c), for the record: on the windows where A17's T+30 set has IoU 0 with a non-empty truth, the k-weighted share of the true T+30 cells
on penalised links (no same-slot evidence on the earlier days). Writes work/t2/g291_m3.csv. Patched from G170 as G280.
usage: TFB_M3VER=v3 py -B -X utf8 g291_b7s_m3.py"""
import sys, pickle
s = open(r"C:/Users/user/tfb/review/g170_onset_instrument.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


L1 = '            S = sets(CTX[key], PN[w], ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"]))\n'
sub('"hid", "hidf", "ac"); ND = 8', '"hid", "hidf", "ac", "a0n", "pc4", "pen"); ND = 8')
sub('"a0": rule(mix(0.0), 0.25),', '"a0": rule(mix(0.0), 0.25), "a0n": mix(0.0),')
sub('    return {v: decode(x)[-1] for v, x in pm.items()}', '    out_ = {v: decode(x)[-1] for v, x in pm.items()}; out_["_ev"] = ev if nd else np.ones_like(ev); return out_')
sub(L1 + '            for v in VARS:\n                sc[v] += pr * iou1(S[v], y, e)\n',
    L1 + '            for v in VARS[:-2]:\n                sc[v] += pr * iou1(S[v], y, e)\n'
    '            S4 = sets(CTX[key], 0.8 * PN[w] + 0.2 * sum(PC_[j][w] for j in range(4)) / 4, ph, a, m["t"], k, PO.adaptive_a(ev2, m["day"])); i4 = iou1(S4["base"], y, e); sc["pc4"] += pr * i4\n'
    '            yt = y & e; sc["pen"] += pr * (float((yt & ~S4["_ev"]).sum() / yt.sum()) if (yt.any() and i4 == 0) else np.nan)\n')
sub('def march(limit=None):', 'PC_ = [pickle.load(open(W + f"probs_m3v3cap__pcL_s{s_}_fold2030-12.pkl", "rb")) for s_ in range(4)]\n\n\ndef march(limit=None):')
sub("t0 = TM(); Dt = train_months(lim)",
    'P8 = pd.read_csv(W + "g280_m3.csv").set_index("w"); print("identity pc4 against g280:", float(np.abs(Dm.set_index("w").pc4 - P8.pc4.reindex(Dm.w).to_numpy()).max()), "| a0 against g170:", float(np.abs(Dm.set_index("w").a0 - pd.read_csv(W + "g170_m3.csv").set_index("w").a0.reindex(Dm.w).to_numpy()).max()), flush=True)\n'
    '    Dm.to_csv(W + "g291_m3.csv", index=False); ch = ~np.isclose(Dm.a0n, Dm.a0); print(f"B7s - B7 on M3 dev onset: {(Dm.a0n - Dm.a0).mean():+.5f} over {len(Dm)} windows, {int(ch.sum())} change | A17 onset {Dm.pc4.mean():.4f}, B7 {Dm.a0.mean():.4f}, B7s {Dm.a0n.mean():.4f}", flush=True)\n'
    '    mi = Dm.pen.notna(); print(f"(c) windows where A17 missed (IoU 0, truth present): {int(mi.sum())}; mean share of their true cells on penalised links {Dm.pen[mi].mean():.3f}; all on penalised links {int((Dm.pen[mi] > 0.999).sum())}", flush=True)\n'
    '    sys.exit()\n    Dt = None')
exec(compile(s, "g170p5", "exec"))
