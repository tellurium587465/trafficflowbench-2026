"""G114 (10/29, ledger #61): the training-month onset frequency of a site as a shrunk term of the decoding-stage evidence
rule (not as a network input, which hurt on the LB in p11G / p13Gfull). Final A onset flow on M3 v3 (q24f CF5 + physics
onset, decoder). Current rule (when earlier days of the month exist): T+30 probabilities x (1 if a queue was seen within
+-2 links on the earlier days, else 0.25). New: x (0.25 + 0.75 eps), eps = (1 - w) e + w f, e = that same-month evidence,
f = a link with training-month T+30 frequency >= 0.1 within +-2 links (rarity_table: panel x weekend x 2-hour slot).
w in (0.1, 0.2, 0.3); w = 0 reproduces A. Writes work/t2/g114_onset_train_prior.csv (A, W1, W2, W3) for judge.py.
usage: TFB_M3VER=v3 py g114_onset_train_prior.py"""
import os, sys
os.environ.setdefault("TFB_M3VER", "v3")
sys.argv = [sys.argv[0]]
src = open(r"C:/Users/user/tfb/review/g108_rectangles.py", encoding="utf-8").read().split("\ncols = {c: {} for c")[0]
exec(compile(src, "g108", "exec"))
RT = PO.rarity_table(); WS = {"A": 0.0, "W1": 0.1, "W2": 0.2, "W3": 0.3}
cols = {c: {} for c in WS}
for i, w in enumerate(G.WID):
    m = G.META[w]
    if m["cond"] != "queue_onset":
        continue
    Xw = np.asarray(G.X[G.OFF[i]:G.OFF[i + 1]]); n = Xw.shape[0]
    pn = sum(wt * PRO[t][w] for t, wt in ON) / sum(wt for _, wt in ON); ph = phys.predict(PO.feats(Xw)); a_ph = PO.adaptive_a(ev, m["day"])
    wkend = (m3v2.MARCH_FIRST_DOW + int(m["day"])) % 7 >= 5
    f = RT.get((m["panel"], wkend, (int(m["t"]) % 288) // 24), np.zeros(n)); fb = G.dil(np.asarray(f[:n]) >= 0.1, 2).astype(float)
    sc = {c: 0.0 for c in WS}
    for k, pr in m3v2.kdist(m["day"]).items():
        p = pn.copy(); p[5] = (1 - a_ph) * p[5] + a_ph * ph
        s_, nd = m3v2.seen(m["panel"], m["t"], k, -6, 12); e = G.dil(s_, 2).astype(float)
        for c, wt in WS.items():
            pb = p.copy()
            if nd:
                pb[-1] = pb[-1] * (0.25 + 0.75 * ((1 - wt) * e + wt * fb))
            sc[c] += pr * G.iou(best_set(pb.ravel(), np.ones(pb.size, bool)).reshape(pb.shape), m)
    for c in WS:
        cols[c][w] = sc[c]
dmax = max(abs(cols["A"][w] - REF.loc[w, "phys"]) for w in cols["A"])
print(f"identity: A vs G93 phys: max |d| {dmax:.2e}")
D = V.frame(cols); D = D[D.cond == "queue_onset"]; D.to_csv(W + "g114_onset_train_prior.csv", index=False)
dv = ~m3v2.is_holdout(D.day); print("dev onset", int(dv.sum()), " | ".join(f"{c} {D[dv][c].mean():.4f}" for c in WS))
