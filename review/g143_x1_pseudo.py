"""G143 (10/31, ledger #104): X1 with pseudo-labels, M3 v3 DEV ongoing, the path and fine-tune of G141. The pool windows' unobserved future cells, which carry no loss in X1, get the saved network's
own hard label where it is confident (p >= HI -> 1, p <= LO -> 0) at weight WT; observed cells keep their labels. The teacher is the network before the fine-tune, computed once on the pool.
Writes work/t2/x1_arms/parts/pl__<tag>__<D0>.pkl as G141, log work/t2/x1_arms/pl__<tag>.log. G142 reads the arm once it is added to its list.
usage: TFB_M3VER=v3 py g143_x1_pseudo.py dry <tag> | fit <tag>"""
import os, sys, time, pickle
os.environ.setdefault("TFB_M3VER", "v3")
MODE_ = sys.argv[1]; TAG = sys.argv[2]
exec(compile(open(r"C:/Users/user/tfb/review/g141_x1_arms.py", encoding="utf-8").read().split("\nMODE = sys.argv[1]")[0], "g141", "exec"))
HI, LO, WT = 0.9, 0.1, 0.5


class NetPL(Net):
    def teacher(self):
        self.net.load_state_dict(self.s0); self.net.eval(); off = MD["off"]; T = []
        with torch.no_grad():
            for k in range(len(off) - 1):
                xb, y, e, m = FT.batch([(MD, k)], self.cm, self.mu, self.sd, 0.0, None); T.append(torch.sigmoid(self.net(xb, m)[..., :6])[0].numpy().copy())
        self.T = T

    def finetune(self, ks, lr=2e-4, epochs=4, batch_size=32, wd=1e-3, wpow=1.0, fdrop=0.05, seed=0):
        net, mu, sd, cm = self.net, self.mu, self.sd, self.cm; net.load_state_dict(self.s0); ks = np.asarray(ks, dtype=int)
        if len(ks) == 0:
            net.eval(); return 0, 0.0
        gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed); opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); net.train(); added = total = 0
        for ep in range(epochs):
            order = rng.permutation(len(ks))
            for b0 in range(0, len(order), batch_size):
                sel = order[b0:b0 + batch_size]; xb, y, e, m = FT.batch([(MD, int(ks[i])) for i in sel], cm, mu, sd, fdrop, gen); tp = torch.zeros_like(y)
                for j, i in enumerate(sel):
                    t = torch.from_numpy(self.T[int(ks[i])]); tp[j, :t.shape[0]] = t
                conf = (e == 0) & (m[:, :, None] > 0) & ((tp >= HI) | (tp <= LO)); y2 = torch.where(conf, (tp >= HI).float(), y); e2 = torch.where(conf, torch.full_like(e, WT), e); added += int(conf.sum()); total += int((e > 0).sum())
                l = torch.nn.functional.binary_cross_entropy_with_logits(net(xb, m)[..., :6], y2, reduction="none") * e2
                wb = 100.0 * torch.clamp((y * e).sum(dim=(1, 2)), min=3) ** (-wpow); loss = (l.sum(dim=(1, 2)) * wb).sum() / (e2.sum() + 1); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval(); return len(ks), added / max(total, 1)


nt = NetPL(TAG); t0 = time.time(); nt.teacher(); print(f"teacher {len(nt.T)} windows {time.time() - t0:.0f}s", flush=True); G_ = groups("hl")
if MODE_ == "dry":
    saved = pickle.load(open(W + f"probs_m3v3v2x1__{TAG}.pkl", "rb")); D0, k = 4, 2; wids = [m["window_id"] for m in DEV if m["day"] == D0][:5]; ks = pool(D0, k); HI_, HI = HI, 2.0; LO_, LO = LO, -1.0
    t0 = time.time(); n, r = nt.finetune(ks); print(f"identity, no pseudo cell: {n} windows {time.time() - t0:.1f}s ratio {r:.3f} max |dp| against the saved file {max(float(np.abs(nt.predict([w])[w] - saved[(w, k)]).max()) for w in wids):.2e}", flush=True)
    HI, LO = HI_, LO_; n, r = nt.finetune(ks); print(f"pseudo cells per observed cell {r:.2f}, mean |dp| against plain {float(np.abs(nt.predict(wids[:1])[wids[0]] - saved[(wids[0], k)]).mean()):.2e}", flush=True); sys.exit()
log = open(OUT + f"pl__{TAG}.log", "a"); t0 = time.time(); nft = 0
for D0 in sorted({d for d, _ in G_}):
    f = OUT + f"parts/pl__{TAG}__{D0}.pkl"
    if os.path.exists(f):
        continue
    out = {}; wids = G_[(D0, None)]
    for k in range(0, min(6, D0) + 1):
        nt.finetune(pool(D0, k) if k else []); nft += k > 0; pr = nt.predict(wids); out.update({(w, k): pr[w] for w in wids})
    pickle.dump(out, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f); log.write(f"day {D0 + 1}: {nft} fine-tunes, {time.time() - t0:.0f}s {time.strftime('%H:%M:%S')}\n"); log.flush()
log.write(f"ALLDONE {time.time() - t0:.0f}s\n"); log.close()
