"""G141 (10/31, ledger #101-#103): X1 variants on M3 v3 DEV windows, the path of src/t2_m3v2_x1.py (each dev day, k = 0 .. min(6, day), simulated blanks excluded), networks c24f_attnglob s0 / s1.
Held-out days are not computed. src is not changed: the fine-tune below is t2_nn_finetune.finetune with a window weight and a reused network.
  sd    : #101 same-day X1, one fine-tune per (day, hour, k): days D-k .. D-1 + the same day's pool windows with T' + 6 <= c, c = the hour's first row <= T (k = 0: the same day only)
  hl<h> : #102 X1 with the window loss weighted 0.5 ** ((D - 1 - day) / h), normalised to mean 1 over the fine-tune's windows
  tta   : #103 plain X1, predictions at the flow scales SC (the columns of t2_nn3 JMODE=w), onset windows by the saved q24f networks in `tton`
Writes work/t2/x1_arms/parts/<arm>__<tag>__<D0>.pkl per day (restart-safe): {(window_id, k[, scale]): (6, n)}, log work/t2/x1_arms/<arm>__<tag>.log
usage: TFB_M3VER=v3 py g141_x1_arms.py plan | dry <tag> | fit <arm> <tag> | tton <tag[,tag...]>"""
import os, sys, time, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, torch
torch.set_num_threads(3); torch.set_num_threads = lambda n: None
R = r"C:/Users/user/tfb/"; W = R + "work/t2/"; OUT = W + "x1_arms/"; os.makedirs(OUT + "parts", exist_ok=True); sys.path.insert(0, R + "src")
import t2_nn_finetune as FT, t2_nn_cpu_infer as CI, m3v2
from common import NN
SC = (0.8, 0.9, 1.0, 1.1, 1.2); TAGS = [f"c24f_attnglob_s{s}" for s in (0, 1)]; hold = lambda d: (d + 1) % 3 == 0
META = pickle.load(open(W + "m3v3_meta.pkl", "rb"))["meta"]; DEV = [m for m in META if m["cond"] == "queue_ongoing" and not hold(m["day"])]
MD = FT.data("nn_data_march"); MP = np.load(f"{NN}/nn_data_march/panel.npy", allow_pickle=True); MT = np.load(f"{NN}/nn_data_march/t.npy"); PANELS = [str(p) for p in np.unique(MP)]
N = FT.NAMES; idx = lambda c: N.index(c)
QC = [idx(c) for c in ("q_T", "q_T1", "q_T2", "q_m3", "q_m12", "q_max", "last_q", "q_trend", "s_flow", "dem10_c", "dem20_c", "dem30_c", "dem30_over_dnmin") + tuple(f"arr{i}" for i in range(7)) if c in N]
QC += [idx(f"nb{o}_{k}") for o in range(-5, 6) for k in ("q", "qmax") if f"nb{o}_{k}" in N]


def excluded(D0, k, cut):
    """the March pool windows whose rows T-12 .. T+6 touch the simulated blanks (from the start of day D0 - k up to row cut), as t2_m3v2_x1.excluded (cut = D0 * 288 there)"""
    ex = set()
    for p in PANELS:
        hid = m3v2.sim_blank_rows(p, D0, k, cut)
        if hid:
            ex.update(int(i) for i in np.flatnonzero(MP == p) if any(r in hid for r in range(int(MT[i]) - 12, int(MT[i]) + 7)))
    return ex


class Net:
    def __init__(self, tag):
        self.net, self.mu, self.sd, self.cm = CI.load_model(tag); self.s0 = {k: v.clone() for k, v in self.net.state_dict().items()}

    def finetune(self, ks, wts=None, lr=2e-4, epochs=4, batch_size=32, wd=1e-3, wpow=1.0, fdrop=0.05, seed=0):
        net, mu, sd, cm = self.net, self.mu, self.sd, self.cm; net.load_state_dict(self.s0); ks = np.asarray(ks, dtype=int)
        if len(ks) == 0:
            net.eval(); return 0
        wts = np.ones(len(ks)) if wts is None else np.asarray(wts, float) / np.mean(wts); gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); net.train()
        for ep in range(epochs):
            order = rng.permutation(len(ks))
            for b0 in range(0, len(order), batch_size):
                sel = order[b0:b0 + batch_size]; xb, y, e, m = FT.batch([(MD, int(ks[i])) for i in sel], cm, mu, sd, fdrop, gen)
                l = torch.nn.functional.binary_cross_entropy_with_logits(net(xb, m)[..., :6], y, reduction="none") * e
                wb = 100.0 * torch.clamp((y * e).sum(dim=(1, 2)), min=3) ** (-wpow) * torch.tensor(wts[sel], dtype=torch.float32)
                loss = (l.sum(dim=(1, 2)) * wb).sum() / (e.sum() + 1); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval(); return len(ks)

    def predict(self, wids, scale=1.0):
        X = np.load(f"{NN}/nn_test_m3v3/X.npy", mmap_mode="r"); off = np.load(f"{NN}/nn_test_m3v3/off.npy"); Wd = [str(w) for w in np.load(f"{NN}/nn_test_m3v3/window_id.npy", allow_pickle=True)]; cond = np.load(f"{NN}/nn_test_m3v3/cond.npy", allow_pickle=True)
        pos = {w: i for i, w in enumerate(Wd)}; out = {}
        with torch.no_grad():
            for w in wids:
                i = pos[w]; a, b = off[i], off[i + 1]; v = torch.from_numpy(np.array(X[a:b], dtype=np.float32))[None]
                if scale != 1.0:
                    v[..., QC] *= scale
                p = torch.sigmoid(self.net(FT.enc(v, self.cm, self.mu, self.sd), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.copy()
                if cond[i] == "queue_onset":
                    p[:5] = 0.0
                out[w] = p
        return out


def pool(D0, k, cut=None):
    D1 = D0 + 1; ex = excluded(D0, k, D0 * 288 if cut is None else cut)
    ks = np.flatnonzero((MD["day"] >= D1 - k) & (MD["day"] <= D1 - 1))
    if cut is not None:
        ks = np.concatenate([ks, np.flatnonzero((MD["day"] == D1) & (MT + 6 <= cut))])
    return np.array([i for i in ks if int(i) not in ex], dtype=int)


def groups(arm):
    g = {}
    for m in DEV:
        g.setdefault((m["day"], m["t"] // 12 if arm == "sd" else None), []).append(m["window_id"])
    return g


MODE = sys.argv[1]
if MODE == "plan":
    for arm in ("hl", "sd"):
        tot = n = 0
        for (D0, c), wids in groups(arm).items():
            for k in range(0 if arm == "sd" else 1, min(6, D0) + 1):
                tot += len(pool(D0, k, None if c is None else c * 12)); n += 1
        print(f"{arm}: {n} fine-tunes, {tot} windows x 4 epochs = {tot * 4} window-epochs", flush=True)
    sys.exit()
if MODE == "dry":
    tag = sys.argv[2]; nt = Net(tag); saved = pickle.load(open(W + f"probs_m3v3v2x1__{tag}.pkl", "rb")); D0, k = 4, 2; wids = [m["window_id"] for m in DEV if m["day"] == D0][:5]; t0 = time.time()
    n = nt.finetune(pool(D0, k)); print(f"identity plain X1 day {D0} k {k}: {n} windows {time.time() - t0:.1f}s max |dp| against the saved file {max(float(np.abs(nt.predict([w])[w] - saved[(w, k)]).max()) for w in wids):.2e}", flush=True)
    t0 = time.time(); ks = pool(D0, k); d = MD["day"][ks]; n = nt.finetune(ks, np.ones(len(ks))); print(f"unit weights: max |dp| {max(float(np.abs(nt.predict([w])[w] - saved[(w, k)]).max()) for w in wids):.2e}", flush=True)
    t0 = time.time(); n = nt.finetune(ks, 0.5 ** ((D0 - d) / 3.0)); p3 = nt.predict(wids[:1])[wids[0]]; print(f"hl3: {n} windows {time.time() - t0:.1f}s, mean |dp| against plain {float(np.abs(p3 - saved[(wids[0], k)]).mean()):.2e}", flush=True)
    m = next(x for x in DEV if x["window_id"] == wids[0]); c = m["t"] // 12 * 12; t0 = time.time(); ks = pool(D0, k, c); n = nt.finetune(ks); print(f"sd: row {m['t']} cut {c}: {n} windows ({int((MD['day'][ks] == D0 + 1).sum())} same day) {time.time() - t0:.1f}s; all rows of the same-day windows <= cut: {bool((MT[ks][MD['day'][ks] == D0 + 1] + 6 <= c).all())}", flush=True)
    nt.finetune([]); base = pickle.load(open(W + f"probs_m3v3__{tag}.pkl", "rb")); print(f"no fine-tune against the saved base: max |dp| {max(float(np.abs(nt.predict([w])[w] - base[w]).max()) for w in wids):.2e}; scale 1.1 mean |dp| {float(np.abs(nt.predict(wids[:1], 1.1)[wids[0]] - base[wids[0]]).mean()):.2e}", flush=True)
    sys.exit()
if MODE == "tton":                                               # the onset flow's saved networks at the scales, dev onset windows
    wids = [m["window_id"] for m in META if m["cond"] == "queue_onset" and not hold(m["day"])]
    for tag in sys.argv[2].split(","):
        f = OUT + f"parts/tton__{tag}.pkl"
        if os.path.exists(f):
            continue
        t0 = time.time(); nt = Net(tag); nt.finetune([]); pickle.dump({s: nt.predict(wids, s) for s in SC}, open(f, "wb")); print(tag, f"{time.time() - t0:.0f}s", flush=True)
    sys.exit()
arm, tag = sys.argv[2], sys.argv[3]; nt = Net(tag); log = open(OUT + f"{arm}__{tag}.log", "a"); t0 = time.time(); G = groups(arm); nft = 0
for D0 in sorted({d for d, _ in G}):
    f = OUT + f"parts/{arm}__{tag}__{D0}.pkl"
    if os.path.exists(f):
        continue
    out = {}
    for (d, c), wids in sorted(G.items(), key=str):
        if d != D0:
            continue
        for k in range(0, min(6, D0) + 1):
            if arm == "sd":
                ks = pool(D0, k, c * 12); nt.finetune(ks); nft += 1; pr = nt.predict(wids); out.update({(w, k): pr[w] for w in wids})
            elif arm.startswith("hl"):
                ks = pool(D0, k) if k else np.zeros(0, int); wt = 0.5 ** ((D0 - MD["day"][ks]) / float(arm[2:])) if k else None; nt.finetune(ks, wt); nft += k > 0; pr = nt.predict(wids); out.update({(w, k): pr[w] for w in wids})
            else:
                nt.finetune(pool(D0, k) if k else []); nft += k > 0
                for s in SC:
                    pr = nt.predict(wids, s); out.update({(w, k, s): pr[w] for w in wids})
    pickle.dump(out, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f); log.write(f"day {D0 + 1}: {nft} fine-tunes, {time.time() - t0:.0f}s {time.strftime('%H:%M:%S')}\n"); log.flush()
log.write(f"ALLDONE {time.time() - t0:.0f}s\n"); log.close()
