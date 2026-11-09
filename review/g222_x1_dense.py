"""G222 (11/03, ledger #189): X1 on a denser pool of the month's earlier days. The pool nn_data_march_sp1 is build_march.py
with origins every row (SP = 1); a spacing s is recovered by the build's own greedy rule (per panel and condition, an
origin is kept when it is >= s rows after the last kept one), so s = 6 must give nn_data_march bit for bit (identity).
The fine-tune is src/t2_m3v2_x1.py's (t2_nn_finetune.finetune, lr 2e-4, the simulated blanks of M3 v2 excluded) with the
pool restricted to the kept origins; src is not changed.
usage: TFB_M3VER=v3 py g222_x1_dense.py identity
       TFB_M3VER=v3 py g222_x1_dense.py run <tag> <spacing> <epochs> <threads>   -> work/t2/probs_m3v3v2x1_d<spacing>e<epochs>__<tag>.pkl"""
import os, sys, time, pickle, numpy as np
os.environ.setdefault("TFB_M3VER", "v3")
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import NN

DD = "nn_data_march_sp1"; W = "C:/Users/user/tfb/work/t2/"
M = {f: np.load(f"{NN}/{DD}/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "off")}


def keep(sp):
    """indices of the pool kept at spacing sp (the build's greedy rule; the pool is in build order: panel, then t)"""
    out, last = [], {}
    for i, (p, t, c) in enumerate(zip(M["panel"], M["t"].astype(int), M["cond"])):
        if t - last.get((p, c), -10 ** 9) >= sp:
            out.append(i); last[(p, c)] = t
    return np.array(out, int)


if sys.argv[1] == "identity":
    k6 = keep(6); R = {f: np.load(f"{NN}/nn_data_march/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "date", "off")}
    S = {f: np.load(f"{NN}/{DD}/{f}.npy", allow_pickle=True) for f in ("date",)}
    meta = all(np.array_equal(np.asarray(M[f][k6]).astype(str), np.asarray(R[f]).astype(str)) for f in ("panel", "t", "cond")) and np.array_equal(S["date"][k6].astype(str), R["date"].astype(str))
    ok = meta
    if meta:
        for f in ("X", "Y", "EL"):
            a = np.load(f"{NN}/{DD}/{f}.npy", mmap_mode="r"); b = np.load(f"{NN}/nn_data_march/{f}.npy", mmap_mode="r")
            rows = np.concatenate([np.arange(M["off"][k], M["off"][k + 1]) for k in k6])
            ok &= bool(np.array_equal(np.asarray(a[rows]), np.asarray(b), equal_nan=True))
    print(f"identity: spacing 6 from the dense pool -> {len(k6)} windows (nn_data_march {len(R['t'])}), meta {meta}, X/Y/EL equal {ok}")
    for sp in (1, 3, 6):
        print(f"spacing {sp}: {len(keep(sp))} windows")
    sys.exit()

tag, SP, EP, NT = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
import torch
torch.set_num_threads(NT); torch.set_num_threads = lambda n: None
import t2_nn_finetune as FT, m3v2
VER = os.environ["TFB_M3VER"]; META = pickle.load(open(W + f"m3{VER}_meta.pkl", "rb"))["meta"]; byday = {}
for m in META:
    byday.setdefault(m["day"], []).append(m["window_id"])
KEEP = set(keep(SP).tolist()); NOTK = set(range(len(M["t"]))) - KEEP; MP, MT = M["panel"], M["t"]


def excluded(D0, k):
    ex = set(NOTK)
    for p in np.unique(MP):
        hid = m3v2.sim_blank_rows(str(p), D0, k, D0 * 288)
        if hid:
            ex |= {int(i) for i in np.flatnonzero(MP == p) if any(r in hid for r in range(int(MT[i]) - 12, int(MT[i]) + 7))}
    return ex


if len(sys.argv) > 6 and sys.argv[6] == "dry":                 # spacing 6, 4 epochs, day 4, k 2, five windows == the saved X1
    D0, k = 3, 2; wids = byday[D0][:5]; ref = pickle.load(open(W + f"probs_m3{VER}v2x1__{tag}.pkl", "rb"))
    net, mu, sd, cm, n = FT.finetune(tag, DD, D0 + 1 - k, D0, lr=2e-4, epochs=EP, exclude=excluded(D0, k)); pr = FT.predict(net, mu, sd, cm, f"nn_test_m3{VER}", wids)
    print(f"dry: {n} windows, max |p - saved X1| {max(float(np.abs(pr[w] - ref[(w, k)]).max()) for w in wids):.2e}"); sys.exit()
arm = f"_d{SP}e{EP}"; f_out = W + f"probs_m3{VER}v2x1{arm}__{tag}.pkl"; t0 = time.time()
base = pickle.load(open(W + f"probs_m3{VER}__{tag}.pkl", "rb")); out = {}; nft = nwin = 0; done = set()
if os.path.exists(f_out + ".part"):                             # 11/03: restart-safe, one checkpoint per day (the PC rebooted at 05:51)
    out, done = pickle.load(open(f_out + ".part", "rb")); print(f"resumed: {len(done)} days done", flush=True)
for D0, wids in sorted(byday.items()):
    if D0 in done:
        continue
    for w in wids:
        out[(w, 0)] = base[w]
    for k in range(1, min(6, D0) + 1):
        D1 = D0 + 1
        net, mu, sd, cm, n = FT.finetune(tag, DD, D1 - k, D1 - 1, lr=2e-4, epochs=EP, exclude=excluded(D0, k))
        pr = FT.predict(net, mu, sd, cm, f"nn_test_m3{VER}", wids); nft += 1; nwin += n
        for w in wids:
            out[(w, k)] = pr[w]
    done.add(D0); pickle.dump((out, done), open(f_out + ".part", "wb")); print(f"day {D0 + 1} done, {time.time() - t0:.0f}s", flush=True)
pickle.dump(out, open(f_out, "wb")); os.remove(f_out + ".part")
print(f"{tag} {arm}: {nft} fine-tunes, {nwin} windows seen, {time.time() - t0:.0f}s -> {f_out}", flush=True)
