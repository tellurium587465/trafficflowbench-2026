"""G225 (11/03, ledger #189, January / February arm): G135's rolling-origin X1 on 2031-01 / 2031-02 with the fold-2 B0 networks
(capB0_s0-3_fold2030-12, never trained on those months), the fine-tune pool replaced by g224's nn_data_jf_sp1 at a given
spacing, so the base arm (spacing 6, 4 epochs) and the dense arm come from one builder and are paired. As in G135, each
day d >= 2 fine-tunes on days 1 .. d-1 of the same month with the number of windows March gives at the same spacing on
those days (nn_data_march_sp1 thinned the same way), lr 2e-4, seed 0, and predicts day d's selector ongoing windows
(nn_data_afdem is_sel); day 1 keeps the base.
  fit <tag> <spacing> <epochs> <threads> : work/t2/x1_months_dense/d<spacing>e<epochs>/<tag>.pkl = {window index: (base, x1)}
  score : G135's score (raw best_set and the SCG flow, 4-seed mean) per arm, then dense - base paired, months merged
usage: py g225_x1_jf_dense.py fit capB0_s0_fold2030-12 6 4 4 | TFB_M3VER=v3 py g225_x1_jf_dense.py score d1e1"""
import os, sys, time, pickle
import numpy as np
R0 = r"C:/Users/user/tfb/"; sys.path.insert(0, R0 + "src"); OUT0 = R0 + "work/t2/x1_months_dense/"; MONTHS = ("2031-01", "2031-02")
MODE = sys.argv[1]
from common import NN


def keep(M, sp):
    out, last = [], {}
    for i, (p, t, c) in enumerate(zip(M["panel"], M["t"].astype(int), M["cond"])):
        if t - last.get((p, c), -10 ** 9) >= sp:
            out.append(i); last[(p, c)] = t
    return np.array(out, int)


if MODE == "fit":
    tag, SP, EP, NT = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    import torch
    torch.set_num_threads(NT); torch.set_num_threads = lambda n: None
    import t2_nn_finetune as FT
    OUT = OUT0 + f"d{SP}e{EP}/"; os.makedirs(OUT, exist_ok=True); f = OUT + tag + ".pkl"; t0 = time.time()
    J = {k: np.load(f"{NN}/nn_data_jf_sp1/{k}.npy", allow_pickle=True) for k in ("panel", "t", "cond")}; NOTK = set(range(len(J["t"]))) - set(keep(J, SP).tolist())
    MM = {k: np.load(f"{NN}/nn_data_march_sp1/{k}.npy", allow_pickle=True) for k in ("panel", "t", "cond", "date")}; km = keep(MM, SP); mday = np.array([int(str(x)[8:10]) for x in MM["date"]])[km]
    nm = lambda d: int((mday <= d - 1).sum())
    D = FT.data("nn_data_afdem"); isel = np.load(f"{NN}/nn_data_afdem/is_sel.npy"); cond = np.load(f"{NN}/nn_data_afdem/cond.npy", allow_pickle=True); mon = np.array([str(d)[:7] for d in D["date"]])

    def pred(net, mu, sd, cm, ks):
        out = {}
        with torch.no_grad():
            for k in ks:
                a, b = int(D["off"][k]), int(D["off"][k + 1]); v = torch.from_numpy(np.array(D["X"][a:b], dtype=np.float32))[None]
                out[int(k)] = torch.sigmoid(net(FT.enc(v, cm, mu, sd), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.copy()
        return out

    net0, mu0, sd0, cm0 = FT.CI.load_model(tag); res = {}; nft = 0; done = set()
    if os.path.exists(f + ".part"):                             # 11/03: restart-safe, one checkpoint per day
        res, done = pickle.load(open(f + ".part", "rb")); print("resumed", len(done), "days", flush=True)
    for mo in MONTHS:
        ks = np.flatnonzero(isel & (mon == mo) & (cond == "queue_ongoing")); base = pred(net0, mu0, sd0, cm0, ks)
        for d in sorted(set(D["day"][ks])):
            if (mo, int(d)) in done:
                continue
            kd = ks[D["day"][ks] == d]
            if d == 1:
                x1 = {int(k): base[int(k)] for k in kd}
            else:
                net, mu, sd, cm, n = FT.finetune(tag, "nn_data_jf_sp1", 1, d - 1, lr=2e-4, epochs=EP, wpow=1.0, month=mo, exclude=NOTK, max_n=nm(d)); x1 = pred(net, mu, sd, cm, kd); nft += 1
            res.update({k: (base[k], x1[k]) for k in x1}); done.add((mo, int(d))); pickle.dump((res, done), open(f + ".part", "wb"))
        print(tag, mo, "done", nft, f"{time.time() - t0:.0f}s", flush=True)
    pickle.dump(res, open(f, "wb")); os.remove(f + ".part"); print("ALLDONE", tag, f"{time.time() - t0:.0f}s", flush=True); sys.exit()

# score: G135's score on each arm's directory (its OUT and file names), then the paired difference of the x1 columns
import importlib.util
import pandas as pd
src = open(R0 + "review/g135_x1_months.py", encoding="utf-8").read(); body = src[src.index('os.environ.setdefault("TFB_M3VER", "v3")'):]
body = body[:body.index("D = pd.DataFrame(rows)")] + "D = pd.DataFrame(rows)\n"
FR = {}; ARM = sys.argv[2]                                       # before the loop: G135's body rewrites sys.argv
for arm in ("d6e4", ARM):
    ns = dict(__name__="g135s", OUT=OUT0 + arm + "/", MONTHS=MONTHS, R0=R0, np=np, os=os, pickle=pickle, sys=sys, importlib=importlib)
    exec(compile(body, "g135score", "exec"), ns); FR[arm] = ns["D"].set_index("w")
a, b = FR["d6e4"], FR[ARM]; a = a.loc[b.index]


def boot(d, cl, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); N = np.bincount(ids)
    Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n_boot); return float(((Rm @ M) / (Rm @ N)).std())


cl = (b.month + "_" + b.panel + "_" + b.day.astype(str)).to_numpy()
for col in ("raw_x1", "scg_x1"):
    d = (b[col] - a[col]).to_numpy(); se = boot(d, cl); res = [f"{col}: merged {d.mean():+.4f} (SE {se:.4f}, z {d.mean() / max(se, 1e-12):+.2f})"]
    for mo in MONTHS:
        m = (b.month == mo).to_numpy(); res.append(f"{mo} {d[m].mean():+.4f} (SE {boot(d[m], cl[m]):.4f})")
    print(f"{ARM} - d6e4, {len(d)} windows | " + " | ".join(res), flush=True)
print(f"identity: the two arms' base columns max |d| {float(np.max(np.abs(b.raw_base - a.raw_base))):.2e}")
