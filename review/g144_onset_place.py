"""G144 (10/31, ledger #105 gate 1, strategy D1): can a small link-sequence model on inputs <= T rank the T+30 onset site better than the physics onset LightGBM?
Inputs per link: t2_phys_onset.feats (within-window percentiles of the non-memory inputs) + capacity and lane ratios to the next link + upstream flow per lane over 3 links. No position, time of day or static id.
Model: 3 dilated residual Conv1d blocks over the links + the window mean, one logit per link. Loss: listwise softmax CE over the eligible links, target = the T+30 queued eligible links.
Reads, per window on the eligible links at T+30 (windows with both classes): AP (main, ledger #105: AUC is saturated at 0.99 on M3 by the many free-flow links), AUC, hit@1 (the top link is queued):
  R1 fit on 2030-06 .. 2030-11, read on the out-of-fold months 2030-12 .. 2031-02 (LightGBM refitted on the same rows with t2_phys_onset's parameters)
  R2 fit on the 9 months, read on M3 v3 dev onset (100 windows, held-out days are not read; LightGBM = the saved physon_base.txt), also J11's 88 precursor windows and final A's onset probability (record)
usage: TFB_M3VER=v3 py g144_onset_place.py dry | run"""
import os, sys, time, pickle
os.environ.setdefault("TFB_M3VER", "v3")
import numpy as np, pandas as pd, torch, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
torch.set_num_threads(3)
R = r"C:/Users/user/tfb/"; W = R + "work/t2/"; OUT = W + ("onset_place_ns/" if os.environ.get("G144_NOSTATIC") == "1" else "onset_place/"); os.makedirs(OUT, exist_ok=True); sys.path.insert(0, R + "src")
import t2_phys_onset as PO
from common import NN
NOSTATIC = os.environ.get("G144_NOSTATIC") == "1"   # the variant added after gate 1 (ledger #105): without the two static ratios
MODE = sys.argv[1]; N = PO.NAMES; ix = N.index; EP, SEEDS = (2, 1) if MODE == "dry" else (20, 3); T00 = time.time()


def extra(Xw):
    cap = Xw[:, ix("capl")] * Xw[:, ix("lanes")]; lanes = Xw[:, ix("lanes")]; q = np.nan_to_num(Xw[:, ix("q_m3")]) / np.maximum(lanes, 1)
    up3 = np.array([q[max(i - 3, 0):i].mean() if i else q[0] for i in range(len(q))]) / np.maximum(Xw[:, ix("capl")], 1)
    return up3[:, None] if NOSTATIC else np.column_stack([np.r_[cap[1:] / np.maximum(cap[:-1], 1), 1.0], np.r_[lanes[1:] / np.maximum(lanes[:-1], 1), 1.0], up3])


def inputs(Xw):
    return np.column_stack([PO.feats(Xw), extra(Xw)]).astype(np.float32)


def load_train():
    d = "nn_data_afdem"; X = np.load(f"{NN}/{d}/X.npy", mmap_mode="r"); Y = np.load(f"{NN}/{d}/Y.npy", mmap_mode="r"); E = np.load(f"{NN}/{d}/EL.npy", mmap_mode="r")
    off = np.load(f"{NN}/{d}/off.npy"); cond = np.load(f"{NN}/{d}/cond.npy", allow_pickle=True); date = np.load(f"{NN}/{d}/date.npy", allow_pickle=True); out = []
    ks = np.flatnonzero(cond == "queue_onset"); ks = ks[::40] if MODE == "dry" else ks
    for k in ks:
        a, b = off[k], off[k + 1]; y = np.asarray(Y[a:b, 5]).astype(bool); e = np.asarray(E[a:b, 5]).astype(bool)
        if (y & e).any():
            out.append(dict(x=inputs(np.asarray(X[a:b])), y=y, e=e, month=str(date[k])[:7], cl=f"{date[k]}"))
    return out


class Blk(torch.nn.Module):
    def __init__(s, h, d):
        super().__init__(); s.c1 = torch.nn.Conv1d(h, h, 5, padding=2 * d, dilation=d); s.c2 = torch.nn.Conv1d(h, h, 5, padding=2 * d, dilation=d)

    def forward(s, x):
        return x + s.c2(torch.relu(s.c1(torch.relu(x))))


class Net(torch.nn.Module):
    def __init__(s, f, h=32):
        super().__init__(); s.i = torch.nn.Conv1d(f, h, 1); s.b = torch.nn.ModuleList([Blk(h, d) for d in (1, 2, 4)]); s.o = torch.nn.Conv1d(2 * h, 1, 1)

    def forward(s, x):
        z = s.i(x.transpose(1, 2))
        for b in s.b:
            z = b(z)
        return s.o(torch.cat([z, z.mean(2, keepdim=True).expand_as(z)], 1))[:, 0]


def fit(rows, seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); net = Net(rows[0]["x"].shape[1]); opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3); net.train()
    for ep in range(EP):
        tot = 0.0
        for i in rng.permutation(len(rows)):
            r = rows[i]; l = net(torch.from_numpy(r["x"])[None])[0]; e = torch.from_numpy(r["e"]); y = torch.from_numpy(r["y"] & r["e"]).float(); lp = torch.log_softmax(l[e], 0); loss = -(y[e] * lp).sum() / y[e].sum()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); tot += float(loss)
        print(f"seed {seed} epoch {ep} loss {tot / len(rows):.3f} ({time.time() - T00:.0f}s)", flush=True)
    return net.eval()


def score(nets, x):
    with torch.no_grad():
        return np.mean([torch.softmax(n(torch.from_numpy(x)[None])[0], 0).numpy() for n in nets], axis=0)


def auc(s, y, e):
    return np.array([roc_auc_score(y[e], s[e]), average_precision_score(y[e], s[e]), float(y[e][np.argmax(s[e])])]) if 0 < (y & e).sum() < e.sum() else np.full(3, np.nan)


def boot(d, cl, n=2000):
    rng = np.random.default_rng(0); ids = pd.factorize(cl)[0]; M = np.bincount(ids, d); C = np.bincount(ids); Rm = rng.multinomial(len(M), np.full(len(M), 1.0 / len(M)), size=n); return float(((Rm @ M) / (Rm @ C)).std())


def line(nm, A, B, cl):
    for j, mt_ in enumerate(("AUC", "AP", "hit@1")):
        a, b = np.stack(A)[:, j], np.stack(B)[:, j]; ok = np.isfinite(a) & np.isfinite(b); d = (a - b)[ok]; se = boot(d, np.asarray(cl)[ok]); print(f"{nm} {mt_}: windows {ok.sum()} new {a[ok].mean():.4f} existing {b[ok].mean():.4f} diff {d.mean():+.4f} (SE {se:.4f})", flush=True)


rows = load_train(); print("onset windows with a T+30 queue", len(rows), f"{time.time() - T00:.0f}s", flush=True)
PF = len(PO.COLS); LB = lambda rs: lgb.train(PO.PRM, lgb.Dataset(np.vstack([r["x"][r["e"], :PF] for r in rs]), np.concatenate([r["y"][r["e"]].astype(int) for r in rs]), free_raw_data=False), 300)
tr = [r for r in rows if r["month"] < "2030-12"]; te = [r for r in rows if r["month"] >= "2030-12"]; print("R1 train", len(tr), "read", len(te), flush=True)
nets = [fit(tr, s) for s in range(SEEDS)]; m = LB(tr); A = np.stack([auc(score(nets, r["x"]), r["y"], r["e"]) for r in te]); B = np.stack([auc(m.predict(r["x"][:, :PF]), r["y"], r["e"]) for r in te]); line("R1 out-of-fold months", A, B, [r["cl"] for r in te])
pd.DataFrame(dict(month=[r["month"] for r in te], new_ap=A[:, 1], lgb_ap=B[:, 1], new_auc=A[:, 0], lgb_auc=B[:, 0])).to_csv(OUT + "g144_r1.csv", index=False); print("AP by month", {mo: (round(float(np.nanmean(A[[r["month"] == mo for r in te], 1])), 4), round(float(np.nanmean(B[[r["month"] == mo for r in te], 1])), 4)) for mo in sorted({r["month"] for r in te})}, flush=True)
nets = [fit(rows, s) for s in range(SEEDS)]; pb = lgb.Booster(model_file=W + "physon_base.txt"); META = pickle.load(open(W + "m3v3_meta.pkl", "rb"))["meta"]
X = np.load(f"{NN}/nn_test_m3v3/X.npy", mmap_mode="r"); off = np.load(f"{NN}/nn_test_m3v3/off.npy"); Wd = [str(w) for w in np.load(f"{NN}/nn_test_m3v3/window_id.npy", allow_pickle=True)]; pos = {w: i for i, w in enumerate(Wd)}
J = pd.read_csv(W + "g117_onset_oracle.csv").set_index("w"); NOPRE = set(J[(J.v_ff >= 0.98) & J.flow_ratio.between(0.85, 1.15) & (J.fmax <= 0.1)].index); res = []
PRO = {f"q24f_{c}_s{s}": pickle.load(open(W + f"probs_m3v3__q24f_{c}_s{s}.pkl", "rb")) for s in range(8) for c in ("attnglob", "attnwp05", "attn1", "drop1", "r13")} if MODE == "run" else {}
for mt in META:
    if mt["cond"] != "queue_onset" or (mt["day"] + 1) % 3 == 0:
        continue
    w = mt["window_id"]; i = pos[w]; Xw = np.asarray(X[off[i]:off[i + 1]]); x = inputs(Xw); y = mt["y"].T.astype(bool)[5]; e = mt["m"].T.astype(bool)[5]; pl = pb.predict(x[:, :PF]); r = dict(w=w, cl=f"{mt['panel']}_{mt['day']}", pre=w not in NOPRE, new=auc(score(nets, x), y, e), lgb=auc(pl, y, e))
    if PRO:
        pn = sum(wt * PRO[f"q24f_{c}_s{s}"][w][5] for s in range(8) for c, wt in (("attnglob", 2), ("attnwp05", 2), ("attn1", 2), ("drop1", 1), ("r13", 1))) / 64; r["final"] = auc(0.7 * pn + 0.3 * pl, y, e)
    res.append(r)
D = pd.DataFrame(res); pickle.dump(res, open(OUT + "g144_r2.pkl", "wb")); line("R2 M3 dev onset", list(D.new), list(D.lgb), D.cl.to_numpy()); P = D[D.pre]; line("R2 precursor windows", list(P.new), list(P.lgb), P.cl.to_numpy())
if "final" in D:
    line("R2 against the final mix a = 0.3 (record)", list(D.new), list(D.final), D.cl.to_numpy())
print(f"total {time.time() - T00:.0f}s", flush=True)
