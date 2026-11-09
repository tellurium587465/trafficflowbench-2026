"""G189 (11/01, ledger #152): B26 closed the month fine-tuning of f3big on February, a month inside the training distribution. Here the instrument of #132, on a redrawn month: the release's visible cells of March or April are hidden with the month's own mask shifted by 1, 2, 3 days, features rebuilt on that view (ADAPT on, as production); new trees on the odd days (lr 0.05, 100 rounds, 63 leaves, continued from f3big), read on the even days, on the same mask shift 1.
The S formula pooled over the panels, base against fine-tuned. Labels are released observations only.
usage: py -B -X utf8 g189_lgb_month_ft.py <validation|private>"""
import os, sys, copy, time, numpy as np, lightgbm as lgb, warnings
os.environ.setdefault("TFB_T1_ADAPT", "1"); os.environ.setdefault("TFB_T1_FEAT_TRAINSTAT", "1"); warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src")
from cachelib import load
from common import PANELS, WORK
from t1_features3 import grid_features3, rows3, base_of3

SP = sys.argv[1]; MD = WORK + "/t1/"; mv = lgb.Booster(model_file=MD + "speed_f3big.txt"); mq = lgb.Booster(model_file=MD + "flow_f3big.txt"); rng = np.random.default_rng(152); T0 = time.time(); NT = int(os.environ.get("G189_T", "8"))
PRM = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, num_threads=NT, verbose=-1, seed=26, force_row_wise=True, deterministic=True); XT = {"v": [], "q": []}; YT = {"v": [], "q": []}; EV = []; NROW = 120000


def view(P, s_):
    h = np.roll(P.target, s_ * 288, axis=0); fin = np.isfinite(P.V) & np.isfinite(P.Q); hid = h & fin & P.E & ~P.target; Pa = copy.copy(P); Pa.V = np.where(hid, np.nan, P.V); Pa.Q = np.where(hid, np.nan, P.Q); Pa.target = P.target | hid; return Pa, hid


for p in PANELS:
    P = load(p, SP); pi = PANELS.index(p); day = np.arange(P.T) // 288
    for s_ in (1, 2, 3):
        Pa, hid = view(P, s_); F, _, _, _ = grid_features3(Pa); bv, bq = base_of3(F); odd = (day % 2 == 0)[:, None]; r, c = np.nonzero(hid & odd); k = rng.permutation(len(r))[:NROW]; r_, c_ = r[k], c[k]; X, _ = rows3(Pa, F, r_, c_); X = np.column_stack([np.full(len(r_), pi, np.float32), X]).astype(np.float32); XT["v"].append(X); YT["v"].append((P.V[r_, c_] / P.vf[c_] - bv[r_, c_]).astype(np.float32)); YT["q"].append((P.Q[r_, c_] / P.lanes[c_] - bq[r_, c_]).astype(np.float32))
        if s_ == 1:
            r, c = np.nonzero(hid & ~odd); k = rng.permutation(len(r))[:NROW]; r_, c_ = r[k], c[k]; X, _ = rows3(Pa, F, r_, c_); X = np.column_stack([np.full(len(r_), pi, np.float32), X]).astype(np.float32); EV.append(dict(X=X, bv=bv[r_, c_], bq=bq[r_, c_], vf=P.vf[c_], ln=P.lanes[c_], V=P.V[r_, c_], Q=P.Q[r_, c_]))
        del F
    print(p, f"{time.time() - T0:.0f}s", flush=True); load.cache_clear()
X = np.vstack(XT["v"]); print("rows", X.shape, flush=True); out = {}
for nm, y, base in (("speed", np.concatenate(YT["v"]), mv), ("flow", np.concatenate(YT["q"]), mq)):
    out[nm] = lgb.train(PRM, lgb.Dataset(X, y, free_raw_data=False), 100, init_model=base); print(nm, f"{time.time() - T0:.0f}s", flush=True)
E = {k: np.concatenate([d[k] for d in EV]) for k in EV[0]}; Xe = np.vstack([d["X"] for d in EV])


def S(v, q):
    rv = float(np.sqrt(np.mean((np.clip(v, 3, None) - E["V"]) ** 2))); rq = float(np.sqrt(np.mean((np.clip(q, 60 / E["ln"], None) - E["Q"] / E["ln"]) ** 2))); return rv, rq, 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)


b = S((E["bv"] + mv.predict(Xe, num_threads=NT)) * E["vf"], E["bq"] + mq.predict(Xe, num_threads=NT)); f = S((E["bv"] + out["speed"].predict(Xe, num_threads=NT)) * E["vf"], E["bq"] + out["flow"].predict(Xe, num_threads=NT)); print(f"{SP} even days: base rmse v {b[0]:.3f} q {b[1]:.2f} S {b[2]:.4f} | fine-tuned rmse v {f[0]:.3f} q {f[1]:.2f} S {f[2]:.4f} | {f[2] - b[2]:+.4f} | {(time.time() - T0) / 60:.0f} min", flush=True)
out["speed"].save_model(MD + f"speed_f3big_ft_{SP}_odd.txt"); out["flow"].save_model(MD + f"flow_f3big_ft_{SP}_odd.txt")
