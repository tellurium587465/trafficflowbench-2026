"""G140 (10/31, ledger #101 check): can X1 fine-tune on same-day windows? For an official window at T, cut the release of every panel of its split at T (as E6),
rebuild the training pool (twin/build_march.py) and compare with the stored pool: every window with T' + 6 <= T of the month must exist in the cut copy with the same meta, X, Y and EL,
and the cut copy must hold no other such window (windows past T - 6 appear in a cut copy with an incomplete future; the same-day pool leaves them out by its rule). Passing means the same-day pool of the entry at T equals the stored rows, so the fine-tune sees the same data.
usage: py g140_sameday_cut.py <window_id> [<window_id> ...]"""
import os, sys, glob, time, shutil, subprocess
R = r"C:/Users/user/tfb/"; sys.path.insert(0, R + "src")
import numpy as np, pandas as pd
from common import REL, NN, WORK, ROOT as RROOT
from cachelib import T2_PANELS
OUT = "C:/tfb_data/e6_sameday"; SRC = R + "src"
WI = pd.concat([pd.read_csv(REL / "task2" / p / sp / "window_index.csv") for p in T2_PANELS for sp in ("validation", "private")], ignore_index=True)


def cut_copy(T, split, dst):
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(REL / "config", f"{dst}/config")
    for panel in T2_PANELS:
        shutil.copytree(REL / "corridors" / panel / "network", f"{dst}/corridors/{panel}/network")
        shutil.copytree(REL / "corridors" / panel / "train", f"{dst}/corridors/{panel}/train", copy_function=os.link)
        for sub in ("mainline_states_masked", "ramp_states"):
            for f in sorted(glob.glob(str(REL / "corridors" / panel / split / sub / "**" / "*.parquet"), recursive=True)):
                d = pd.read_parquet(f); keep = (d.timestamp.astype(str) <= T).to_numpy()
                if keep.any():
                    o = f"{dst}/" + os.path.relpath(f, REL).replace("\\", "/"); os.makedirs(os.path.dirname(o), exist_ok=True); d[keep].to_parquet(o, index=False)
        t1 = pd.read_csv(REL / "task1" / panel / split / "sample_submission_state.csv", dtype=str); os.makedirs(f"{dst}/task1/{panel}/{split}")
        t1[t1.timestamp <= T].to_csv(f"{dst}/task1/{panel}/{split}/sample_submission_state.csv", index=False)
        wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv"); os.makedirs(f"{dst}/task2/{panel}/{split}"); wi[wi.forecast_origin <= T].to_csv(f"{dst}/task2/{panel}/{split}/window_index.csv", index=False)


def pool(d):
    M = {f: np.load(f"{NN}/{d}/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "date", "off")}
    return M, {(str(p), int(t), str(c)): i for i, (p, t, c) in enumerate(zip(M["panel"], M["t"], M["cond"]))}


bad = 0
for w in sys.argv[1:]:
    r = WI[WI.window_id == w].iloc[0]; sp = r.split; T = r.forecast_origin; base = f"{OUT}/{w}"; md = "nn_data_sameday_cut"; t0 = time.time()
    cut_copy(T, sp, f"{base}/rel"); env = dict(os.environ, TFB_REL=f"{base}/rel", TFB_CACHE=f"{base}/cache"); os.makedirs(f"{base}/cache", exist_ok=True)
    code = ("import os, build_cache, cachelib\nfrom cachelib import T2_PANELS\nfor p in T2_PANELS:\n    tr = cachelib.CACHE / f'{p}_train.npz'\n    os.path.exists(tr) or os.link(f'" + WORK + "/cache/{p}_train.npz', tr)\n"
            f"    build_cache.build(p, '{sp}')\n")
    subprocess.run([sys.executable, "-c", code], cwd=SRC, check=True, env=env, capture_output=True)
    subprocess.run([sys.executable, str(RROOT / "twin" / "build_march.py"), "6", sp], cwd=SRC, check=True, capture_output=True, env=dict(env, TFB_MONTH_OUT=md))
    stored = {"validation": "nn_data_march", "private": "nn_data_april"}[sp]; C, ci = pool(md); S, si = pool(stored)
    Tt = pd.Timestamp(T.replace("Z", "")); month0 = Tt.normalize().replace(day=1); Tr = int((Tt - month0).total_seconds() // 300)
    want = {k for k in si if k[1] + 6 <= Tr}; extra = sorted({k for k in ci if k[1] + 6 <= Tr} - want)[:3]; miss = sorted(want - set(ci))[:3]; same = 0; diff = []
    Xc, Yc, Ec = (np.load(f"{NN}/{md}/{f}.npy") for f in ("X", "Y", "EL")); Xs, Ys, Es = (np.load(f"{NN}/{stored}/{f}.npy", mmap_mode="r") for f in ("X", "Y", "EL"))
    for k in sorted(want & set(ci)):
        i, j = ci[k], si[k]; a, b = int(C["off"][i]), int(C["off"][i + 1]); c, d = int(S["off"][j]), int(S["off"][j + 1])
        ok = b - a == d - c and np.array_equal(np.asarray(Xc[a:b]), np.asarray(Xs[c:d]), equal_nan=True) and np.array_equal(np.asarray(Yc[a:b]), np.asarray(Ys[c:d])) and np.array_equal(np.asarray(Ec[a:b]), np.asarray(Es[c:d]))
        same += ok; diff += [] if ok else [k]
    dayk = {k for k in want if k[1] // 288 == Tr // 288}; print(f"{w} ({r.condition}, {sp}, T {T}, row {Tr}, {time.time() - t0:.0f}s): pool windows wanted {len(want)} (same day {len(dayk)}) cut copy {len(ci)} identical {same} | missing {miss} extra {extra} differ {diff[:3]} ({len(diff)})", flush=True)
    good = not miss and not extra and not diff; bad += not good
    if good:
        shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(f"{NN}/{md}", ignore_errors=True)
print("G140:", "PASS" if not bad else f"FAIL on {bad} windows")
