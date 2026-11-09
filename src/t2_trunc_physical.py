"""E6 (10/22; 10/23 extended to the final candidates): the organisers' Task 2 check with the release files PHYSICALLY cut at
T. Per window, a copy of the release is written with every row after T deleted (mainline_states_masked, ramp_states, the
Task 1 template; files left empty are removed; window_index.csv keeps the windows whose origin is <= T) for the window's
panel (phase 1) or every panel of its split (phases 2, 3), and t2_window_entry.py is run on it with the candidate's
composition (t2_final_spec.FINAL):
  phase 1 (every chosen window, "saved" probabilities):
    a. the window's features and, when TTA is on, its 8 draws rebuilt from the cut copy == the stored nn_test_afdem /
       nn_test_afdem_tta<k> rows (bit for bit)
    b. the window's queue == the candidate's built queue (bit for bit): the composition and the post-processing (cone,
       evidence rule, far gate) read nothing after T
  phase 2 (each split x day D >= 2 with X1 windows, cut at that day's earliest X1 origin): X1's training windows of days
    1 .. D-1 rebuilt from the cut copy (twin/build_march.py) == the stored nn_data_<month> rows (meta, X, Y, EL; bit for
    bit), so the fine-tunes (seed 0, CPU) see the same data; for the q24t family also the X with the row at T filled
    from the cut copy (t2_trow.py) == the stored nn_data_marcht / nn_data_aprilt rows
  phase 3 (--cpu <n> windows, "cpu" probabilities, X1 stored): every part recomputed from the cut copy: the no-memory
    LightGBM, the physics onset and its a == stored (bit for bit); the networks: max |p_cpu - p_stored| and the queue
    cells that differ from the built queue (the reproduction path against the submitted numbers)
  phase 4 (--only 4 with window ids): X1 fine-tuned in place from the cut copy (TFB_ENTRY_X1=run) == the stored X1
usage: python t2_trunc_physical.py <candidate s70|A|B> <reference queue parquet> [--cpu <n>] [--only <phase list>] [window_id ...]
       (no window ids: all 160 windows in phase 1; phase 3 picks <n> windows, seed 0, every panel, both splits and
       conditions, day >= 3)"""
import os, sys, glob, json, time, shutil, pickle, subprocess, numpy as np, pandas as pd
from common import REL, NN, ROOT as RROOT, WORK
from cachelib import T2_PANELS
import t2_final_spec as FS

SRC = os.path.dirname(os.path.abspath(__file__)); W = WORK + "/"; ROOT = os.environ.get("TFB_E6_ROOT", "C:/tfb_data/e6_trunc")
args = sys.argv[1:]; CAND, REF = args[0], args[1]; args = args[2:]
NCPU = int(args[args.index("--cpu") + 1]) if "--cpu" in args else 0
ONLY = set(args[args.index("--only") + 1].split(",")) if "--only" in args else {"1", "2", "3"}
ids = [a for a in args if a.startswith("Q2_")]
ENV, CHOICE = FS.FINAL[CAND]; ch = CHOICE.split(","); TAG_ = os.environ.get("TFB_E6_TAG", "")   # 11/03: a distinct entry name per run, so two candidates can run at once
WI = pd.concat([pd.read_csv(REL / "task2" / p / sp / "window_index.csv") for p in T2_PANELS for sp in ("validation", "private")], ignore_index=True)
WI["day"] = WI.date.astype(str).str[8:10].astype(int)
REFQ = pd.read_parquet(REF); REFG = dict(tuple(REFQ.groupby("window_id", sort=False)))
SX = np.load(f"{NN}/nn_test_afdem/X.npy", mmap_mode="r"); SO = np.load(f"{NN}/nn_test_afdem/off.npy")
SW = [str(w) for w in np.load(f"{NN}/nn_test_afdem/window_id.npy", allow_pickle=True)]
os.environ.update(ENV); CFG = FS.config()


def x1_window(r):
    return (ch[0] if r.condition == "queue_onset" else ch[2]) in ("x1", "half")


def cut_copy(r, dst, panels):
    T = r.forecast_origin; split = r.split; n_rows = n_del = 0
    shutil.rmtree(dst, ignore_errors=True); shutil.copytree(REL / "config", f"{dst}/config")     # static: corridor list, contract
    for panel in panels:
        shutil.copytree(REL / "corridors" / panel / "network", f"{dst}/corridors/{panel}/network")
        shutil.copytree(REL / "corridors" / panel / "train", f"{dst}/corridors/{panel}/train", copy_function=os.link)   # before T: whole
        for sub in ("mainline_states_masked", "ramp_states"):
            for f in sorted(glob.glob(str(REL / "corridors" / panel / split / sub / "**" / "*.parquet"), recursive=True)):
                d = pd.read_parquet(f); ts = d.timestamp.astype(str)              # released as "YYYY-MM-DDTHH:MM:SSZ" strings
                keep = (ts <= T).to_numpy(); n_rows += len(d); n_del += int((~keep).sum())
                if keep.any():
                    o = f"{dst}/" + os.path.relpath(f, REL).replace("\\", "/"); os.makedirs(os.path.dirname(o), exist_ok=True)
                    d[keep].to_parquet(o, index=False)
        t1 = pd.read_csv(REL / "task1" / panel / split / "sample_submission_state.csv", dtype=str)
        os.makedirs(f"{dst}/task1/{panel}/{split}"); t1[t1.timestamp <= T].to_csv(f"{dst}/task1/{panel}/{split}/sample_submission_state.csv", index=False)
        wi = pd.read_csv(REL / "task2" / panel / split / "window_index.csv")
        os.makedirs(f"{dst}/task2/{panel}/{split}"); wi[wi.forecast_origin <= T].to_csv(f"{dst}/task2/{panel}/{split}/window_index.csv", index=False)
    return n_rows, n_del


def entry(w, out, rel, cache, mode):
    for d in glob.glob(f"{NN}/nn_test_entry_{out}") + glob.glob(f"{NN}/nn_test_entry_{out}_tta*"):
        shutil.rmtree(d)
    os.makedirs(cache, exist_ok=True)
    p = subprocess.run([sys.executable, "t2_window_entry.py", w, out, mode, CHOICE], cwd=SRC, capture_output=True, text=True,
                       env=dict(os.environ, TFB_REL=rel, TFB_CACHE=cache))
    assert p.returncode == 0, p.stderr[-3000:]
    return pd.read_parquet(W + f"t2/queue_entry_{out}.parquet")


def same(a, b):
    k = ["timestamp", "link_id"]; a = a.astype({"timestamp": str}).sort_values(k).reset_index(drop=True)
    b = b.astype({"timestamp": str}).sort_values(k).reset_index(drop=True)
    return len(a) == len(b) and (a[k] == b[k]).all().all() and (a.queue_pred.to_numpy() == b.queue_pred.to_numpy()).all(), \
        int((a.queue_pred.to_numpy() != b.queue_pred.to_numpy()).sum()) if len(a) == len(b) else -1


def features_same(w, out):
    i = SW.index(w); ok = bool(np.array_equal(np.load(f"{NN}/nn_test_entry_{out}/X.npy"), np.asarray(SX[SO[i]:SO[i + 1]]), equal_nan=True)); nk = 0
    if os.path.isdir(f"{NN}/nn_test_afdemt"):                   # 11/02 ledger #164: the row at T from the cut copy == the stored inputs of q24t
        ok &= bool(np.array_equal(np.load(f"{NN}/nn_test_entry_{out}t/X.npy"), np.load(f"{NN}/nn_test_afdemt/X.npy", mmap_mode="r")[SO[i]:SO[i + 1]], equal_nan=True))
    for k in range(1, 9):
        d = f"{NN}/nn_test_entry_{out}_tta{k}"
        if os.path.isdir(d):
            ok &= bool(np.array_equal(np.load(f"{d}/X.npy"), np.load(f"{NN}/nn_test_afdem_tta{k}/X.npy", mmap_mode="r")[SO[i]:SO[i + 1]], equal_nan=True)); nk += 1
    return ok, nk


ok = True; t0 = time.time(); log = []
sel = WI[WI.window_id.isin(ids)] if ids else WI
print(f"candidate {CAND} ({ENV}, choice {CHOICE}), reference {REF}, windows {len(sel)}", flush=True)

if "1" in ONLY:
    n1 = 0
    for r in sel.itertuples():
        base = f"{ROOT}/{r.window_id}"; n_rows, n_del = cut_copy(r, f"{base}/rel", [r.panel])
        q = entry(r.window_id, "trunc" + TAG_, f"{base}/rel", f"{base}/cache", "saved")
        a, nk = features_same(r.window_id, "trunc" + TAG_); b, nd = same(q, REFG[r.window_id][["timestamp", "link_id", "queue_pred"]])
        ok &= a and b; n1 += a and b
        log.append(dict(phase=1, window=r.window_id, cond=r.condition, day=r.day, rows_deleted=n_del, features=a, tta_draws=nk, queue=b, cells_diff=nd))
        print(f"[1] {r.window_id} ({r.condition}, day {r.day}, {time.time() - t0:.0f}s): rows deleted {n_del} of {n_rows} | a features "
              f"(+{nk} TTA draws) == stored: {a} | b queue == {os.path.basename(REF)}: {b} ({nd} cells differ)", flush=True)
        if a and b:
            shutil.rmtree(base, ignore_errors=True)
    print(f"phase 1: {n1} / {len(sel)} windows pass", flush=True)

if "2" in ONLY:
    SD = CFG.get("x1arm", "") in ("sd", "sdd3")                  # 11/01 ledger #125: one cut per (split, day, hour), day 1 included: the same day's windows ending before the hour
    D3 = CFG.get("x1arm", "") == "sdd3"                          # 11/03 #189 (A13): April's pool has origins every 3 rows (nn_data_april_sp3)
    xw = sel[np.array([x1_window(r) for r in sel.itertuples()], bool) & (sel.day >= (1 if SD else 2)).to_numpy() & ((sel.condition == "queue_ongoing").to_numpy() if SD else True)].copy(); xw["hr"] = xw.forecast_origin.str[11:13] if SD else ""
    for (sp, D, hr), g in sorted(xw.groupby(["split", "day", "hr"])):
        r = g.sort_values("forecast_origin").iloc[0]; base = f"{ROOT}/month_{sp}_{D}{hr}"; md = f"nn_data_trunc_{sp}"; hcut = (D - 1) * 288 + int(r.forecast_origin[11:13]) * 12 if SD else None
        cut_copy(r, f"{base}/rel", T2_PANELS); env = dict(os.environ, TFB_REL=f"{base}/rel", TFB_CACHE=f"{base}/cache"); os.makedirs(f"{base}/cache", exist_ok=True)
        code = ("import os, build_cache, cachelib\nfrom cachelib import T2_PANELS\nfor p in T2_PANELS:\n"
                "    tr = cachelib.CACHE / f'{p}_train.npz'\n    os.path.exists(tr) or os.link(f'" + WORK + "/cache/{p}_train.npz', tr)\n"
                f"    build_cache.build(p, '{sp}')\n")
        subprocess.run([sys.executable, "-c", code], cwd=SRC, check=True, env=env, capture_output=True)
        subprocess.run([sys.executable, str(RROOT / "twin" / "build_march.py"), "3" if D3 and sp == "private" else "6", sp], cwd=SRC, check=True, capture_output=True, env=dict(env, TFB_MONTH_OUT=md))
        QT = ENV.get("TFB_T2_OG_FAMILY", "").startswith("q24t")      # 11/02: the row at T of X1's q24t training windows, filled from the cut copy
        if QT:
            subprocess.run([sys.executable, "t2_trow.py", sp, md, md + "t"], cwd=SRC, check=True, capture_output=True, env=env)
        res = {}
        for name, d in (("cut", md), ("stored", {"validation": "nn_data_march", "private": "nn_data_april_sp3" if D3 else "nn_data_april"}[sp])):
            M = {f: np.load(f"{NN}/{d}/{f}.npy", allow_pickle=True) for f in ("panel", "t", "cond", "date", "off")}
            dd_ = np.array([int(str(x)[8:10]) for x in M["date"]]); ks = np.flatnonzero((dd_ < D) | ((dd_ == D) & (M["t"].astype(int) + 6 <= hcut)) if SD else dd_ < D); off = M["off"]
            rows = np.concatenate([np.arange(off[k], off[k + 1]) for k in ks]) if len(ks) else np.zeros(0, int)
            X = np.load(f"{NN}/{d}/X.npy", mmap_mode="r"); Y = np.load(f"{NN}/{d}/Y.npy", mmap_mode="r"); EL = np.load(f"{NN}/{d}/EL.npy", mmap_mode="r")
            res[name] = dict(meta=[M[f][ks].astype(str) for f in ("panel", "t", "cond", "date")], X=np.asarray(X[rows]), Y=np.asarray(Y[rows]), EL=np.asarray(EL[rows]), n=len(ks))
            if QT:
                res[name]["XT"] = np.asarray(np.load(f"{NN}/{d}t/X.npy", mmap_mode="r")[rows])
        c, s = res["cut"], res["stored"]
        m_ok = all(np.array_equal(x, y) for x, y in zip(c["meta"], s["meta"]))
        a2 = m_ok and all(np.array_equal(c[k], s[k], equal_nan=True) for k in ("X", "Y", "EL") + (("XT",) if QT else ()))
        ok &= a2; log.append(dict(phase=2, split=sp, day=D, cut_at=r.forecast_origin, windows=c["n"], stored=s["n"], same=a2))
        print(f"[2] {sp} day {D} (cut at {r.forecast_origin}, {time.time() - t0:.0f}s): X1 training windows of days < {D}: cut copy {c['n']}, "
              f"stored {s['n']} -> meta equal {m_ok}, X / Y / EL{' / X with the row at T (nn_data_marcht, aprilt)' if QT else ''} equal {a2}", flush=True)
        if a2:
            shutil.rmtree(base, ignore_errors=True)
        shutil.rmtree(f"{NN}/{md}", ignore_errors=True); shutil.rmtree(f"{NN}/{md}t", ignore_errors=True)

if "3" in ONLY and NCPU:
    rng = np.random.default_rng(0); el = WI[WI.day >= 3]; pick = []
    for i in range(NCPU):
        p = (T2_PANELS + T2_PANELS)[i]
        g = el[(el.panel == p) & (el.split == ("validation", "private")[i % 2]) & (el.condition == ("queue_onset", "queue_ongoing")[(i // 2) % 2])]
        g = g[~g.window_id.isin([x.window_id for x in pick])]
        pick.append(g.iloc[rng.integers(len(g))])
    for r in pick:
        base = f"{ROOT}/cpu_{r.window_id}"; cut_copy(r, f"{base}/rel", T2_PANELS)
        q = entry(r.window_id, "trunccpu" + TAG_, f"{base}/rel", f"{base}/cache", "cpu")
        a, nk = features_same(r.window_id, "trunccpu" + TAG_)
        P = pickle.load(open(W + f"t2/entry_trunccpu{TAG_}_parts.pkl", "rb")); exact, dnn = {}, 0.0
        for (pre, tag), v in P.items():
            if pre == "e3stats":
                import t2_e3_official as E3
                sv = E3._stats().get(r.window_id)
            else:
                sv = pickle.load(open(W + ("t2/physon_a.pkl" if pre == "physon_a" else f"t2/probs_{pre}{tag}.pkl"), "rb"))[r.window_id]
            if pre.startswith("x1") and r.day >= 2:                  # X1 stored (its inputs: phase 2)
                continue
            if pre == "e3stats":                                # sp is not recomputed (beta = 0): kq, ko
                exact[pre] = (v is None and sv is None) or (v is not None and sv is not None and all(np.array_equal(x, y) for x, y in zip(v[:2], sv[:2])))
            elif pre in ("lgbnm", "physon_a", CFG["phys"]):
                exact[pre] = bool(np.array_equal(np.asarray(v), np.asarray(sv)))
            else:
                dnn = max(dnn, float(np.abs(np.asarray(v, float) - np.asarray(sv, float)).max()))
        b, nd = same(q, REFG[r.window_id][["timestamp", "link_id", "queue_pred"]]); ok &= a and all(exact.values())
        log.append(dict(phase=3, window=r.window_id, cond=r.condition, day=r.day, features=a, exact=exact, nn_maxdiff=dnn, cells_diff=nd, cells=len(q)))
        print(f"[3] {r.window_id} ({r.condition}, day {r.day}, {time.time() - t0:.0f}s): features (+{nk} TTA) == stored {a} | recomputed == stored "
              f"{exact} | networks cpu vs stored max |dp| {dnn:.2e} | queue cells differing from {os.path.basename(REF)}: {nd} of {len(q)}", flush=True)
        if a and all(exact.values()):
            shutil.rmtree(base, ignore_errors=True)

if "4" in ONLY:                                                 # 10/23: X1 fine-tuned in place (TFB_ENTRY_X1=run) vs stored
    for w in ids:
        r = WI[WI.window_id == w].iloc[0]; base = f"{ROOT}/x1run_{w}"; cut_copy(r, f"{base}/rel", T2_PANELS); os.environ["TFB_ENTRY_X1"] = "run"
        q = entry(w, "truncx1" + TAG_, f"{base}/rel", f"{base}/cache", "cpu"); os.environ.pop("TFB_ENTRY_X1")
        P = pickle.load(open(W + f"t2/entry_truncx1{TAG_}_parts.pkl", "rb")); dx, same_ = 0.0, True
        for (pre, tag), v in P.items():
            if pre.startswith("x1"):
                sv = pickle.load(open(W + f"t2/probs_{pre}{tag}.pkl", "rb"))[w]; dx = max(dx, float(np.abs(v - sv).max())); same_ &= bool(np.array_equal(v, sv))
        b, nd = same(q, REFG[w][["timestamp", "link_id", "queue_pred"]]); ok &= same_
        log.append(dict(phase=4, window=w, cond=r.condition, day=r.day, x1_bit_identical=same_, x1_maxdiff=dx, cells_diff=nd))
        print(f"[4] {w} ({r.condition}, day {r.day}, {time.time() - t0:.0f}s): X1 fine-tuned here == stored {same_} (max |dp| {dx:.2e}) | queue cells "
              f"differing from {os.path.basename(REF)}: {nd}", flush=True)
        if same_:
            shutil.rmtree(base, ignore_errors=True)

pd.DataFrame(log).to_csv(W + f"t2/trunc_physical_{CAND}.csv", index=False)
print(f"E6 physical truncation ({CAND}):", "PASS" if ok else "FAIL", f"{time.time() - t0:.0f}s")
