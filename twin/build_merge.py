"""Merge NN training sets (10/22, E4): real train months (nn_data_afdem) + March (nn_data_march) + optionally a clean
synthetic family's windows (synthetic-year dates, optionally one condition only). Every row keeps its source; the
synthetic windows get their template month as fold_month (as build_simR SIM_FOLD_TEMPLATE=1), March windows 2031-03.
usage: python build_merge.py <dst> <part> [<part> ...]
       part = <nn_data dir name>                     (all windows)
            | <nn_data dir name>@<YYYY-MM-DD           (windows dated before that day, e.g. nn_data_march@<2031-03-16)
            | <nn_data dir name>:<cond>:<family>     (synthetic-year windows of that condition; family for fold_month)
            | <nn_data dir name>:<cond>:<family>:<YYYY-MM>+<YYYY-MM>...  (only those synthetic calendar months = worlds)
example: python build_merge.py nn_data_afdemM nn_data_afdem nn_data_march
         python build_merge.py nn_data_afdemMsimC24 nn_data_afdem nn_data_march nn_data_twinC24:queue_ongoing:C"""
import os, sys, glob, json, numpy as np
sys.path.insert(0, r"C:/Users/user/tfb/src")
from common import NN

dst, parts = sys.argv[1], sys.argv[2:]
D = f"{NN}/{dst}"; os.makedirs(D, exist_ok=True)
srcs = []
for part in parts:
    name, *opt = part.split(":")
    name, _, before = name.partition("@<")
    S = f"{NN}/{name}"; date = np.load(f"{S}/date.npy", allow_pickle=True); cond = np.load(f"{S}/cond.npy", allow_pickle=True)
    keep = np.ones(len(date), bool) if not before else np.array([str(d)[:10] < before for d in date])
    if opt:
        keep &= np.array([str(d)[:4] >= "2040" for d in date]) & (cond == opt[0])
        if len(opt) > 2:
            keep &= np.isin([str(d)[:7] for d in date], opt[2].split("+"))
    fam = opt[1] if len(opt) > 1 else None
    srcs.append((name, S, np.flatnonzero(keep), fam))
names = [str(x) for x in np.load(f"{NN}/{parts[0].split(':')[0]}/names.npy", allow_pickle=True)]
for name, S, ks, fam in srcs:
    assert [str(x) for x in np.load(f"{S}/names.npy", allow_pickle=True)] == names, ("column names differ", name)
tot, offs = 0, [np.array([0])]
for name, S, ks, fam in srcs:
    o = np.load(f"{S}/off.npy"); lens = o[ks + 1] - o[ks]; offs.append(tot + np.cumsum(lens)); tot += int(lens.sum())
X = np.lib.format.open_memmap(f"{D}/X.npy", mode="w+", dtype=np.float32, shape=(tot, len(names))); pos = 0
cat = {f: [] for f in ("Y", "EL", "cond", "date", "panel", "t", "is_eval", "is_sel", "fold_month")}
for name, S, ks, fam in srcs:
    XS = np.load(f"{S}/X.npy", mmap_mode="r"); o = np.load(f"{S}/off.npy")
    rows = np.concatenate([np.arange(o[k], o[k + 1]) for k in ks]) if len(ks) else np.zeros(0, int)
    for a in range(0, len(rows), 1 << 21):
        r = rows[a:a + (1 << 21)]; X[pos:pos + len(r)] = XS[r]; pos += len(r)
    for f in ("Y", "EL"):
        cat[f].append(np.asarray(np.load(f"{S}/{f}.npy", mmap_mode="r")[rows]))
    for f in ("cond", "date", "panel", "t"):
        cat[f].append(np.load(f"{S}/{f}.npy", allow_pickle=True)[ks])
    for f in ("is_eval", "is_sel"):
        v = np.load(f"{S}/{f}.npy", allow_pickle=True)[ks] if os.path.exists(f"{S}/{f}.npy") else np.zeros(len(ks), bool)
        cat[f].append(v.astype(bool) & (fam is None))                # synthetic windows are never evaluation windows
    dates = np.load(f"{S}/date.npy", allow_pickle=True)[ks]
    if fam:
        tmpl = {}
        for jf in glob.glob(f"C:/Users/user/tfb/work/twin/*_twin{fam}[0-9]*.json"):
            j = json.load(open(jf)); pn = os.path.basename(jf).split("_twin")[0]
            if "calendar" in j and "template" in j:
                assert tmpl.setdefault((pn, j["calendar"]), j["template"]) == j["template"], ("calendar clash", pn, j["calendar"])
        pans = np.load(f"{S}/panel.npy", allow_pickle=True)[ks]
        cat["fold_month"].append(np.array([tmpl[(str(p), str(d)[:7])] for p, d in zip(pans, dates)]))
    else:
        cat["fold_month"].append(np.array([str(d)[:7] for d in dates]))
    print(name, "windows", len(ks), "rows", len(rows), flush=True)
X.flush(); del X
np.save(f"{D}/off.npy", np.concatenate(offs)); np.save(f"{D}/names.npy", np.array(names))
for f, v in cat.items():
    np.save(f"{D}/{f}.npy", np.concatenate(v))
o = np.load(f"{D}/off.npy"); assert o[-1] == tot and len(o) - 1 == len(np.load(f"{D}/cond.npy", allow_pickle=True)), "offsets"
print("done", dst, "windows", len(o) - 1, "rows", tot)
