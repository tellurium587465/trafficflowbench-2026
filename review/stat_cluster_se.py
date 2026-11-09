import sys, os, glob, pickle, numpy as np, pandas as pd
os.chdir(r"C:/Users/user/tfb/src")
wi = pd.concat([pd.read_csv(f) for f in glob.glob(r"C:/Users/user/tfb/data/kaggle_public/task2/*/*/window_index.csv")])
cond = dict(zip(wi.window_id, wi.condition)); pan = dict(zip(wi.window_id, wi.panel))
W = r"C:/Users/user/tfb/work/t2/"
M = {"v8": "probs_v8.pkl", "v9npp": "probs_v9npp.pkl", "s17": "probs_s17_t1f3ens3c_t2v12nppdem_nnDnp6_t4l2.pkl", "s23": "probs_s23_t1GapML_t2v14af_nnAf6_t4l2.pkl"}
PR = {m: pickle.load(open(W + f, "rb")) for m, f in M.items()}
def bs(o):
    pe = np.asarray(o).ravel(); order = np.argsort(-pe); cs = np.cumsum(pe[order]); kk = np.arange(1, pe.size + 1)
    kb = int(np.argmax(cs / (kk + pe.sum() - cs))) + 1; s = np.zeros(pe.size, bool); s[order[:kb]] = True; return s
print("S_queue null SE (x0.68 calibration): independent windows vs panel-clustered signs")
for a, c in (("v8", "v9npp"), ("v9npp", "s17"), ("s17", "s23")):
    for sp in ("validation", "private"):
        ws = [w for w in PR[a] if f"_{sp}_" in w]
        d = {w: 1 - (bs(PR[a][w]) & bs(PR[c][w])).sum() / max((bs(PR[a][w]) | bs(PR[c][w])).sum(), 1) for w in ws}
        # S_queue = mean over 80 windows (equal weights); onset/ongoing each 40
        ind = np.sqrt(sum(v**2 for v in d.values())) / 80
        byp = {}
        for w, v in d.items(): byp.setdefault(pan[w], []).append(v)
        clu = np.sqrt(sum(sum(v)**2 for v in byp.values())) / 80
        print(f"{a:5s}->{c:5s} {sp:10s} max|dS| {np.mean(list(d.values())):.3f}  SE_ind {0.68*ind:.4f}  SE_panel-clustered(upper) {0.68*clu:.4f}  (total x0.3: {0.3*0.68*ind:.4f} / {0.3*0.68*clu:.4f})")
