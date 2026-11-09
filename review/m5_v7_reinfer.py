"""M5 audit V7: re-run FINAL inference on CPU (fp32) from the saved .pt files and the real t2_nn2.py code, compare with
the probs pickles the daemon wrote (GPU, bf16). Shows (a) which test data the daemon used, (b) what the .pt alone
does not carry (COLMASK / RSTAT / architecture come from the queue line)."""
import ast, sys, pickle, numpy as np, torch, torch.nn as nn
torch.set_num_threads(4)
SRC = open("C:/Users/user/tfb/src/t2_nn2.py", encoding="utf-8").read(); tree = ast.parse(SRC)
G = {"torch": torch, "nn": nn, "np": np, "dev": torch.device("cpu")}
names = list(np.load("E:/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)); F = len(names)
G.update(names=names, F=F, CAPC=names.index("capl"), LANEC=names.index("lanes"), ENC={"rstat": 0, "tok": 0, "edges": None})
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in {"DilConv", "Net", "_sh", "rel_static", "encode", "enc_width"}:
        exec(compile(ast.Module([node], []), "t2_nn2.py", "exec"), G)
LMAX = int(np.max(np.diff(np.load("E:/tfb_nn/nn_data_afdem/off.npy"))))
NOMEM = "pos+tod+dow+capl+lanes+length+vf+w"
RUNS = [("q24f_attnglob_s0", dict(glob=1, attn=1, rstat=0, zero=NOMEM)), ("q24f_r13_s0", dict(glob=0, attn=0, rstat=0, zero=NOMEM)),
        ("q26f_attnglob_s0", dict(glob=1, attn=1, rstat=1, zero=NOMEM)), ("q26f_r13_s1", dict(glob=0, attn=0, rstat=1, zero=NOMEM)),
        ("q19f_drop1_s0", dict(glob=1, attn=1, rstat=0, zero="pos"))]


def infer(tag, cfg, test_dir, colmask_zero=None, rstat=None):
    sd_ = torch.load(f"E:/tfb_nn/models/{tag}.pt", map_location="cpu")
    G["ENC"].update(rstat=cfg["rstat"] if rstat is None else rstat, tok=0, edges=None)
    cm = torch.ones(F)
    for c in [c for c in (cfg["zero"] if colmask_zero is None else colmask_zero).split("+") if c]:
        cm[names.index(c)] = 0.0
    G["COLMASK"] = cm
    net = G["Net"](G["enc_width"](), 128, 10, 5, 0.0, cfg["glob"], cfg["attn"], 4, 0)
    net.load_state_dict({k: v for k, v in sd_.items() if not k.startswith("_")}); net.eval()
    mu, sd = sd_["_mu"], sd_["_sd"]
    Xt = np.load(f"E:/tfb_nn/{test_dir}/X.npy").copy(); offt = np.load(f"E:/tfb_nn/{test_dir}/off.npy")
    wid = np.load(f"E:/tfb_nn/{test_dir}/window_id.npy", allow_pickle=True).astype(str); condt = np.load(f"E:/tfb_nn/{test_dir}/cond.npy", allow_pickle=True).astype(str)
    for c in ("prior_on", "prior_og"):
        Xt[:, names.index(c)] = 0.0
    out = {}
    with torch.no_grad():
        for i in range(len(wid)):
            a, b = offt[i], offt[i + 1]; n = b - a
            v = torch.from_numpy(np.asarray(Xt[a:b], dtype=np.float32))[None]; nm = ~torch.isfinite(v); v = torch.where(nm, torch.zeros_like(v), v)
            xb = torch.zeros((1, LMAX, G["enc_width"]())); mb = torch.zeros((1, LMAX))
            xb[0, :n] = G["encode"](v, nm, mu, sd, torch.ones((1, n), dtype=torch.bool))[0]; mb[0, :n] = 1
            p = torch.sigmoid(net(xb, mb)[..., :6].float()).numpy()[0, :n].T.copy()
            if condt[i] == "queue_onset":
                p[:5] = 0.0
            out[wid[i]] = p
    return out


def cmp(a, b):
    d = np.concatenate([np.abs(a[k] - b[k]).ravel() for k in a]); return f"mean|dp| {d.mean():.2e} max {d.max():.3f}"


for tag, cfg in RUNS:
    ref = pickle.load(open(f"C:/Users/user/tfb/work/t2/probs_{tag}.pkl", "rb"))
    print(f"{tag}: CPU fp32 afdem vs daemon probs: {cmp(infer(tag, cfg, 'nn_test_afdem'), ref)} | "
          f"same weights on nn_test_dem: {cmp(infer(tag, cfg, 'nn_test_dem'), ref)}", flush=True)
# what if the COLMASK / RSTAT flag were not known (only the .pt): default COLMASK = pos only, RSTAT off
tag, cfg = RUNS[0]; ref = pickle.load(open(f"C:/Users/user/tfb/work/t2/probs_{tag}.pkl", "rb"))
print(f"{tag} re-inferred with ZERO=pos (COLMASK not in the .pt): {cmp(infer(tag, cfg, 'nn_test_afdem', colmask_zero='pos'), ref)}")
try:
    infer("q26f_attnglob_s0", RUNS[2][1], "nn_test_afdem", rstat=0)
except Exception as e:
    print("q26f_attnglob_s0 loaded without RSTAT:", type(e).__name__, str(e)[:120])
