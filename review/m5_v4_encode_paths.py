"""M5 audit V4: run the real t2_nn2.py functions (extracted by AST, executed on CPU) on a window subset and compare
the training gather (BUCKET 0), the packed gather_rows (BUCKET 2) and the FINAL test-time path; also old (97f824b)
vs new encoding, RNG consumption of the eval-only perturbations, and forward equality padded-batch vs single window."""
import ast, sys, textwrap, numpy as np, torch, torch.nn as nn
torch.set_num_threads(4)
SRC = open("C:/Users/user/tfb/src/t2_nn2.py", encoding="utf-8").read()
tree = ast.parse(SRC)
WANT = {"DilConv", "Net", "jitter", "counterfact", "_sh", "rel_static", "encode", "enc_width", "gather", "gather_rows", "stats", "focal_w"}
G = {"torch": torch, "nn": nn, "np": np, "dev": torch.device("cpu")}
d = "E:/tfb_nn/nn_data_afdem/"
names = list(np.load(d + "names.npy", allow_pickle=True)); F = len(names)
offF = np.load(d + "off.npy"); X = np.load(d + "X.npy", mmap_mode="r"); cond = np.load(d + "cond.npy", allow_pickle=True).astype(str)
LMAX = int(np.max(np.diff(offF)))
rng = np.random.default_rng(1)
ks = np.concatenate([rng.choice(np.flatnonzero(cond == "queue_onset"), 40, replace=False), rng.choice(np.flatnonzero(cond == "queue_ongoing"), 80, replace=False),
                     [int(np.argmax(np.diff(offF)))]])                     # include a longest window
rows = np.concatenate([np.arange(offF[k], offF[k + 1]) for k in ks])
Xs = np.asarray(X[rows], dtype=np.float32).copy()
for c in ("prior_on", "prior_og"):
    Xs[:, names.index(c)] = 0.0
OFF = np.concatenate([[0], np.cumsum(np.diff(offF)[ks])]); NW = len(ks)
m = ~np.isfinite(Xs)
G.update(names=names, F=F, OFF=OFF, LMAX=LMAX, Xg=torch.from_numpy(np.where(m, 0, Xs).astype(np.float32)), Mg=torch.from_numpy(m),
         Yg=torch.zeros((len(Xs), 6), dtype=torch.bool), Eg=torch.ones((len(Xs), 6), dtype=torch.bool))
G.update(starts=torch.from_numpy(OFF[:-1]), lens=torch.from_numpy(np.diff(OFF)), ar=torch.arange(LMAX), HOLD=-1, NEARM=None, G_IDX=None,
         BLK=torch.zeros(len(Xs), dtype=torch.int8), BS=30)
G["VCOL"] = [names.index(c) for c in ("v_T", "v_T1", "v_T2", "v_m3", "v_m12", "v_min", "last_v", "v_trend") if c in names]
G["NBV"] = [(names.index(f"nb{o}_v"), o) for o in range(-5, 6) if f"nb{o}_v" in names]
G["TODC"], G["DOWC"] = names.index("tod"), names.index("dow")
G["SCOL"] = [names.index(c) for c in ("capl", "lanes", "length", "vf", "w") if c in names]
G["CAPC"], G["LANEC"] = names.index("capl"), names.index("lanes")
G["ENC"] = {"rstat": 0, "tok": 0, "edges": None}
G["STATMASK"] = torch.zeros(F, dtype=torch.bool)
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in WANT:
        exec(compile(ast.Module([node], []), "t2_nn2.py", "exec"), G)
# the TOK edge block and the FINAL per-window block, verbatim from run()
lines = SRC.splitlines()
def block(start_pat, n):
    i = next(j for j, l in enumerate(lines) if start_pat in l)
    return textwrap.dedent("\n".join(lines[i:i + n]))
TOKB = block("rows = torch.cat([torch.arange(OFF[k], OFF[k + 1], device=dev) for k in tr[::29]])", 5)
FINB = block("a, b = offt[i], offt[i + 1]; n = b - a", 5)
print("TOK block:\n" + TOKB + "\nFINAL block:\n" + FINB)
NOMEM = "pos+tod+dow+capl+lanes+length+vf+w"


def setup(rstat, tok, zero):
    torch.manual_seed(0)
    G["ENC"].update(rstat=rstat, tok=tok, edges=None)
    cm = torch.ones(F)
    for c in [c for c in zero.split("+") if c]:
        cm[names.index(c)] = 0.0
    G["COLMASK"] = cm
    mu, sd = G["stats"](np.arange(NW))
    if tok:
        L = dict(G, tr=np.arange(NW), mu=mu, sd=sd, TOK=tok); exec(TOKB, L)
    return mu, sd


def final_path(mu, sd):
    out = []
    L = dict(G, mu=mu, sd=sd, Xt=Xs, offt=OFF)                       # Xs: NaN kept, priors 0, as Dt["X"] in FINAL
    for i in range(NW):
        L["i"] = i; exec(FINB, L); out.append(L["xb"][0, :L["n"]].clone())
    return out


def train_path(mu, sd, bs=40):
    out = []
    for a in range(0, NW, bs):
        k = torch.arange(a, min(a + bs, NW)); xb, yb, eb, mb = G["gather"](k, mu, sd, LMAX)
        for j, kk in enumerate(k.tolist()):
            n = OFF[kk + 1] - OFF[kk]; out.append(xb[j, :n].clone())
            assert (xb[j, n:] == 0).all() and (mb[j, n:] == 0).all() and (mb[j, :n] == 1).all()
    return out


def packed_path(mu, sd, LP=512, GAP=32):
    rows, rem = [], []                                               # first-fit as in pack(), deterministic order
    for w in range(NW):
        n = int(OFF[w + 1] - OFF[w]); hit = [r for r in range(len(rows)) if rem[r] >= n + GAP]
        if hit:
            r = hit[0]; o = LP - rem[r] + GAP
        else:
            r = len(rows); rows.append([]); rem.append(LP); o = 0
        rows[r].append((w, o)); rem[r] = LP - (o + n)
    IDX = np.zeros((len(rows), LP), np.int64); VAL = np.zeros((len(rows), LP), bool)
    for r, row in enumerate(rows):
        for w, o in row:
            n = OFF[w + 1] - OFF[w]; IDX[r, o:o + n] = np.arange(OFF[w], OFF[w] + n); VAL[r, o:o + n] = True
    xb, yb, eb, vb = G["gather_rows"](torch.from_numpy(IDX), torch.from_numpy(VAL), mu, sd)
    out = [None] * NW
    for r, row in enumerate(rows):
        for w, o in row:
            n = OFF[w + 1] - OFF[w]; out[w] = xb[r, o:o + n].clone()
    return out, IDX, VAL, rows


def maxdiff(A, B):
    return max(float((a - b).abs().max()) for a, b in zip(A, B))


for rstat, tok, zero in ((0, 0, "pos"), (0, 0, NOMEM), (1, 0, NOMEM), (0, 8, NOMEM), (1, 8, NOMEM)):
    mu, sd = setup(rstat, tok, zero)
    A = train_path(mu, sd); B = final_path(mu, sd); C, IDX, VAL, prow = packed_path(mu, sd)
    w = G["enc_width"]()
    print(f"RSTAT={rstat} TOK={tok} ZERO={zero[:12]:12s} width {w}: train vs FINAL max|diff| {maxdiff(A, B):.3g}, "
          f"packed vs FINAL {maxdiff(C, B):.3g}, zeroed-col flags in FINAL {sum(int(b[:, F:2*F][:, (G['COLMASK'] == 0)].sum()) for b in B)}")
    if tok:
        e = G["ENC"]["edges"]; print("   edges shape", tuple(e.shape), "non-increasing rows", int((e[:, 1:] < e[:, :-1]).any(1).sum()),
                                     "flat rows (all equal)", int((e[:, -1] == e[:, 0]).sum()))

# old (97f824b) gather / FINAL encoding vs the new encode (RSTAT=0, TOK=0)
mu, sd = setup(0, 0, NOMEM); CM = G["COLMASK"]
old_tr, old_fin = [], []
for i in range(NW):
    a, b = OFF[i], OFF[i + 1]; idx = torch.arange(a, b)[None]
    x = (G["Xg"][idx] - mu) / sd * CM; nm = G["Mg"][idx] & (CM > 0); x = torch.where(nm, torch.zeros_like(x), x.clamp(-10, 10))
    old_tr.append(torch.cat([x, nm.float()], -1)[0])
    v = Xs[a:b]; nmn = ~np.isfinite(v); mu_c, sd_c = mu.numpy(), sd.numpy(); vv = np.where(nmn, 0, (np.where(nmn, 0, v) - mu_c) / sd_c)
    vv = np.clip(vv, -10, 10); cm = CM.numpy(); vv = vv * cm; nmn = nmn & (cm > 0)
    old_fin.append(torch.from_numpy(np.concatenate([vv, nmn], -1).astype(np.float32)))
B = final_path(mu, sd)
print(f"old train vs new {maxdiff(old_tr, B):.3g}; old FINAL vs new {maxdiff(old_fin, B):.3g}")
# old training with FDROP: flags of zeroed columns get the dropout noise, the old FINAL path never sets them
torch.manual_seed(0); idx = torch.arange(OFF[0], OFF[1])[None]
nm = (G["Mg"][idx] & (CM > 0)) | (torch.rand(idx.shape + (F,)) < 0.05)
print("old FDROP=0.05: share of zeroed-column flags = 1 in training", float(nm[..., CM == 0].float().mean()), "(FINAL: 0)")

# eval-only perturbations must not touch the global RNG
mu, sd = setup(1, 0, NOMEM)
s0 = torch.get_rng_state(); gen = torch.Generator(); gen.manual_seed(1234)
G["gather"](torch.arange(0, 10), mu, sd, LMAX, jit=0.01, gen=gen, cf="tr")
print("global RNG untouched by jit+cf with a generator:", torch.equal(s0, torch.get_rng_state()))
G["gather"](torch.arange(0, 10), mu, sd, LMAX, crop=0.6)
print("CROP consumes the global RNG (training only):", not torch.equal(s0, torch.get_rng_state()))

# forward equality: padded batch vs single window; packed rows vs single (conv-only)
torch.manual_seed(0); mu, sd = setup(1, 0, NOMEM)
for glob, attn in ((1, 1), (0, 0)):
    torch.manual_seed(0); net = G["Net"](G["enc_width"](), 64, 10, 5, 0.0, glob, attn, 4, 0).eval()
    with torch.no_grad():
        k = torch.arange(0, NW); xb, _, _, mb = G["gather"](k, mu, sd, LMAX); pb = torch.sigmoid(net(xb, mb))
        B = final_path(mu, sd); dmax = 0.0
        for i in range(NW):
            n = OFF[i + 1] - OFF[i]; x1 = torch.zeros((1, LMAX, G["enc_width"]())); m1 = torch.zeros((1, LMAX)); x1[0, :n] = B[i]; m1[0, :n] = 1
            p1 = torch.sigmoid(net(x1, m1))[0, :n]; dmax = max(dmax, float((p1 - pb[i, :n]).abs().max()))
        line = f"Net GLOBAL={glob} ATTN={attn}: padded batch vs single window max|dp| {dmax:.2e}"
        if not glob and not attn:
            C, IDX, VAL, prow = packed_path(mu, sd)
            xr, _, _, vr = G["gather_rows"](torch.from_numpy(IDX), torch.from_numpy(VAL), mu, sd); pr = torch.sigmoid(net(xr, vr)); dm2 = 0.0
            for r, row in enumerate(prow):
                for w, o in row:
                    n = OFF[w + 1] - OFF[w]; dm2 = max(dm2, float((pr[r, o:o + n] - pb[w, :n]).abs().max()))
            line += f"; packed rows vs padded {dm2:.2e}"
        print(line)
