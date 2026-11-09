"""11/03 ledger #194: C:/tfb_data/kaggle_up/k_t1mo/tfb-t1-ftd.py = the #132 month fine-tune (tfb-t1-ft.py) with the hide pattern thinned:
each run of hidden rows of one link in the borrowed mask h is kept with probability rho (U_RHO; 1 = the current fine-tune, whose input is
the release's visible cells minus a whole other-day mask, about 0.36 visible against about 0.6 at inference). Stage A only (32 epochs on
the odd days, the same number of steps for every rho), read on the even days under read-rho 0.15, 0.3 and 1 with fixed draws, per panel.
usage: py mk194.py"""
import hashlib
s = open("C:/tfb_data/kaggle_up/k_t1ft/tfb-t1-ft.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)


sub("# TFB (11/01, ledger #132):", "# TFB (11/03, ledger #194): the hide pattern of the month fine-tune thinned to a share rho of its runs (U_RHO), stage A only, reads per panel.\n# TFB (11/01, ledger #132):")
sub("EPS = (16, 32); BS = 3;", "EPS = (32,); RHO = float(os.environ.get(\"U_RHO\", \"1\")); RREAD = (0.15, 0.3, 1.0); BS = 3;")
sub("import os, sys, glob, time, subprocess, numpy as np", "import os, sys, glob, time, json, subprocess, numpy as np")
sub('sd0 = torch.load(find("unet_s4.pt") + f"unet_s{SEED}.pt", map_location="cpu", weights_only=False)', 'sd0 = torch.load(WD_ + f"unet_s{SEED}.pt", map_location="cpu", weights_only=False)')
sub("def sample(p, sp, i, rng_):\n    d = D[p]; m_ = d[sp]; a, b = i * 288, (i + 1) * 288; h = d[\"hide\"][rng_.integers(d[\"nd\"]) * 288:][:288];",
    "def thin(h, rho, rng_):\n"
    "    if rho >= 1.0:\n"
    "        return h\n"
    "    st = np.diff(np.vstack([np.zeros((1, h.shape[1]), bool), h]).astype(np.int8), axis=0) == 1     # run starts, per link\n"
    "    rid = np.cumsum(st.ravel(order=\"F\")).reshape(h.shape[::-1]).T; keep = rng_.random(int(rid.max()) + 1) < rho\n"
    "    return h & keep[rid]\n\n\n"
    "def sample(p, sp, i, rng_, rho=1.0):\n    d = D[p]; m_ = d[sp]; a, b = i * 288, (i + 1) * 288; h = thin(d[\"hide\"][rng_.integers(d[\"nd\"]) * 288:][:288], rho, rng_);")
sub("            S = [sample(p, sp, int(i), rng) for i in ii];", "            S = [sample(p, sp, int(i), rng, RHO) for i in ii];")
sub("def read(net, sp, days):\n    net.eval(); r2 = np.random.default_rng(99); sv = sq = n = 0.0",
    "def read(net, sp, days, rho=1.0):\n    net.eval(); r2 = np.random.default_rng(99); sv = sq = n = 0.0; PP = {}")
sub("                x, tv, tq, m, vf, cl = sample(p, sp, int(i), r2);", "                x, tv, tq, m, vf, cl = sample(p, sp, int(i), r2, rho);")
sub("ev = (o[0] - ten(tv))[e] * ten(vf).expand_as(ten(tv))[e]; eq = (o[1] - ten(tq))[e] * ten(cl).expand_as(ten(tq))[e]; sv += float(ev.square().sum()); sq += float(eq.square().sum()); n += float(e.sum())",
    "ev = (o[0] - ten(tv))[e] * ten(vf).expand_as(ten(tv))[e]; eq = (o[1] - ten(tq))[e] * ten(cl).expand_as(ten(tq))[e]; sv += float(ev.square().sum()); sq += float(eq.square().sum()); n += float(e.sum())\n"
    "                a_ = PP.setdefault(p, [0.0, 0.0, 0.0]); a_[0] += float(ev.square().sum()); a_[1] += float(eq.square().sum()); a_[2] += float(e.sum())")
sub("    net.train(); rv, rq = (sv / max(n, 1)) ** 0.5, (sq / max(n, 1)) ** 0.5; return rv, rq, 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)\n",
    "    net.train(); RS[0] = PP; rv, rq = (sv / max(n, 1)) ** 0.5, (sq / max(n, 1)) ** 0.5; return rv, rq, 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)\n\n\n"
    "RS = [None]\n\n\n"
    "def reads(net, sp, ev_, stage, mo):\n"
    "    for rr in RREAD:\n"
    "        r = read(net, sp, ev_, rr); print(f\"RES seed {SEED} {mo} rho_train {RHO} {stage} rho_read {rr} S {r[2]:.5f} \" + json.dumps(RS[0]), flush=True)\n")
sub("net = fresh(); print(f\"seed {SEED} {mo} stage A base\", [round(x, 4) for x in read(net, sp, ev_)], flush=True); done = 0",
    "net = fresh(); reads(net, sp, ev_, \"base\", mo); done = 0")
sub("        train(net, sp, od, e_ - done); done = e_; print(f\"seed {SEED} {mo} stage A after {e_} epochs on odd days: even days\", [round(x, 4) for x in read(net, sp, ev_)], f\"{(time.time() - T0) / 60:.0f} min\", flush=True)\n",
    "        train(net, sp, od, e_ - done); done = e_; reads(net, sp, ev_, f\"ft{e_}\", mo); print(f\"seed {SEED} {mo} rho {RHO} stage A {e_} epochs done, {(time.time() - T0) / 60:.0f} min\", flush=True)\n")
sub("    for e_ in (32,):\n", "    for e_ in ():\n")
open("C:/tfb_data/kaggle_up/k_t1mo/tfb-t1-ftd.py", "w", encoding="utf-8").write(s)
print("written", hashlib.sha256(s.encode()).hexdigest()[:8])
