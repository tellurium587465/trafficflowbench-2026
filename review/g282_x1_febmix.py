"""G282 (11/05, ledger #229 stage 1, declared 11/04 22:05 before the result): the proxy of "March's windows mixed into April's X1".
G141's same-day X1 (arm sd, networks c24f_attnglob s0 / s1, M3 v3 dev days, hold-out days not computed) with February windows of the
training layout (nn_data_jf_sp1, dates 2031-02-*, both conditions, names identical to nn_data_march) added to every fine-tune's pool:
n_feb = round(r * n_pool) drawn without replacement, the draw seeded by (day, k, hour) so it is the same for both networks.
The days are split by parity of the day number (D0 + 1): r in {0.25, 0.5} on the odd days (selection), the chosen r on the even days.
Writes work/t2/x1_arms/parts/mix<100r>__<tag>__<D0>.pkl per day (restart-safe), log mix<100r>_<odd|even>__<tag>.log.
usage: TFB_M3VER=v3 py -B -X utf8 g282_x1_febmix.py dry <tag> | fit <r> <odd|even> <tag>"""
import os, sys
R_ = r"C:/Users/user/tfb/review/"; sys.dont_write_bytecode = True; ARGS = sys.argv[1:]
exec(compile(open(R_ + "g141_x1_arms.py", encoding="utf-8").read().split("MODE = sys.argv[1]")[0], "g141", "exec"))
FD = FT.data("nn_data_jf_sp1"); FEB = np.array([i for i, x in enumerate(FD["date"]) if str(x).startswith("2031-02")], dtype=int)
assert list(np.load(f"{NN}/nn_data_jf_sp1/names.npy", allow_pickle=True)) == list(np.load(f"{NN}/nn_data_march/names.npy", allow_pickle=True))


def finetune_mix(nt, ks, fk, lr=2e-4, epochs=4, batch_size=32, wd=1e-3, wpow=1.0, fdrop=0.05, seed=0):
    """Net.finetune with items from two data sets: the March pool ks (MD) then the February windows fk (FD); fk empty = Net.finetune"""
    net, mu, sd, cm = nt.net, nt.mu, nt.sd, nt.cm; net.load_state_dict(nt.s0); items = [(MD, int(i)) for i in ks] + [(FD, int(j)) for j in fk]
    if not items:
        net.eval(); return 0
    wts = np.ones(len(items)); gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); net.train()
    for ep in range(epochs):
        order = rng.permutation(len(items))
        for b0 in range(0, len(order), batch_size):
            sel = order[b0:b0 + batch_size]; xb, y, e, m = FT.batch([items[i] for i in sel], cm, mu, sd, fdrop, gen)
            l = torch.nn.functional.binary_cross_entropy_with_logits(net(xb, m)[..., :6], y, reduction="none") * e
            wb = 100.0 * torch.clamp((y * e).sum(dim=(1, 2)), min=3) ** (-wpow) * torch.tensor(wts[sel], dtype=torch.float32)
            loss = (l.sum(dim=(1, 2)) * wb).sum() / (e.sum() + 1); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval(); return len(items)


def feb(D0, k, c, n, r):
    nf = int(round(r * n)); return np.sort(np.random.default_rng(1_000_003 * D0 + 1009 * k + c).choice(FEB, nf, replace=False)) if nf else np.zeros(0, int)


if ARGS[0] == "dry":
    tag = ARGS[1]; nt = Net(tag); G = groups("sd"); D0 = 4; (d, c), wids = next((kv for kv in sorted(G.items(), key=str) if kv[0][0] == D0)); saved = pickle.load(open(OUT + f"parts/sd__{tag}__{D0}.pkl", "rb")); t0 = time.time()
    for k in (0, 2):
        ks = pool(D0, k, c * 12); finetune_mix(nt, ks, []); pr = nt.predict(wids); print(f"r = 0, day {D0} hour {c} k {k}: pool {len(ks)}, max |dp| against the stored sd part {max(float(np.abs(pr[w] - saved[(w, k)]).max()) for w in wids):.2e} ({time.time() - t0:.0f}s)", flush=True)
        fk = feb(D0, k, c, len(ks), 0.5); finetune_mix(nt, ks, fk); p2 = nt.predict(wids); print(f"r = 0.5: + {len(fk)} February windows, mean |dp| against r = 0 {float(np.mean([np.abs(p2[w] - pr[w]).mean() for w in wids])):.2e} ({time.time() - t0:.0f}s)", flush=True)
    print("February windows available", len(FEB), "conditions", dict(zip(*np.unique(np.load(f"{NN}/nn_data_jf_sp1/cond.npy", allow_pickle=True)[FEB], return_counts=True))), flush=True); sys.exit()
r, par, tag = float(ARGS[1]), ARGS[2], ARGS[3]; arm = f"mix{int(round(100 * r))}"; nt = Net(tag); log = open(OUT + f"{arm}_{par}__{tag}.log", "a"); t0 = time.time(); G = groups("sd"); nft = nfe = 0
for D0 in sorted({d for d, _ in G if (d + 1) % 2 == (1 if par == "odd" else 0)}):
    f = OUT + f"parts/{arm}__{tag}__{D0}.pkl"
    if os.path.exists(f):
        continue
    out = {}
    for (d, c), wids in sorted(G.items(), key=str):
        if d != D0:
            continue
        for k in range(0, min(6, D0) + 1):
            ks = pool(D0, k, c * 12); fk = feb(D0, k, c, len(ks), r); finetune_mix(nt, ks, fk); nft += 1; nfe += len(fk); pr = nt.predict(wids); out.update({(w, k): pr[w] for w in wids})
    pickle.dump(out, open(f + ".tmp", "wb")); os.replace(f + ".tmp", f); log.write(f"day {D0 + 1}: {nft} fine-tunes, {nfe} February windows, {time.time() - t0:.0f}s {time.strftime('%H:%M:%S')}\n"); log.flush()
log.write(f"ALLDONE {time.time() - t0:.0f}s\n"); log.close()
