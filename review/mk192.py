"""11/03 ledger #192: tfb_a8/src/t1_unet_pre.py = t1_unet_train.py (#118) with the training set widened, by substitution only:
February back in (no February harness), and the test months' visible cells (validation = March, private = April) hidden with another
day's mask as the month fine-tune does (#132); U_PRE_DAYS=odd keeps their odd days only (the even-day instrument then reads cells the
pretraining never saw as targets), all = every day (production). Restart-safe: a checkpoint per epoch (weights, optimiser, schedule,
scaler, generators) in U_OUT, resumed when present. usage: py mk192.py"""
import hashlib
s = open(r"C:/Users/user/tfb_a8/src/t1_unet_train.py", encoding="utf-8").read()


def sub(a, b):
    global s
    assert s.count(a) == 1, a[:80]
    s = s.replace(a, b)


sub("# TFB (11/01, ledger #118): Task 1 image network.",
    "# TFB (11/03, ledger #192): t1_unet_train.py with February and the test months' visible cells in training (U_PRE_DAYS odd | all), restart-safe per epoch.\n# TFB (11/01, ledger #118): Task 1 image network.")
sub('DEADLINE_H = 9.0; DRY = os.environ.get("U_DRY", "") == "1"', 'DEADLINE_H = 99.0; DRY = os.environ.get("U_DRY", "") == "1"; PRE_DAYS = os.environ.get("U_PRE_DAYS", "odd")')
sub('    idx = [(p, i) for p in PANELS for i, dd in enumerate(D[p]["train_days"]) if not str(dd).startswith("2031-02")]; fe = [(p, i) for p in PANELS for i, dd in enumerate(D[p]["train_days"]) if str(dd).startswith("2031-02")]\n',
    '    idx = [(p, i, "train") for p in PANELS for i in range(len(D[p]["train_days"]))]; fe = []\n'
    '    for sp in ("validation", "private"):                        # 11/03 #192: the test months, visible cells only, hidden as the month fine-tune does\n'
    '        idx += [(p, i, sp) for p in PANELS for i in range(D[p][sp + "_Vn"].shape[0] // 288) if PRE_DAYS == "all" or i % 2 == 0]\n')
sub('    def sample(p, i, hide=None):\n        d = D[p]; a, b = i * 288, (i + 1) * 288; V, Q, f, E = d["train_Vn"][a:b], d["train_Qn"][a:b], d["train_fin"][a:b], d["train_E"][a:b]\n'
    '        h = d["hide"][rng.integers(d["ndays"]) * 288:][:288] if hide is None else hide; vis = f & ~h; x, fv, fq = image(p, V, Q, vis, np.arange(288), d["stat"], wk(d["train_days"][i]))',
    '    def sample(p, i, sp="train", hide=None):\n        d = D[p]; a, b = i * 288, (i + 1) * 288; V, Q, f, E = d[sp + "_Vn"][a:b], d[sp + "_Qn"][a:b], d[sp + "_fin"][a:b], d[sp + "_E"][a:b]\n'
    '        h = d["hide"][rng.integers(d["ndays"]) * 288:][:288] if hide is None else hide; vis = f & ~h; x, fv, fq = image(p, V, Q, vis, np.arange(288), d["stat"], wk(d[sp + "_days"][i]))')
sub("    st = 0; t1 = time.time()\n    for ep in range(EP):\n",
    "    st = 0; t1 = time.time(); ep0 = 0; ck = f\"{OUT}ckpt_s{seed}.pt\"\n"
    "    if os.path.exists(ck):                                      # 11/03: resume after a reboot\n"
    "        c_ = torch.load(ck, map_location=dev, weights_only=False); net.load_state_dict(c_[\"net\"]); opt.load_state_dict(c_[\"opt\"]); sch.load_state_dict(c_[\"sch\"]); scaler.load_state_dict(c_[\"scaler\"])\n"
    "        rng.bit_generator.state = c_[\"rng\"]; torch.set_rng_state(c_[\"trng\"]); ep0, st = c_[\"ep\"], c_[\"st\"]; print(f\"resumed seed {seed} at epoch {ep0}\", flush=True)\n"
    "    for ep in range(ep0, EP):\n")
sub("        order = rng.permutation(len(idx)); by = {}\n        for k in order:                                           # same-corridor batches, widths differ\n            by.setdefault(idx[k][0], []).append(idx[k][1])\n",
    "        order = rng.permutation(len(idx)); by = {}\n        for k in order:                                           # same-corridor batches, widths differ\n            by.setdefault(idx[k][0], []).append(idx[k][1:])\n")
sub("            S = [sample(p, i) for i in ii];", "            S = [sample(p, i, sp_) for i, sp_ in ii];")
sub("        rv, rq, s = feb() if (ep % 3 == 2 or ep == EP - 1 or DRY) else (0, 0, 0)\n", "        rv, rq, s = 0, 0, 0                                     # 11/03 #192: February is in training, no harness\n")
sub('        torch.save({k: v.cpu().half() for k, v in net.state_dict().items()} | dict(_seed=seed, _ep=ep + 1, _base=BASE), f"{OUT}unet_s{seed}.pt")\n',
    '        torch.save({k: v.cpu().half() for k, v in net.state_dict().items()} | dict(_seed=seed, _ep=ep + 1, _base=BASE), f"{OUT}unet_s{seed}.pt")\n'
    '        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(), ep=ep + 1, st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)\n')
open(r"C:/Users/user/tfb_a8/src/t1_unet_pre.py", "w", encoding="utf-8").write(s)
print("written", hashlib.sha256(s.encode()).hexdigest()[:8])
