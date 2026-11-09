# TFB (11/03, ledger #192): t1_unet_train.py with February and the test months' visible cells in training (U_PRE_DAYS odd | all), restart-safe per epoch.
# TFB (11/01, ledger #118): Task 1 image network. One day of one corridor as a (time, link) image: visible speed and flow, a visibility flag, the per-link day mean of the visible cells,
# time of day, weekend, statics. The mask is drawn afresh for every sample from the mask of another day of the same corridor. Output = residual to the day mean, speed and flow.
# Loss = 0.54 RMSE_v / 25 + 0.46 RMSE_q per lane / 600 on the hidden cells with a truth, as the Task 1 score. February is held out of training; no epoch is chosen on it.
# Written for this ledger entry, no public code. Private dataset, private kernel. The final weights go to the CPU at home for the February harness and for March and April.
import os, sys, glob, time, subprocess, numpy as np

EP = int(os.environ.get("U_EP", "90")); BS = int(os.environ.get("U_BS", "3")); BASE = int(os.environ.get("U_BASE", "96")); DEADLINE_H = 99.0; DRY = os.environ.get("U_DRY", "") == "1"; PRE_DAYS = os.environ.get("U_PRE_DAYS", "odd")
ROOT = os.environ.get("U_ROOT", ""); OUT = os.environ.get("U_OUT", "/kaggle/working/")


def find_root():
    if ROOT:
        return ROOT
    return os.path.dirname(glob.glob("/kaggle/input/**/D7_I10_E.npz", recursive=True)[0]) + "/"


if not (len(sys.argv) >= 2 and sys.argv[1].replace(",", "").isdigit()):   # launcher: one process per GPU
    import torch
    n = torch.cuda.device_count(); print("GPUs", [torch.cuda.get_device_name(i) for i in range(n)], flush=True); ps = []
    for g in range(max(n, 1)):
        seeds = [g + 4] if n >= 2 else [4, 5]
        ps.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g} python -u {globals().get('__file__', sys.argv[0])} " + ",".join(map(str, seeds)), shell=True))
    print("exit codes", [p.wait() for p in ps], flush=True); sys.exit()

import torch, torch.nn as nn, torch.nn.functional as Fn
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); T0 = time.time()


class Res(nn.Module):
    def __init__(s, c, d=1):
        super().__init__(); s.a = nn.Conv2d(c, c, 3, padding=d, dilation=d); s.b = nn.Conv2d(c, c, 3, padding=d, dilation=d); s.n1 = nn.GroupNorm(8, c); s.n2 = nn.GroupNorm(8, c)

    def forward(s, x):
        return x + s.b(Fn.gelu(s.n2(s.a(Fn.gelu(s.n1(x))))))


class Net(nn.Module):
    def __init__(s, fin, b):
        super().__init__(); c = (b, 2 * b, 4 * b, 4 * b); s.inp = nn.Conv2d(fin, c[0], 3, padding=1)
        s.enc = nn.ModuleList([nn.Sequential(Res(c[i]), Res(c[i])) for i in range(3)]); s.down = nn.ModuleList([nn.Conv2d(c[i], c[i + 1], 3, stride=2, padding=1) for i in range(3)])
        s.mid = nn.Sequential(Res(c[3], 1), Res(c[3], 2), Res(c[3], 4), Res(c[3], 8), Res(c[3], 1))
        s.up = nn.ModuleList([nn.ConvTranspose2d(c[i + 1], c[i], 2, stride=2) for i in range(3)]); s.dec = nn.ModuleList([nn.Sequential(nn.Conv2d(2 * c[i], c[i], 1), Res(c[i]), Res(c[i])) for i in range(3)]); s.out = nn.Conv2d(c[0], 2, 1)
        nn.init.zeros_(s.out.weight); nn.init.zeros_(s.out.bias)

    def forward(s, x):
        z = s.inp(x); sk = []
        for e, d in zip(s.enc, s.down):
            z = e(z); sk.append(z); z = d(z)
        z = s.mid(z)
        for i in (2, 1, 0):
            z = s.dec[i](torch.cat([s.up[i](z), sk[i]], 1))
        return s.out(z)


def fill(x, vis):
    """x (T, n), vis (T, n) bool -> the per-link mean of the visible cells, the panel's mean where a link has none"""
    n = vis.sum(0); m = (x * vis).sum(0) / np.maximum(n, 1); return np.where(n > 0, m, m[n > 0].mean() if (n > 0).any() else 0.5)


def image(p, V, Q, vis, rows, stat, wkend):
    """one day -> (channels, T, n) float32; V, Q already normalised by vf and cap"""
    v, q = np.nan_to_num(V) * vis, np.nan_to_num(Q) * vis; fv, fq = fill(v, vis), fill(q, vis); T, n = V.shape; t = (rows % 288) / 288 * 2 * np.pi; ones = np.ones((T, n), np.float32)
    c = [v - fv * vis, q - fq * vis, vis.astype(np.float32), fv * ones, fq * ones, np.sin(t)[:, None] * ones, np.cos(t)[:, None] * ones, wkend * ones] + [s_[None, :] * ones for s_ in stat]
    return np.stack(c).astype(np.float32), fv, fq


def pad8(x):
    n = x.shape[-1]; r = (8 - n % 8) % 8; return (Fn.pad(x, (0, r, 0, 0), mode="replicate") if r else x), n


def load_all():
    root = find_root(); D = {}
    for p in PANELS:
        z = np.load(root + p + ".npz", allow_pickle=True); d = {k: z[k] for k in z.files}; vf, cap = d["vf"], d["cap"]
        for sp in ("train", "validation", "private"):
            d[sp + "_Vn"] = d.pop(sp + "_V") / vf[None, :]; d[sp + "_Qn"] = d.pop(sp + "_Q") / cap[None, :]
        d["stat"] = [vf / 120, d["lanes"] / 6, cap / 12000, np.minimum(d["length"], 3) / 3, d["on"].astype(np.float32), d["off"].astype(np.float32)]
        d["hide"] = np.concatenate([d["train_target"], ~d["validation_fin"], ~d["private_fin"]], 0); d["ndays"] = d["hide"].shape[0] // 288
        D[p] = d; print(p, f"{time.time() - T0:.0f}s", flush=True)
    return D


def wk(day):
    return float(np.datetime64(str(day)[:10]).astype("datetime64[D]").astype(int) % 7 in (2, 3))   # 1970-01-01 was a Thursday


def run(seed):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); D = load_all(); fin = 13 + 1
    idx = [(p, i, "train") for p in PANELS for i in range(len(D[p]["train_days"]))]; fe = []
    for sp in ("validation", "private"):                        # 11/03 #192: the test months, visible cells only, hidden as the month fine-tune does
        idx += [(p, i, sp) for p in PANELS for i in range(D[p][sp + "_Vn"].shape[0] // 288) if PRE_DAYS == "all" or i % 2 == 0]
    print("train days", len(idx), "february", len(fe), flush=True); net = Net(fin, BASE).to(dev); print("params", sum(x.numel() for x in net.parameters()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4); steps = len(idx) // BS * EP; sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=max(steps, 2), pct_start=0.05); scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    def sample(p, i, sp="train", hide=None):
        d = D[p]; a, b = i * 288, (i + 1) * 288; V, Q, f, E = d[sp + "_Vn"][a:b], d[sp + "_Qn"][a:b], d[sp + "_fin"][a:b], d[sp + "_E"][a:b]
        h = d["hide"][rng.integers(d["ndays"]) * 288:][:288] if hide is None else hide; vis = f & ~h; x, fv, fq = image(p, V, Q, vis, np.arange(288), d["stat"], wk(d[sp + "_days"][i]))
        tv = np.nan_to_num(V) - fv; tq = np.nan_to_num(Q) - fq; m = f & E & ~vis; return x, tv.astype(np.float32), tq.astype(np.float32), m, d["vf"], d["cap"] / d["lanes"]

    def loss_of(x, tv, tq, m, vf, cl):
        o, _ = pad8(x); o = net(o)[..., :x.shape[-1]]; ev = (o[:, 0] - tv)[m] * vf.expand_as(tv)[m]; eq = (o[:, 1] - tq)[m] * cl.expand_as(tq)[m]; rv = (ev.square().mean() + 1e-6).sqrt(); rq = (eq.square().mean() + 1e-6).sqrt(); return 0.54 * rv / 25 + 0.46 * rq / 600, rv, rq

    def feb():
        net.eval(); sv = sq = n = 0.0
        with torch.no_grad():
            for p, i in fe[::3]:
                d = D[p]; x, tv, tq, m, vf, cl = sample(p, i, hide=d["train_target"][i * 288:(i + 1) * 288]); ten = lambda a: torch.from_numpy(a)[None].to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    o = pad8(ten(x))[0]; o = net(o)[..., :x.shape[-1]].float()
                mm = ten(m); ev = (o[:, 0] - ten(tv))[mm] * ten(vf).expand_as(ten(tv))[mm]; eq = (o[:, 1] - ten(tq))[mm] * ten(cl).expand_as(ten(tq))[mm]; sv += float(ev.square().sum()); sq += float(eq.square().sum()); n += float(mm.sum())
        net.train(); rv, rq = (sv / n) ** 0.5, (sq / n) ** 0.5; return rv, rq, 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)

    st = 0; t1 = time.time(); ep0 = 0; ck = f"{OUT}ckpt_s{seed}.pt"
    if os.path.exists(ck):                                      # 11/03: resume after a reboot
        c_ = torch.load(ck, map_location=dev, weights_only=False); net.load_state_dict(c_["net"]); opt.load_state_dict(c_["opt"]); sch.load_state_dict(c_["sch"]); scaler.load_state_dict(c_["scaler"])
        rng.bit_generator.state = c_["rng"]; torch.set_rng_state(c_["trng"]); ep0, st = c_["ep"], c_["st"]; print(f"resumed seed {seed} at epoch {ep0}", flush=True)
    for ep in range(ep0, EP):
        order = rng.permutation(len(idx)); by = {}
        for k in order:                                           # same-corridor batches, widths differ
            by.setdefault(idx[k][0], []).append(idx[k][1:])
        bat = [(p, ii[j:j + BS]) for p, ii in by.items() for j in range(0, len(ii) - BS + 1, BS)]; bat = [bat[k] for k in rng.permutation(len(bat))]; run_l = 0.0
        for nb, (p, ii) in enumerate(bat):
            S = [sample(p, i, sp_) for i, sp_ in ii]; ten = lambda a: torch.from_numpy(np.stack(a)).to(dev, non_blocking=True); x, tv, tq, m = ten([s[0] for s in S]), ten([s[1] for s in S]), ten([s[2] for s in S]), ten([s[3] for s in S])
            vf = torch.from_numpy(S[0][4].astype(np.float32)).to(dev)[None, None]; cl = torch.from_numpy(S[0][5].astype(np.float32)).to(dev)[None, None]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                l, rv, rq = loss_of(x, tv, tq, m, vf, cl)
            opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); st += 1
            if st <= steps:
                sch.step()
            run_l += float(l)
            if DRY and nb >= 1:
                break
            if time.time() - T0 > DEADLINE_H * 3600:
                break
        rv, rq, s = 0, 0, 0                                     # 11/03 #192: February is in training, no harness
        print(f"seed {seed} ep {ep + 1}/{EP} loss {run_l / max(nb + 1, 1):.5f} | feb rmse v {rv:.3f} q {rq:.2f} S {s:.4f} | {(time.time() - t1) / (ep + 1):.0f}s per epoch, {(time.time() - T0) / 60:.0f} min", flush=True)
        torch.save({k: v.cpu().half() for k, v in net.state_dict().items()} | dict(_seed=seed, _ep=ep + 1, _base=BASE), f"{OUT}unet_s{seed}.pt")
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(), scaler=scaler.state_dict(), rng=rng.bit_generator.state, trng=torch.get_rng_state(), ep=ep + 1, st=st), ck + ".tmp"); os.replace(ck + ".tmp", ck)
        if DRY or time.time() - T0 > DEADLINE_H * 3600:
            break


for s_ in sys.argv[1].split(","):
    run(int(s_))
