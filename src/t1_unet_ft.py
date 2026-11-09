# TFB (11/01, ledger #132): the image network of #118b fine-tuned on the month it predicts (Task 1 is offline, 742068). Stage A, the instrument: train on the odd days of March or April, read the even days, on the month's visible cells hidden with mask patterns. Stage B: fine-tune on
# the whole month, predict every day with the release's visible cells, save the (T, n) speed and flow. Fine-tune = AdamW lr 2e-4 falling to 0, epochs in EPS. Weights come from the private dataset tfb-t1-unet-w. Written for this ledger entry, no public code.
import os, sys, glob, time, subprocess, numpy as np

EPS = (8, 16, 32, 64); BS = 4; DRY = os.environ.get("U_DRY", "") == "1"; ROOT = os.environ.get("U_ROOT", ""); WD_ = os.environ.get("U_W", ""); OUT = os.environ.get("U_OUT", "/kaggle/working/")
if not (len(sys.argv) >= 2 and sys.argv[1].isdigit()):
    import torch
    n = torch.cuda.device_count(); print("GPUs", [torch.cuda.get_device_name(i) for i in range(n)], flush=True); ps = []
    for g, s_ in enumerate((2, 3)):
        ps.append(subprocess.Popen(f"CUDA_VISIBLE_DEVICES={g if n >= 2 else 0} python -u {globals().get('__file__', sys.argv[0])} {s_}", shell=True))
    print("exit codes", [p.wait() for p in ps], flush=True); sys.exit()
import torch, torch.nn as nn, torch.nn.functional as Fn
PANELS = ["D7_I10_E", "D7_I10_W", "D7_I210_E", "D7_I210_W", "D7_I405_N", "D7_I405_S", "D12_I5_N", "D12_I5_S", "D12_I405_N", "D12_I405_S"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); T0 = time.time(); SEED = int(sys.argv[1]); find = lambda f: ROOT or os.path.dirname(glob.glob(f"/kaggle/input/**/{f}", recursive=True)[0]) + "/"


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

    def forward(s, x):
        z = s.inp(x); sk = []
        for e, d in zip(s.enc, s.down):
            z = e(z); sk.append(z); z = d(z)
        z = s.mid(z)
        for i in (2, 1, 0):
            z = s.dec[i](torch.cat([s.up[i](z), sk[i]], 1))
        return s.out(z)


def fill(x, vis):
    n = vis.sum(0); m = (x * vis).sum(0) / np.maximum(n, 1); return np.where(n > 0, m, m[n > 0].mean() if (n > 0).any() else 0.5)


def image(V, Q, vis, stat, wkend):
    v, q = np.nan_to_num(V) * vis, np.nan_to_num(Q) * vis; fv, fq = fill(v, vis), fill(q, vis); T, n = V.shape; t = (np.arange(288) % 288) / 288 * 2 * np.pi; ones = np.ones((T, n), np.float32)
    c = [v - fv * vis, q - fq * vis, vis.astype(np.float32), fv * ones, fq * ones, np.sin(t)[:, None] * ones, np.cos(t)[:, None] * ones, wkend * ones] + [s_[None, :] * ones for s_ in stat]
    return np.stack(c).astype(np.float32), fv, fq


pad8 = lambda x: Fn.pad(x, (0, (8 - x.shape[-1] % 8) % 8, 0, 0), mode="replicate") if x.shape[-1] % 8 else x
wk = lambda day: float(np.datetime64(str(day)[:10]).astype("datetime64[D]").astype(int) % 7 in (2, 3))
D = {}
for p in PANELS:
    z = np.load(find("D7_I10_E.npz") + p + ".npz", allow_pickle=True); d = dict(vf=z["vf"], cap=z["cap"], lanes=z["lanes"], stat=[z["vf"] / 120, z["lanes"] / 6, z["cap"] / 12000, np.minimum(z["length"], 3) / 3, z["on"].astype(np.float32), z["off"].astype(np.float32)])
    for sp in ("validation", "private"):
        d[sp] = dict(V=z[sp + "_V"] / z["vf"][None, :], Q=z[sp + "_Q"] / z["cap"][None, :], f=z[sp + "_fin"], E=z[sp + "_E"], days=z[sp + "_days"])
    d["hide"] = np.concatenate([z["train_target"], ~z["validation_fin"], ~z["private_fin"]], 0); d["nd"] = d["hide"].shape[0] // 288; D[p] = d; print(p, f"{time.time() - T0:.0f}s", flush=True)
sd0 = torch.load(find("unet_s2.pt") + f"unet_s{SEED}.pt", map_location="cpu", weights_only=False); B0 = sd0["_base"]; rng = np.random.default_rng(SEED)


def fresh():
    n = Net(14, B0); n.load_state_dict({k: v.float() for k, v in sd0.items() if not k.startswith("_")}); return n.to(dev)


def sample(p, sp, i, rng_):
    d = D[p]; m_ = d[sp]; a, b = i * 288, (i + 1) * 288; h = d["hide"][rng_.integers(d["nd"]) * 288:][:288]; V, Q, f, E = m_["V"][a:b], m_["Q"][a:b], m_["f"][a:b], m_["E"][a:b]; vis = f & ~h; x, fv, fq = image(V, Q, vis, d["stat"], wk(m_["days"][i]))
    return x, (np.nan_to_num(V) - fv).astype(np.float32), (np.nan_to_num(Q) - fq).astype(np.float32), f & E & ~vis, d["vf"].astype(np.float32), (d["cap"] / d["lanes"]).astype(np.float32)


def train(net, sp, days, ep):
    opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); nst = max(sum(len(v) // BS for v in days.values()) * ep, 1); sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / nst)); scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); net.train()
    for _ in range(ep):
        bat = [(p, ii[j:j + BS]) for p, v in days.items() for ii in [list(rng.permutation(v))] for j in range(0, len(ii) - BS + 1, BS)]; bat = [bat[k] for k in rng.permutation(len(bat))]
        for p, ii in (bat[:2] if DRY else bat):
            S = [sample(p, sp, int(i), rng) for i in ii]; ten = lambda a: torch.from_numpy(np.stack(a)).to(dev); x, tv, tq, m = ten([s[0] for s in S]), ten([s[1] for s in S]), ten([s[2] for s in S]), ten([s[3] for s in S]); vf = torch.from_numpy(S[0][4]).to(dev)[None, None]; cl = torch.from_numpy(S[0][5]).to(dev)[None, None]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                o = net(pad8(x))[..., :x.shape[-1]]; ev = (o[:, 0] - tv)[m] * vf.expand_as(tv)[m]; eq = (o[:, 1] - tq)[m] * cl.expand_as(tq)[m]; l = 0.54 * (ev.square().mean() + 1e-6).sqrt() / 25 + 0.46 * (eq.square().mean() + 1e-6).sqrt() / 600
            opt.zero_grad(set_to_none=True); scaler.scale(l).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()


def read(net, sp, days):
    net.eval(); r2 = np.random.default_rng(99); sv = sq = n = 0.0
    with torch.no_grad():
        for p, v in days.items():
            for i in v:
                x, tv, tq, m, vf, cl = sample(p, sp, int(i), r2); ten = lambda a: torch.from_numpy(a).to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    o = net(pad8(ten(x)[None]))[..., :x.shape[-1]].float()[0]
                e = ten(m); ev = (o[0] - ten(tv))[e] * ten(vf).expand_as(ten(tv))[e]; eq = (o[1] - ten(tq))[e] * ten(cl).expand_as(ten(tq))[e]; sv += float(ev.square().sum()); sq += float(eq.square().sum()); n += float(e.sum())
    net.train(); rv, rq = (sv / max(n, 1)) ** 0.5, (sq / max(n, 1)) ** 0.5; return rv, rq, 0.54 * max(0, 1 - rv / 25) + 0.46 * max(0, 1 - rq / 600)


for sp, mo in (("validation", "03"), ("private", "04")):
    nd = D[PANELS[0]][sp]["V"].shape[0] // 288; od = {p: [i for i in range(nd) if i % 2 == 0] for p in PANELS}; ev_ = {p: [i for i in range(nd) if i % 2 == 1] for p in PANELS}; net = fresh(); print(f"seed {SEED} {mo} stage A base", [round(x, 4) for x in read(net, sp, ev_)], flush=True); done = 0
    for e_ in EPS:
        train(net, sp, od, e_ - done); done = e_; print(f"seed {SEED} {mo} stage A after {e_} epochs on odd days: even days", [round(x, 4) for x in read(net, sp, ev_)], f"{(time.time() - T0) / 60:.0f} min", flush=True)
    for e_ in (8, 16, 32):
        net = fresh(); train(net, sp, {p: list(range(nd)) for p in PANELS}, e_); net.eval(); out = {}
        with torch.no_grad():
            for p in PANELS:
                d = D[p]; m_ = d[sp]; T_, n_ = m_["V"].shape; VU = np.zeros((T_, n_), np.float32); QU = VU.copy()
                for i in range(T_ // 288):
                    a, b = i * 288, (i + 1) * 288; x, fv, fq = image(m_["V"][a:b], m_["Q"][a:b], m_["f"][a:b], d["stat"], wk(m_["days"][i]))
                    with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                        o = net(pad8(torch.from_numpy(x)[None].to(dev)))[..., :n_].float()[0].cpu().numpy()
                    VU[a:b] = (o[0] + fv) * d["vf"]; QU[a:b] = (o[1] + fq) * d["cap"]
                out[p + "_V"] = VU; out[p + "_Q"] = QU
        np.savez_compressed(f"{OUT}ft_s{SEED}_{mo}_e{e_}.npz", **out); print(f"seed {SEED} {mo} stage B {e_} epochs saved, {(time.time() - T0) / 60:.0f} min", flush=True)
