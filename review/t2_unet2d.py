"""#133 (11/01, strategy T2_DEEP_REVIEW 15): a Task 2 network of another family, a 2D U-Net on (history row, link): rows T-12 .. T-1 of the raw history (R.npy: v / vcut, q / cap, observed) and a queue flag, convolutions in time and along the corridor, no engineered features. Output: six logits per link.
Loss as t2_nn2.py: BCE on eligible cells x window weight 100 max(queued eligible cells, 3) ** -1, normalised by the eligible cells of the batch. fold: trained on the months before 2030-12, out-of-fold on 2030-12 .. 2031-02, and M3 v3 windows; final: all nine months, M3 and the official windows.
Runs on the local RX 7800 XT (ROCm under WSL, NCHW, fp16) or on CUDA. C: only. Paired reads stay on one machine.
usage: python t2_unet2d.py <seed> <epochs> <fold|final>   (env U2D_DRY=1: 30 steps and a timing, no inference)"""
import os, sys, time, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
SEED, EP, MODE = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; DRY = os.environ.get("U2D_DRY", "") == "1"; BS = int(os.environ.get("U2D_BS", "48")); B0 = int(os.environ.get("U2D_B", "64")); NN = os.environ.get("U2D_NN", "/mnt/c/tfb_data/tfb_nn/"); OUT = os.environ.get("U2D_OUT", NN + "unet2d/"); os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); torch.manual_seed(SEED); rng = np.random.default_rng(SEED); T0 = time.time(); D_ = NN + "nn_data_afdem/"
R = np.load(D_ + "R.npy", mmap_mode="r"); off = np.load(D_ + "off.npy"); Y = np.load(D_ + "Y.npy", mmap_mode="r"); EL = np.load(D_ + "EL.npy", mmap_mode="r"); date = np.load(D_ + "date.npy", allow_pickle=True); cond = np.load(D_ + "cond.npy", allow_pickle=True); LEN = np.diff(off); MON = np.array([str(d)[:7] for d in date])
print("rows", R.shape, Y.shape, "windows", len(LEN), f"{time.time() - T0:.0f}s", flush=True)
Rg = torch.from_numpy(np.array(R[:, :12], dtype=np.float16)).to(dev); Yg = torch.from_numpy(np.array(Y, dtype=np.uint8)).to(dev); Eg = torch.from_numpy(np.array(EL, dtype=np.uint8)).to(dev); Og = torch.from_numpy(off[:-1].astype(np.int64)).to(dev); print("on the device", f"{time.time() - T0:.0f}s", flush=True)


class Res(nn.Module):
    def __init__(s, c, dl=1):
        super().__init__(); s.a = nn.Conv2d(c, c, 3, padding=(1, dl), dilation=(1, dl)); s.b = nn.Conv2d(c, c, 3, padding=(1, dl), dilation=(1, dl)); s.n1 = nn.GroupNorm(8, c); s.n2 = nn.GroupNorm(8, c)

    def forward(s, x):
        return x + s.b(Fn.gelu(s.n2(s.a(Fn.gelu(s.n1(x))))))


class Net(nn.Module):
    def __init__(s, b):
        super().__init__(); s.inp = nn.Conv2d(4, b, 3, padding=1); s.e1 = nn.Sequential(Res(b), Res(b)); s.d1 = nn.Conv2d(b, 2 * b, 3, stride=(1, 2), padding=1); s.e2 = nn.Sequential(Res(2 * b), Res(2 * b)); s.d2 = nn.Conv2d(2 * b, 4 * b, 3, stride=(1, 2), padding=1)
        s.mid = nn.Sequential(Res(4 * b, 1), Res(4 * b, 2), Res(4 * b, 4), Res(4 * b, 8)); s.u2 = nn.ConvTranspose2d(4 * b, 2 * b, (1, 2), stride=(1, 2)); s.c2 = nn.Sequential(nn.Conv2d(4 * b, 2 * b, 1), Res(2 * b)); s.u1 = nn.ConvTranspose2d(2 * b, b, (1, 2), stride=(1, 2)); s.c1 = nn.Sequential(nn.Conv2d(2 * b, b, 1), Res(b)); s.h = nn.Sequential(nn.Conv1d(2 * b, 2 * b, 3, padding=1), nn.GELU(), nn.Conv1d(2 * b, 6, 1)); nn.init.constant_(s.h[-1].bias, -3.0)

    def forward(s, x):
        z1 = s.e1(s.inp(x)); z2 = s.e2(s.d1(z1)); z = s.mid(s.d2(z2)); z = s.c2(torch.cat([s.u2(z), z2], 1)); z = s.c1(torch.cat([s.u1(z), z1], 1)); return s.h(torch.cat([z[:, :, -1], z.mean(2)], 1))


def inp(r):
    """r: (B, L, 12, 3) float -> (B, 4, 12, L + pad to a multiple of 4)"""
    v, q, o = r[..., 0], r[..., 1], r[..., 2]; x = torch.stack([v, q, o, ((v <= 1) & (o > 0)).float()], 1).permute(0, 1, 3, 2); p = (4 - x.shape[-1] % 4) % 4; return (Fn.pad(x, (0, p, 0, 0), mode="replicate") if p else x)


tr = np.flatnonzero(MON < "2030-12") if MODE == "fold" else np.arange(len(LEN)); by = {}
for k in tr:
    by.setdefault(int(LEN[k]), []).append(int(k))
print("train windows", len(tr), "widths", sorted((k, len(v)) for k, v in by.items()), flush=True)
net = Net(B0).to(dev); print("params", sum(x.numel() for x in net.parameters()), flush=True); nb = sum(len(v) // BS for v in by.values()); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-3); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1.5e-3, total_steps=max(EP * nb, 2), pct_start=0.1); scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda"); st = 0; t1 = time.time()
for ep in range(EP):
    bat = [(L, ks[j:j + BS]) for L, v in by.items() for ks in [rng.permutation(v)] for j in range(0, len(ks) - BS + 1, BS)]; bat = [bat[i] for i in rng.permutation(len(bat))]; run = 0.0
    for L, ks in bat:
        ks_t = torch.as_tensor(ks, device=dev); ix = Og[ks_t][:, None] + torch.arange(L, device=dev)[None, :]; r = Rg[ix].float(); y = Yg[ix].float(); e = Eg[ix].float(); wb = 100.0 * ((y * e).sum(dim=(1, 2)).clamp(min=3)) ** -1.0
        with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
            lo = net(inp(r))[..., :L].permute(0, 2, 1)
        l = Fn.binary_cross_entropy_with_logits(lo.float(), y, reduction="none") * e; loss = (l.sum(dim=(1, 2)) * wb).sum() / e.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); run += float(loss.detach()); st += 1
        if DRY:
            print(f"step {st} width {L} {time.time() - t1:.1f}s", flush=True)
        if DRY and st >= 30:
            print(f"dry: {st} steps, {(time.time() - t1) / st:.3f} s per step, {(time.time() - t1) / st * nb:.0f} s per epoch, loss {run / st:.4f}, peak GPU {torch.cuda.max_memory_allocated() / 1e9 if dev.type == 'cuda' else 0:.1f} GB", flush=True); sys.exit()
    print(f"ep {ep + 1}/{EP} loss {run / len(bat):.4f} {(time.time() - t1) / (ep + 1):.0f} s per epoch {(time.time() - T0) / 60:.0f} min", flush=True); torch.save({k: v.cpu() for k, v in net.state_dict().items()}, OUT + f"u2d_s{SEED}_{MODE}.pt")
net.eval()
if os.environ.get("U2D_NOINF") == "1":
    print("ALLDONE weights only", flush=True); sys.exit()


def infer(R_, off_, wid, cond_):
    out = {}
    with torch.no_grad():
        for i in range(len(wid)):
            a, b = int(off_[i]), int(off_[i + 1]); r = torch.from_numpy(np.array(R_[a:b, :12], dtype=np.float32)).to(dev)[None]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                p = torch.sigmoid(net(inp(r))[..., :b - a].float()).cpu().numpy()[0].copy()
            if cond_[i] == "queue_onset":
                p[:5] = 0.0
            out[wid[i]] = p
    return out


if MODE == "fold":
    ks = np.flatnonzero(MON >= "2030-12"); o = {}
    with torch.no_grad():
        for k in ks:
            k = int(k); a, b = int(off[k]), int(off[k + 1]); r = torch.from_numpy(np.array(R[a:b, :12], dtype=np.float32)).to(dev)[None]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
                p = torch.sigmoid(net(inp(r))[..., :b - a].float()).cpu().numpy()[0].copy()
            if cond[k] == "queue_onset":
                p[:5] = 0.0
            o[k] = p
    pickle.dump(o, open(OUT + f"oof_s{SEED}.pkl", "wb")); print("oof", len(o), flush=True)
d = NN + "nn_test_m3v3/"; pickle.dump(infer(np.load(d + "R.npy", mmap_mode="r"), np.load(d + "off.npy"), [str(w) for w in np.load(d + "window_id.npy", allow_pickle=True)], np.load(d + "cond.npy", allow_pickle=True)), open(OUT + f"m3_{MODE}_s{SEED}.pkl", "wb"))
if MODE == "final":
    d = NN + "nn_test_afdem/"; pickle.dump(infer(np.load(d + "R.npy", mmap_mode="r"), np.load(d + "off.npy"), [str(w) for w in np.load(d + "window_id.npy", allow_pickle=True)], np.load(d + "cond.npy", allow_pickle=True)), open(OUT + f"off_s{SEED}.pkl", "wb"))
print("ALLDONE", f"{(time.time() - T0) / 60:.0f} min", flush=True)
