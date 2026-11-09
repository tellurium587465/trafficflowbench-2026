"""X1 (INSTRUCTIONS_WORKER X lane; user 10/22 "X1 を進めて"): fine-tune a saved FINAL Task 2 model on the same month's
COMPLETED earlier days, then predict that day's windows. CPU, same model / input encoding as t2_nn_cpu_infer.py and the
same loss as t2_nn2.py (BCE on eligible cells x window weight 100 * max(queued eligible cells, 3)^-WPOW, normalised by the
eligible cells of the batch; feature drop FDROP as in training). Training windows come from an nn_data_* directory in the
train layout (nn_data_march: March windows cut with the training-pool rules, EL = observed future cells only).
  finetune(tag, data_dir, day_lo, day_hi, lr, epochs) -> fine-tuned net (days are the 1-based day of month, inclusive)
  predict(net, mu, sd, cm, test_dir, window_ids)       -> {window_id: (6, n) probabilities}
Nothing at or after the predicted day is read: the training windows end (T+6) on a day before it (same-day windows,
build_march keeps T-12..T+6 inside one day)."""
import sys, time, numpy as np, torch
from common import NN, ROOT, WORK
import t2_nn_cpu_infer as CI

torch.set_num_threads(16)
NAMES = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]
ZERO = [NAMES.index(c) for c in ("prior_on", "prior_og")]
_DATA = {}


def wpow(tag):
    """the model's window-loss power WPOW from its training config line (moved here from t2_x1_official.py, 10/23)"""
    import glob
    line = None
    for f in sorted(glob.glob(f"{NN}/queue*.txt")) + [str(ROOT / "repro" / "nn_configs" / "production_runs.txt")]:
        for l in open(f, encoding="utf-8", errors="ignore"):
            if l.split(",")[0].strip() == tag:
                line = l.strip()
    kv = dict(x.split("=", 1) for x in line.split(",")[1:]) if line else {}
    return float(kv.get("WPOW", 1.0))


def data(d):
    if d not in _DATA:
        D = {f: np.load(f"{NN}/{d}/{f}.npy", allow_pickle=True, mmap_mode="r" if f in ("X", "Y", "EL") else None)
             for f in ("X", "Y", "EL", "off", "date")}
        D["day"] = np.array([int(str(x)[8:10]) for x in D["date"]]); _DATA[d] = D
    return _DATA[d]


def enc(v, cm, mu, sd, fdrop=0.0, gen=None):
    v = v.clone(); v[..., ZERO] = 0.0
    nm = ~torch.isfinite(v)
    if fdrop > 0:
        nm = nm | (torch.rand(nm.shape, generator=gen) < fdrop)
    v = torch.where(nm, torch.zeros_like(v), v); nm = nm & (cm > 0)
    x = torch.where(nm, torch.zeros_like(v), (((v - mu) / sd) * cm).clamp(-10, 10))
    return torch.cat([x, nm.float()], dim=-1)


def batch(items, cm, mu, sd, fdrop, gen):
    """items: list of (data dict, window index) -> encoded padded inputs, labels, eligibility, mask"""
    L = max(int(D["off"][k + 1] - D["off"][k]) for D, k in items)
    x = torch.full((len(items), L, len(NAMES)), float("nan")); y = torch.zeros((len(items), L, 6)); e = torch.zeros((len(items), L, 6))
    m = torch.zeros((len(items), L))
    for j, (D, k) in enumerate(items):
        a, b = int(D["off"][k]), int(D["off"][k + 1]); n = b - a
        x[j, :n] = torch.from_numpy(np.array(D["X"][a:b], dtype=np.float32))
        y[j, :n] = torch.from_numpy(np.array(D["Y"][a:b], dtype=np.float32))
        e[j, :n] = torch.from_numpy(np.array(D["EL"][a:b], dtype=np.float32)); m[j, :n] = 1.0
    return enc(x, cm, mu, sd, fdrop, gen) * m[:, :, None], y, e, m


def finetune(tag, data_dir, day_lo, day_hi, lr=2e-4, epochs=4, batch_size=32, wd=1e-3, wpow=1.0, fdrop=0.05, seed=0,
             replay_dir=None, replay_frac=0.5, exclude=None, month=None, max_n=None, l2sp=0.0):
    """replay_dir (10/22 X1'): each batch takes replay_frac of its windows at random from that train-layout directory
    (the training months), the rest from the month's earlier days; the earlier days are still seen `epochs` times.
    l2sp (10/23, the X1 onset rescue test): + l2sp * sum (theta - theta_saved)^2 (L2-SP, Li et al. 2018); default off."""
    net, mu, sd, cm = CI.load_model(tag); D = data(data_dir)
    ks = np.flatnonzero((D["day"] >= day_lo) & (D["day"] <= day_hi))
    if month is not None:                                       # V9 placebo: the same days of another month
        ks = ks[np.array([str(D["date"][k]).startswith(month) for k in ks], bool)]
    if exclude is not None:                                     # M3 v2: windows inside selector-simulated blanks
        ks = np.array([k for k in ks if k not in exclude], dtype=int)
    if max_n is not None and len(ks) > max_n:                   # V9 placebo: the same number of windows as the real month
        ks = np.sort(np.random.default_rng(seed + 7).choice(ks, max_n, replace=False))
    if len(ks) == 0:
        return net, mu, sd, cm, 0
    R = data(replay_dir) if replay_dir else None
    nb = batch_size - (int(round(batch_size * replay_frac)) if R is not None else 0)
    gen = torch.Generator().manual_seed(seed); rng = np.random.default_rng(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd); net.train()
    p0 = [p.detach().clone() for p in net.parameters()] if l2sp > 0 else None
    for ep in range(epochs):
        order = rng.permutation(ks)
        for b0 in range(0, len(order), nb):
            items = [(D, int(k)) for k in order[b0:b0 + nb]]
            if R is not None:
                items += [(R, int(k)) for k in rng.integers(0, len(R["off"]) - 1, batch_size - nb)]
            xb, y, e, m = batch(items, cm, mu, sd, fdrop, gen)
            logit = net(xb, m)[..., :6]
            l = torch.nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * e
            wb = 100.0 * torch.clamp((y * e).sum(dim=(1, 2)), min=3) ** (-wpow)
            loss = (l.sum(dim=(1, 2)) * wb).sum() / (e.sum() + 1)
            if p0 is not None:
                loss = loss + l2sp * sum(((p - q) ** 2).sum() for p, q in zip(net.parameters(), p0))
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    net.eval()
    return net, mu, sd, cm, len(ks)


def predict(net, mu, sd, cm, test_dir, wids):
    X = np.load(f"{NN}/{test_dir}/X.npy", mmap_mode="r"); off = np.load(f"{NN}/{test_dir}/off.npy")
    W = [str(w) for w in np.load(f"{NN}/{test_dir}/window_id.npy", allow_pickle=True)]; cond = np.load(f"{NN}/{test_dir}/cond.npy", allow_pickle=True)
    pos = {w: i for i, w in enumerate(W)}; out = {}
    with torch.no_grad():
        for w in wids:
            i = pos[w]; a, b = off[i], off[i + 1]
            v = torch.from_numpy(np.array(X[a:b], dtype=np.float32))[None]
            p = torch.sigmoid(net(enc(v, cm, mu, sd), torch.ones((1, b - a)))[..., :6]).numpy()[0].T.copy()
            if cond[i] == "queue_onset":
                p[:5] = 0.0
            out[w] = p
    return out


if __name__ == "__main__":                                       # timing / identity check
    tag = sys.argv[1] if len(sys.argv) > 1 else "e4s_a_s0"
    import pickle
    net, mu, sd, cm = CI.load_model(tag); W0 = [str(w) for w in np.load(f"{NN}/nn_test_m3/window_id.npy", allow_pickle=True)][:20]
    ref = pickle.load(open(f"{WORK}/t2/probs_m3__{tag}.pkl", "rb"))
    p = predict(net, mu, sd, cm, "nn_test_m3", W0)
    print("identity (no fine-tune) max |p - cpu_infer|:", max(float(np.abs(p[w] - ref[w]).max()) for w in W0))
    for lo, hi in ((5, 5), (3, 5)):
        t0 = time.time(); net2, *_, n = finetune(tag, "nn_data_march", lo, hi)
        print(f"fine-tune days {lo}-{hi}: {n} windows, {time.time() - t0:.1f}s")
