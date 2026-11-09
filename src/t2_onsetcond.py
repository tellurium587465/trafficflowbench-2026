"""11/05 ledger #287: the onset probabilities conditioned on the official window holding a queue in its horizon. An official onset window
has no visible queue in its history and a queue somewhere in T+5 .. T+30 (repo config: requires_queue_in_horizon); with the first-origin
selection its queue appears at T+30 (39 of the 40 official June onset windows). condition(pb, window_id) -> (pb', kept): NS = 2000 joint
samples of the T+30 row of pb (the probabilities best_set sees): each run of cells with p > 0.1 one uniform per sample, other cells
independent; kept = the samples with at least one queued cell; the T+30 row of pb' = the kept samples' marginals (the other rows unchanged;
fewer than 10 kept -> pb unchanged). Deterministic: seeded by crc32(window_id). Reads nothing but pb; no constant is fitted."""
import zlib, numpy as np

NS = 2000


def condition(pb, window_id, ns=NS):
    rng = np.random.default_rng(zlib.crc32(str(window_id).encode())); r = pb[-1]; n = r.size; U = rng.random((ns, n)); m = r > 0.1; j = 0
    while j < n:
        if m[j]:
            e_ = j
            while e_ + 1 < n and m[e_ + 1]:
                e_ += 1
            U[:, j:e_ + 1] = rng.random((ns, 1)); j = e_ + 1
        else:
            j += 1
    S = U < r[None]; keep = S.any(axis=1); k = int(keep.sum())
    if k < 10:
        return pb, k
    out = pb.copy(); out[-1] = S[keep].mean(axis=0); return out, k
