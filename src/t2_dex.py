"""Decode variants for ledger #216 (11/04, the strategy's (N); B12 uses DEX on the ongoing windows under TFB_T2_OGDEX=1, #179 6):
DEX : the exact expected IoU of the top-k sets under independent cells (Monte Carlo, S draws, fixed seed) instead of best_set's
      E[I] / E[U] ratio; the same order of cells (by probability), only the choice of k changes
NZ  : the official windows are drawn on the condition that the truth is not empty (requires_queue_in_horizon): the onset row T+30
      rescaled p' = p / (1 - prod(1 - p)) (capped below 1)"""
import numpy as np


def DEX(pb, S=400, seed=0):
    pe = np.clip(np.asarray(pb, float).ravel(), 0.0, 1.0); order = np.argsort(-pe, kind="stable"); ps = pe[order]
    Y = np.random.default_rng(seed).random((S, len(ps))) < ps[None, :]; Ic = np.cumsum(Y, axis=1); T_ = Y.sum(axis=1, keepdims=True)
    U = np.arange(1, len(ps) + 1)[None, :] + T_ - Ic; iou = (Ic / U).mean(axis=0); kb = int(np.argmax(iou)) + 1
    sel = np.zeros(len(pe), bool); sel[order[:kb]] = True; return sel.reshape(np.shape(pb))


def NZ(pb):
    y = np.array(pb, float, copy=True); r = np.clip(y[-1], 0.0, 1 - 1e-6); z = 1 - np.prod(1 - r); y[-1] = np.clip(r / max(z, 1e-6), 0.0, 1 - 1e-6); return y
