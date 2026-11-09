"""G217 (11/02, ledger #184, stage 1a): does the forecast flow against q* add to the ongoing probabilities on links queued at T? G172's design: logistic y ~ logit(p) + h against + r x horizon group + (r > 1) x horizon group, r = the demand forecast at T + h (dem10_c, dem20_c, dem30_c of the inputs, in units of capacity) x cap / q*, five folds by window, log loss per cell, window bootstrap. Cells of links queued at T - 5 (the row at T where seen). Line: improvement >= 2 SE with the same sign on M3 dev and on January + February -> lgbnm with the six columns. Hold-out days are not read.
usage: TFB_M3VER=v3 py -B -X utf8 g217_qstar_gate.py"""
import sys
s = open(r"C:/Users/user/tfb/review/g172_t_row.py", encoding="utf-8").read()


def sub(a, b, n=1):
    global s
    assert s.count(a) == n, (a[:60], s.count(a))
    s = s.replace(a, b)


sub('def cells(w, inst, p, y, e, row, vcut, rows):\n', 'IDM = None\n\n\ndef rr_of(Xw, P):\n    global IDM\n    if IDM is None:\n        n_ = [str(x) for x in np.load("C:/tfb_data/tfb_nn/nn_data_afdem/names.npy", allow_pickle=True)]; IDM = [n_.index(c) for c in ("dem10_c", "dem20_c", "dem30_c")]\n    qs = P.w * P.kj / (1 + P.w / (0.6 * P.vf)); D_ = np.asarray(Xw[:, IDM], float); return np.nan_to_num(D_) * (P.cap / qs)[:, None]\n\n\ndef cells(w, inst, p, y, e, row, vcut, rows, RR):\n')
sub('qd=sh(q, -1)[jj], fd=sh(fr, -1)[jj])', 'qd=sh(q, -1)[jj], fd=sh(fr, -1)[jj], r=RR[jj, np.minimum(ii // 2, 2)].astype(np.float32))')
sub('cells(w, "2031-03", p, m["y"].T.astype(bool), m["m"].T.astype(bool), P.V[m["t"]], P.vcut, rows)', 'cells(w, "2031-03", p, m["y"].T.astype(bool), m["m"].T.astype(bool), P.V[m["t"]], P.vcut, rows, rr_of(Xw, P))')
sub('cells(k, mo, (OOF[0][k] + OOF[1][k]) / 2, lab.astype(bool), el.astype(bool), Vv[t], P.vcut, rows)', 'cells(k, mo, (OOF[0][k] + OOF[1][k]) / 2, lab.astype(bool), el.astype(bool), Vv[t], P.vcut, rows, rr_of(np.asarray(XN[OFFN[k]:OFFN[k + 1]]), P))')
sub('D.to_parquet(W + "g172_t_cells.parquet", index=False)', 'D = D[D.q0].reset_index(drop=True)')
sub('S = g[["q0", "f0", "qu", "fu", "qd", "fd"]].to_numpy(float); X += [S[:, :, None][:, :, 0] * G3[:, [a]] for a in range(3)]', 'r_ = g.r.to_numpy(float)[:, None]; S = np.hstack([np.clip(r_, 0, 2), (r_ > 1).astype(float)]); X += [S * G3[:, [a]] for a in range(3)]')
sys.argv = [sys.argv[0]]; exec(compile(s, "g172q", "exec"))
