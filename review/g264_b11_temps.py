"""G264 (11/04, ledger #225): B11's flow on M3 v3 dev (G212's frame, b11 = B10's flow with q24f seeds 0-15, raw best_set) at the
temperature 1.0 / 1.2 (production) / 1.4 before best_set, per window; with G263's A-side columns and spread this is the M3 part of #225.
Dev only, writes g264_dev.csv. usage: TFB_X3=x3af TFB_M3VER=v3 py -B -X utf8 g264_b11_temps.py dev"""
import sys
assert sys.argv[1:] == ["dev"], "dev only"
R_ = r"C:/Users/user/tfb/review/"; END = 'exec(compile(s, "g175a", "exec"))'
src = open(R_ + "g212_a10_read.py", encoding="utf-8").read(); assert src.count(END) == 1
exec(compile(src.replace(END, ""), "g212", "exec"))
sub('ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; WS = sum(w for _, w in ON)',
    'ON = [(f"q24f_{c}_s{s}", w) for s in range(8) for c, w in CF5]; WS = sum(w for _, w in ON); ON16 = [(f"q24f_{c}_s{s}", w) for s in range(16) for c, w in CF5]; WS16 = sum(w for _, w in ON16)')
sub('PQ = {t: pickle.load(open(W + f"probs_m3v3__{t}.pkl", "rb")) for t, _ in ON}', 'PQ = {t: pickle.load(open(W + f"probs_m3v3__{t}.pkl", "rb")) for t, _ in ON16}')
IN = '0.8 * ((1 - X3) * pq16 + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2'
sub('sc["b10"] += pr * G.iou(raw(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)), m);',
    'sc["b10"] += pr * G.iou(raw(tmp(0.8 * ((1 - X3) * pq + X3 * px) + 0.2 * (PU[0][w] + PU[1][w]) / 2)), m); pq16 = sum(wt * PQ[t_][w] for t_, wt in ON16) / WS16; '
    'sc["b11"] += pr * G.iou(raw(tmp(' + IN + ')), m); sc["b11t10"] += pr * G.iou(raw(' + IN + '), m); sc["b11t14"] += pr * G.iou(raw(sg(lg(' + IN + ') / 1.4)), m);')
sub('"a12", "b10"); cols', '"a12", "b10", "b11", "b11t10", "b11t14"); cols')
sub('D.to_csv(W + "g212_dev.csv"', 'D.to_csv(W + "g264_dev.csv"')
sub('    Df = train_B(); Df.to_csv(W + "g212_train.csv", index=False)\n', '    sys.exit()\n')
exec(compile(s, "g175u", "exec"))
