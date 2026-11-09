"""G166 (11/01, strategy's ruling after #116): the T2 bounds of Section 6.4 and the loss split of Fig. 6 on M3 dev raked to March's official composition (m3v2.weights, target validation), beside the unweighted
values. Onset conditional = the mean of IoU on the no-precursor windows of J11 and 1 elsewhere (G117); ongoing oracle attribution = 1 - (i), and 1 - (i) - (iii) new jams as onsets (G121). Loss parts of G127 and G121.
usage: TFB_M3VER=v3 py -B -X utf8 g166_raked_bounds.py"""
import sys
sys.dont_write_bytecode = True; sys.path.insert(0, r"C:/Users/user/tfb/src"); sys.argv = [sys.argv[0], "0"]
import numpy as np, pandas as pd
import m3v2

W = "C:/Users/user/tfb/work/t2/"; O = pd.read_csv(W + "g117_onset_oracle.csv"); O["cond"] = "queue_onset"; F = pd.read_csv(W + "g155_onset_b_m3.csv")[["w", "hour"]]; O = O.merge(F, on="w"); O = O[O.nY > 0].reset_index(drop=True)
O["np"] = (O.v_ff >= 0.98) & O.flow_ratio.between(0.85, 1.15) & (O.fmax <= 0.1); w = m3v2.weights(O, "validation")[0]; v = np.where(O.np, O.iou, 1.0)
wm = lambda a, w_: float((a * w_).sum() / w_.sum())
print(f"onset windows {len(O)}, no precursor {int(O.np.sum())} | level unweighted {O.iou.mean():.4f} raked {wm(O.iou.to_numpy(), w):.4f} | conditional unweighted {v.mean():.4f} raked {wm(v, w):.4f} | no-precursor share raked {wm(O.np.to_numpy(float), w):.3f}")
print(f"  loss unweighted {1 - O.iou.mean():.4f}: no precursor {((1 - O.iou) * O.np).mean():.4f}, others {((1 - O.iou) * ~O.np).mean():.4f} | raked {1 - wm(O.iou.to_numpy(), w):.4f}: no precursor {wm(((1 - O.iou) * O.np).to_numpy(), w):.4f}, others {wm(((1 - O.iou) * ~O.np).to_numpy(), w):.4f}")
P = pd.read_csv(W + "g127_onset_loss_split.csv"); print("g127 columns", P.columns.tolist()[:24])
D = pd.read_csv(W + "g121_ongoing_oracle.csv"); print("g121 columns", D.columns.tolist()[:30], len(D))
M = pd.read_csv(W + "g156_m3_frame.csv")[["w", "cond", "panel", "day", "hour", "A"]]; D = M.merge(D, on="w", suffixes=("", "_g")); wg = m3v2.weights(D, "validation")[0]
print(D.filter(regex="^(i|ii|iii|loss|end|ext|place)").mean().round(4).to_dict())
D.to_pickle("C:/Users/user/AppData/Local/Temp/claude/C--Users-user-OneDrive-Desktop/c4594c81-2bd2-407c-8c14-87045d5cd22a/scratchpad/g121m.pkl"); np.save("C:/Users/user/AppData/Local/Temp/claude/C--Users-user-OneDrive-Desktop/c4594c81-2bd2-407c-8c14-87045d5cd22a/scratchpad/g121w.npy", wg)
