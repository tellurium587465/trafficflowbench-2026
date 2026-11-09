"""Row T for the ongoing networks q24t (11/02, ledger #164; 745700, 742068: data at or before T may be read). The columns v_T and q_T of the feature matrix are empty under NO_T; here they take the row at the window's origin, as the other history columns (v / vf, q / cap): the Task 1 masked layer on the train split, the release on the test months. Every other column stays. An nn_data_* or nn_test_* directory is copied with X filled.
usage: python t2_trow.py <split: validation|private> <src dir under NN> <dst dir under NN>   (the directory must carry panel.npy and t.npy)"""
import os, sys, shutil, numpy as np
os.environ["TFB_T2_NOT"] = "1"; os.environ["TFB_T2_ALLFIN"] = "1"
from cachelib import load
from common import NN
from t2_model import PanelInputs


def fill_dir(src, dst, split):
    d, o = f"{NN}/{src}/", f"{NN}/{dst}/"; os.makedirs(o, exist_ok=True)
    for f in os.listdir(d):
        if f != "X.npy":
            shutil.copy(d + f, o + f)
    names = [str(x) for x in np.load(f"{NN}/nn_data_afdem/names.npy", allow_pickle=True)]; iv, iq = names.index("v_T"), names.index("q_T")
    X = np.array(np.load(d + "X.npy", mmap_mode="r")); off = np.load(d + "off.npy"); pan = [str(x) for x in np.load(d + "panel.npy", allow_pickle=True)]; t_ = np.load(d + "t.npy"); C = {}
    for k in range(len(off) - 1):
        p = pan[k]
        if p not in C:
            P = load(p, split); PI = PanelInputs(P); C[p] = (PI.mask_v / P.vf[None, :], PI.mask_q / P.cap[None, :])
        a, b = int(off[k]), int(off[k + 1]); X[a:b, iv] = C[p][0][int(t_[k])]; X[a:b, iq] = C[p][1][int(t_[k])]
    np.save(o + "X.npy", X)


if __name__ == "__main__":
    fill_dir(sys.argv[2], sys.argv[3], sys.argv[1])
