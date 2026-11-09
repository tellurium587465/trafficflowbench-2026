"""Real + randomized twin worlds R0..R47 (every other synthetic window of nn_data_twinAR48t2 = every 4th pool window
of each world; all 48 worlds kept): GPU memory allows ~20M rows. -> E:/tfb_nn/nn_data_afdemsimR48q."""
import os, numpy as np
src = open(r"C:/Users/user/tfb/twin/build_simR.py", encoding="utf-8").read()
src = src.replace('S = "E:/tfb_nn/nn_data_twinAR"; D = "E:/tfb_nn/nn_data_afdemsimR"', 'S = "E:/tfb_nn/nn_data_twinAR48t2"; D = "E:/tfb_nn/nn_data_afdemsimR48q"')
src = src.replace('ks = np.flatnonzero(keep); print(', 'ks = np.flatnonzero(keep)[::2]; print(')
exec(compile(src, "build_simR48", "exec"))
