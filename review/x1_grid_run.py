"""G133 part 1 (10/31, ledger #88, strategy): the X1 fine-tune of one saved network under lr x epochs, M3 v3 (src/t2_m3v2_x1.py unchanged, 6 threads).
Restart-safe: a cell whose work/t2/probs_m3v3v2x1<arm>__<tag>.pkl exists is skipped. The base cell (2e-4, 4) is the saved file.
usage: TFB_M3VER=v3 py x1_grid_run.py <tag>"""
import os, sys, time, runpy
os.environ.setdefault("TFB_M3VER", "v3")
import torch
torch.set_num_threads(6); torch.set_num_threads = lambda n: None
R = r"C:/Users/user/tfb/"; tag = sys.argv[1]; sys.path.insert(0, R + "src")
for lr in ("1e-4", "2e-4", "4e-4"):
    for ep in (2, 4, 8):
        if lr == "2e-4" and ep == 4:
            continue
        arm = f"_g{lr.replace('-', '')}e{ep}"; f = R + f"work/t2/probs_m3v3v2x1{arm}__{tag}.pkl"
        if os.path.exists(f):
            print("skip", arm, flush=True); continue
        t0 = time.time(); sys.argv = ["t2_m3v2_x1.py", tag, lr, arm, str(ep)]
        runpy.run_path(R + "src/t2_m3v2_x1.py", run_name="__main__"); print("done", arm, f"{time.time() - t0:.0f}s", time.strftime("%H:%M"), flush=True)
print("ALLDONE", tag, flush=True)
