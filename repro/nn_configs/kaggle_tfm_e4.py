# TFB lane 9 (10/29), E4 type: the attention backbone trained like the E4 family (training months + March + the C-world
# synthetic ongoing windows, nn_data_afdemMsimC24, DOM=1 SIMW=0.5 EMA=0.998), 4 seeds, for A's April windows.
# Earlier header (v1/v2 kernel tfb-t2-tfm1): Task 2 network on Kaggle GPU. Platform check (the production q24f_attnglob line, retrained here)
# and the attention-backbone variant (2 dilated conv blocks for local order + 4 transformer layers), FINAL on the
# training months (nn_data_afdem), probabilities for the official test windows; weights saved for CPU inference on M3.
# Private: data and code from the private datasets telluriumrrr/tfb-nn-afdem and telluriumrrr/tfb-nn-code.
import os, glob, time, subprocess, shutil, numpy as np, torch

t0 = time.time()
def find(fname):
    hits = [p for p in glob.glob("/kaggle/input/**/" + fname, recursive=True)]
    assert hits, fname
    return os.path.dirname(hits[0])

data = find("Y.npy"); code = find("t2_nn2.py")
print("data", data, "| code", code, flush=True)
root = "/tmp/tfb/"; os.makedirs(root + "tfb_nn/nn_test_afdem", exist_ok=True)
if not os.path.exists(root + "tfb_nn/nn_data_afdemMsimC24"):
    os.symlink(data, root + "tfb_nn/nn_data_afdemMsimC24")
for f in ("X", "cond", "off", "window_id"):
    dst = f"{root}tfb_nn/nn_test_afdem/{f}.npy"
    if not os.path.exists(dst):
        os.symlink(f"{code}/test_{f}.npy", dst)
for d in ("/kaggle/working/out", "/kaggle/working/models"):
    os.makedirs(d, exist_ok=True)

t = time.time(); X = np.load(data + "/X.npy", mmap_mode="r"); s = X.nbytes / 1e9; _ = float(np.asarray(X[::4096]).sum()); del X
print(f"read X {s:.2f} GB in {time.time() - t:.0f}s ({s / max(time.time() - t, 1e-9) * 1000:.0f} MB/s)", flush=True)

names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPUs", names, flush=True)
amp = "bf16" if any(k in names[0] for k in ("L4", "A100", "H100", "A10")) else "fp16"
ZERO = "pos+tod+dow+capl+lanes+length+vf+w"
BASE = f"H=128,LR=0.002,EP=24,WD=0.001,K=5,FDROP=0.05,B=256,COMPILE=0,DROP=0,BUCKET=0,ZERO={ZERO},WPOW=1,FINAL=1"
TFM = f"{BASE},DEPTH=2,ATTN=4,HEADS=4,DOM=1,SIMW=0.5,EMA=0.998"                 # attention backbone, E4 training (as the e4f lines)
JOBS = [[f"e4tfm4_s2,{TFM},SEED=2"], [f"e4tfm4_s3,{TFM},SEED=3"]]          # v2 (10/29 12:50): v1 gave s0 / s1; the 2nd config of a
                                                                               # process ran out of GPU memory -> one seed per process
env0 = dict(os.environ, TFB_ROOT_E=root, TFB_OUT="/kaggle/working/out/", TFB_MODELS="/kaggle/working/models",
            NN_DATA="nn_data_afdemMsimC24", NN_TEST="nn_test_afdem", NN_AMP=amp, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONUNBUFFERED="1")   # v2: no MMAP (chunked mmap reads of /kaggle/input took 41-45 min in v1; one np.load ~1 min)
print("amp", amp, "| jobs", [j[0].split(",")[0] for j in JOBS], flush=True)
procs = []
if len(names) >= 2:                                    # two GPUs: one job each, in parallel
    for g, job in enumerate(JOBS):
        env = dict(env0, CUDA_VISIBLE_DEVICES=str(g), NN_CONFIGS=";".join(job))
        procs.append(subprocess.Popen(["python", "-u", f"{code}/t2_nn2.py", "daemon", "0"], env=env,
                                      stdout=open(f"/kaggle/working/log_gpu{g}.txt", "w"), stderr=subprocess.STDOUT))
        time.sleep(180)                                 # stagger the two host-side loads of X (8.4 GB each)
    for p in procs:
        p.wait()
else:
    env = dict(env0, NN_CONFIGS=";".join(sum(JOBS, [])))
    subprocess.run(["python", "-u", f"{code}/t2_nn2.py", "daemon", "0"], env=env,
                   stdout=open("/kaggle/working/log_gpu0.txt", "w"), stderr=subprocess.STDOUT)
for f in sorted(glob.glob("/kaggle/working/log_gpu*.txt")):
    print("====", f); print("".join(open(f).readlines()[-15:]), flush=True)
print("outputs", os.listdir("/kaggle/working/out"), os.listdir("/kaggle/working/models"), f"total {time.time() - t0:.0f}s", flush=True)
