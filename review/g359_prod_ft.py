"""G359 (11/05, ledger #279b stage 2): fine-tune production ongoing networks on the V6 first-origin windows (#279b, NN/nn_data_v6first;
t2_nn_finetune.finetune as G351: lr 2e-4, 4 epochs, batch 32, wd 1e-3, the model's WPOW, feature drop 0.05, seed 0, no replay), save each
as models/q24fv6_<config>_s<seed>.pt (the original file's metadata, _mu / _sd / _colmask / _cfg, kept; the weights replaced), append its
config line to NN/queue_q24fv6.txt (WPOW for X1), then: base probabilities on the official windows (t2_nn_cpu_infer.py -> probs_<newtag>.pkl),
same-day X1 as A21's (March: X1_ARM=sd on nn_data_march; April: X1_ARM=sdd3 on nn_data_april_sp3 with X1_FILL=sd), X1_THREADS = THREADS.
One worker per call over its tags, restart-safe (.done files). usage: py -B -X utf8 g359_prod_ft.py <threads> <tag,tag,...>"""
import os, sys, time, subprocess, glob
THREADS = int(sys.argv[1]); TAGS = sys.argv[2].split(",")
sys.path.insert(0, r"C:/Users/user/tfb_a8/src"); SRC = r"C:/Users/user/tfb_a8/src"
import torch
import t2_nn_finetune as FT
from common import NN, WORK
torch.set_num_threads(THREADS); W = WORK + "/t2/"; T0 = time.time(); env = dict(os.environ, X1_THREADS=str(THREADS))
for tag in TAGS:
    new = tag.replace("q24f_", "q24fv6_", 1); done = W + f"g359_{new}.done"
    if os.path.exists(done):
        continue
    if not os.path.exists(f"{NN}/models/{new}.pt"):
        wp = FT.wpow(tag); net, mu, sd, cm, n = FT.finetune(tag, "nn_data_v6first", 1, 31, lr=2e-4, epochs=4, wpow=wp)
        d = torch.load(f"{NN}/models/{tag}.pt", map_location="cpu", weights_only=False); sdict = net.state_dict()
        assert set(sdict) <= set(d), (tag, set(sdict) - set(d))
        for k, v in sdict.items():
            d[k] = v.detach().clone()
        torch.save(d, f"{NN}/models/{new}.pt")
        line = next(l for f in sorted(glob.glob(f"{NN}/queue*.txt")) + [r"C:/Users/user/tfb_a8/repro/nn_configs/production_runs.txt"] for l in open(f, encoding="utf-8", errors="ignore") if l.split(",")[0].strip() == tag)
        with open(f"{NN}/queue_q24fv6.txt", "a", encoding="utf-8") as fh:
            fh.write(new + line.strip()[len(tag):] + "\n")
        print(f"{new}: fine-tuned on {n} windows (WPOW {wp}) {time.time() - T0:.0f}s", flush=True)
    if not os.path.exists(W + f"probs_{new}.pkl"):
        subprocess.run([sys.executable, "-B", "-X", "utf8", "t2_nn_cpu_infer.py", new, "nn_test_afdem", "q24fv6b"], cwd=SRC, check=True, env=env, stdout=subprocess.DEVNULL)
        os.replace(W + f"probs_q24fv6b__{new}.pkl", W + f"probs_{new}.pkl")
    if not os.path.exists(W + f"probs_x1offsd__{new}.pkl"):
        subprocess.run([sys.executable, "-B", "-X", "utf8", "t2_x1_sameday.py", new], cwd=SRC, check=True, env=dict(env, X1_SPLITS="validation", X1_ARM="sd"), stdout=subprocess.DEVNULL)
    if not os.path.exists(W + f"probs_x1offsdd3__{new}.pkl"):
        subprocess.run([sys.executable, "-B", "-X", "utf8", "t2_x1_sameday.py", new], cwd=SRC, check=True, env=dict(env, X1_SPLITS="private", X1_ARM="sdd3", X1_POOL_PRIV="nn_data_april_sp3", X1_FILL="sd"), stdout=subprocess.DEVNULL)
    open(done, "w").close(); print(f"{new}: base + X1 sd (March) + X1 sdd3 (April) done {time.time() - T0:.0f}s", flush=True)
