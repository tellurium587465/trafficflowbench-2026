"""E4 production queue (10/22): the five production configurations (q24f lines of production_runs.txt) x seeds, FINAL,
with extra key=values (e.g. DOM=1,SIMW=0.5,MW=4), tags <prefix>_<cfg>_s<seed>; plus the daemon script for the dataset.
usage: py make_e4_queue.py <nn_data name> <tag prefix> <extra kv or -> [seeds=0,1,2,3]"""
import sys, re
data, pre, extra = sys.argv[1], sys.argv[2], sys.argv[3]; seeds = sys.argv[4].split(",") if len(sys.argv) > 4 else "0,1,2,3".split(",")
R = "C:/Users/user/tfb/repro/nn_configs/"
lines = {l.split(",")[0]: l.strip() for l in open(R + "production_runs.txt") if l.startswith("q24f_")}
out = []
for s in seeds:
    for c in ("attnglob", "attnwp05", "attn1", "drop1", "r13"):
        l = lines[f"q24f_{c}_s0"].split(",", 1)[1].replace("SEED=0", f"SEED={s}")
        if extra != "-":
            l = l.replace(",FINAL=1", f",{extra},FINAL=1")
        out.append(f"{pre}_{c}_s{s},{l}")
q = f"C:/tfb_data/tfb_nn/queue_{pre}.txt"; open(q, "w", newline="\n").write("\n".join(out + ["STOP"]) + "\n")
sh = open(R + "wsl_nnd_afdem.sh").read().replace("NN_DATA=nn_data_afdem ", f"NN_DATA={data} ")
assert f"NN_DATA={data} " in sh; open(R + f"wsl_nnd_{data[8:]}.sh", "w", newline="\n").write(sh)
print(q, len(out), "runs |", R + f"wsl_nnd_{data[8:]}.sh"); print(out[0])
