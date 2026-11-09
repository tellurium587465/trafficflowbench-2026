"""Build the release-asset archive of the final submissions (10/24): everything the reproduction needs that is not code and
not the public release, in the layout the entry scripts expect, with a manifest (size, sha256 per file).
  nn/    -> TFB_NN   : the 100 final networks (q24f 8 seeds, e4f 12 seeds, 5 configurations each), the official windows' features, the
                       feature names
  work/  -> TFB_WORK : the networks' saved test probabilities and X1 predictions, the LightGBM / ranker model files, the
                       training-month rarity table, the Task 1 models and the blank-cell model's window list, and the
                       submitted T1 / T4 components (for the verify stage)
The months' fine-tuning windows for X1 (nn_data_march / nn_data_april) are not shipped: twin/build_march.py rebuilds them
from the release. Usage after unpacking: TFB_NN=<dir>/nn TFB_WORK=<dir>/work TFB_REL=<release> (repro/README.md).
usage: python make_release_assets.py [out.zip]   (default C:/tfb_release/tfb_final_assets.zip)"""
import os, sys, json, glob, hashlib, zipfile

SRCROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(SRCROOT, "src"))
from common import NN, WORK

OUT = sys.argv[1] if len(sys.argv) > 1 else "C:/tfb_release/tfb_final_assets.zip"
C5 = ("attnglob", "attnwp05", "attn1", "drop1", "r13")
TAGS = [f"q24f_{c}_s{s}" for s in range(8) for c in C5] + [f"e4f_{c}_s{s}" for s in range(12) for c in C5]   # 10/24: final A = finA5 (E4 12 seeds)
files = []                                                       # (archive path, source path)
for t in TAGS:
    files.append((f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"))
    files += [(f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl"), (f"work/t2/probs_x1off__{t}.pkl", f"{WORK}/t2/probs_x1off__{t}.pkl")]
for t in [f"tfm4_s{s}" for s in range(5)] + [f"e4tfm4_s{s}" for s in range(4)]:        # 10/31: the attention backbone of A6 / B5 / B6 (March tfm4, April e4tfm4)
    files += [(f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"), (f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl")]
for t in [f"q24t_{c}_s{s}" for s in range(8) for c in C5]:                                # 11/02 ledger #164: the ongoing networks with the row at T, same-day X1
    files += [(f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"), (f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl"), (f"work/t2/probs_x1offsd__{t}.pkl", f"{WORK}/t2/probs_x1offsd__{t}.pkl")]
for t in [f"q24f_{c}_s{s}" for s in range(8) for c in C5]:                                # 11/01 ledger #125: same-day X1 of A8 (11/03 #189: A13's, April on origins every 3 rows)
    files += [(f"work/t2/probs_x1offsd__{t}.pkl", f"{WORK}/t2/probs_x1offsd__{t}.pkl"), (f"work/t2/probs_x1offsdd3__{t}.pkl", f"{WORK}/t2/probs_x1offsdd3__{t}.pkl")]
for t in [f"q24f_{c}_s{s}" for s in range(8, 16) for c in C5]:                            # 11/04 ledgers #107, #221: B11 / B12's April ongoing over seeds 0-15
    files += [(f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"), (f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl")]
    files.append((f"work/t2/probs_x1offsdd3__{t}.pkl", f"{WORK}/t2/probs_x1offsdd3__{t}.pkl"))   # 11/05 ledger #230: A15 / A16's April dense same-day X1 for seeds 8-15
for t in [f"q24m_{c}_s{s}" for s in range(8) for c in C5]:                                # 11/06 ledger #252: A22 / A22c's April networks (training months + March) and their dense same-day X1
    files += [(f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"), (f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl"), (f"work/t2/probs_x1offsdd3__{t}.pkl", f"{WORK}/t2/probs_x1offsdd3__{t}.pkl")]
for t in ("x3af_s0", "x3af_s1"):                                                          # 11/01 ledger #125: the raw-history networks
    files += [(f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"), (f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl")]
for t in ("u2d_s0", "u2d_s1", "pcL_s0", "pcL_s1", "pcL_s2", "pcL_s3"):   # 11/05 ledger #232: pcL s2, s3 for A16                                        # 11/03 ledger #201: the 2D U-Net (A12, B10) and the rank-input network (A12)
    files += [(f"nn/models/{t}.pt", f"{NN}/models/{t}.pt"), (f"work/t2/probs_{t}.pkl", f"{WORK}/t2/probs_{t}.pkl")]
files += [(f"nn/nn_test_afdemt/{os.path.basename(p)}", p) for p in sorted(glob.glob(f"{NN}/nn_test_afdemt/*.npy")) if not p.endswith("R.npy")]
files += [(f"work/t1/unet_ft/ft_s{s}_{m}_e32.npz", f"{WORK}/t1/unet_ft/ft_s{s}_{m}_e32.npz") for s in (4, 5) for m in ("03", "04")]   # 11/02 the image network: month-fine-tuned predictions
files += [(f"work/t1/unet_s{s}.pt", f"C:/tfb_data/kaggle_out/t1unet3/unet_s{s}.pt") for s in (4, 5)]
files.append(("work/t1/state_t1final.parquet", f"{WORK}/t1/state_t1final.parquet"))
files.append(("nn/nn_data_afdem/names.npy", f"{NN}/nn_data_afdem/names.npy"))
files += [(f"nn/nn_test_afdem/{os.path.basename(p)}", p) for p in sorted(glob.glob(f"{NN}/nn_test_afdem/*.npy")) if not p.endswith("R.npy")]
for f in ("lgbnm_queue_onset.txt", "lgbnm_queue_ongoing.txt", "physon_base.txt", "physon_m.txt", "rarity_table.pkl", "gapml_windows.npz"):
    files.append((f"work/t2/{f}", f"{WORK}/t2/{f}"))
for m in ("f3big", "f3all", "f3all2"):
    files += [(f"work/t1/speed_{m}.txt", f"{WORK}/t1/speed_{m}.txt"), (f"work/t1/flow_{m}.txt", f"{WORK}/t1/flow_{m}.txt")]
files += [("work/t1/bnd3_speed_final.txt", f"{WORK}/t1/bnd3_speed_final.txt"), ("work/t1/bnd3_flow_final.txt", f"{WORK}/t1/bnd3_flow_final.txt"),
          ("work/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet", f"{WORK}/t1/state_f3ens3c_ad2_gapml_clip_fd_bnd.parquet"),
          ("work/t4/odme_l2.parquet", f"{WORK}/t4/odme_l2.parquet")]
missing = [s for _, s in files if not os.path.exists(s)]
assert not missing, missing[:5]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


os.makedirs(os.path.dirname(OUT), exist_ok=True)
man = [dict(path=a, bytes=os.path.getsize(s), sha256=sha(s)) for a, s in files]
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for a, s in files:
        z.write(s, a)
    z.writestr("MANIFEST.json", json.dumps(dict(files=man, final=["finA13", "finA12", "finA11", "finB12", "finB11", "finB10", "finA10", "finA8", "finA7", "finB9s", "finB7"]), indent=1))
print(f"{OUT}: {len(files)} files, {sum(m['bytes'] for m in man) / 2**20:.0f} MB before compression, {os.path.getsize(OUT) / 2**20:.0f} MB zipped")
