#!/bin/bash
# 10/24: the distribution exactly as README.md describes it — a fresh `git clone` of the repository + the release-asset zip
# (make_release_assets.py) unpacked next to it, the public release in place, then the single entry script
# `bash repro/run_final.sh` for both finals (caches from the release -> t2aux -> t2final -> verify: sha256 of each rebuilt
# submission.csv against the submitted one). No package download: the current Python environment.
# ASSETS may list several zips separated by spaces (11/06: tfb_final_assets_a22_nn.zip and tfb_final_assets_a22_work.zip).
#   Windows (Git Bash):  PY=py bash repro/clone_check.sh
#   Linux / WSL:         PY=~/arc2venv/bin/python REPO=/mnt/c/Users/user/tfb ASSETS=/mnt/c/tfb_release/tfb_final_assets.zip \
#                        D=$HOME/tfb_clone TFB_REL=/mnt/c/Users/user/tfb/data/kaggle_public bash repro/clone_check.sh
set -euo pipefail; export MSYS_NO_PATHCONV=1
win() { (cd "$1" && (pwd -W 2>/dev/null || pwd)); }                # a path Python can open (Windows form under Git Bash)
REPO=${REPO:-/c/Users/user/tfb}; ASSETS=${ASSETS:-/c/tfb_release/tfb_final_assets.zip}; D=${D:-/c/tfb_clone}; PY=${PY:-py}
rm -rf "$D"; mkdir -p "$D"
git clone -q "$(win "$REPO")" "$(win "$D")/tfb"
for A_ in $ASSETS; do                                              # 11/06: one or more asset zips (the release splits them under 2 GiB each)
  AW="$(win "$(dirname "$A_")")/$(basename "$A_")"
  (cd "$D" && "$PY" -c "import zipfile; zipfile.ZipFile(r'$AW').extractall('assets')")
done
export TFB_NN="$(win "$D")/assets/nn" TFB_WORK="$(win "$D")/assets/work" TFB_CACHE="$(win "$D")/assets/work/cache" TFB_REL=${TFB_REL:-$(win "$REPO")/data/kaggle_public}
"$PY" -c "import sys, numpy, pandas, lightgbm; print('python', sys.version.split()[0], sys.platform, '| numpy', numpy.__version__, '| pandas', pandas.__version__, '| lightgbm', lightgbm.__version__)"
t0=$(date +%s)
PY="$PY" FINAL="${FINAL:-A B}" bash "$D/tfb/repro/run_final.sh" 2>&1 | grep -v "^Warning"
echo "clone rebuild: $(( $(date +%s) - t0 )) s"
