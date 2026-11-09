#!/bin/bash
# 10/24: the README's setup in a brand-new virtual environment (Python 3.14, only repro/requirements_repro.txt from PyPI),
# then repro/clone_check.sh with that interpreter: a fresh clone + the asset zip -> both finals, sha256 against the
# submitted files. Downloads packages from PyPI (run only with the user's go-ahead).
set -euo pipefail; export MSYS_NO_PATHCONV=1
V=${V:-/c/tfb_venv_check}; REPO=${REPO:-/c/Users/user/tfb}
rm -rf "$V"; py -3.14 -m venv "$(cygpath -m "$V")"
"$V/Scripts/python.exe" -m pip install --quiet --upgrade pip
"$V/Scripts/python.exe" -m pip install --quiet -r "$(cygpath -m "$REPO")/repro/requirements_repro.txt"
"$V/Scripts/python.exe" -m pip freeze
PY="$V/Scripts/python.exe" D=/c/tfb_clone_venv bash "$REPO/repro/clone_check.sh"
