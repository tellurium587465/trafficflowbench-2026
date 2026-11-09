#!/bin/bash
# 10/31: a fresh clone + the asset zip -> A6, B6, B7 and B8, sha256 against the submitted files. Windows with the isolated venv (built 10/24 from requirements_repro.txt,
# no download), then Linux (WSL). Writes only under /c/tfb_clone_venv and ~/tfb_clone in WSL; log in repro/logs/recheck_1031b.txt.
L=/c/Users/user/tfb/repro/logs/recheck_1031b.txt; date > $L
PY=/c/tfb_venv_check/Scripts/python.exe D=/c/tfb_clone_venv ASSETS=/c/tfb_release/tfb_final_assets_A6B6.zip FINAL="A6 B6 B7 B8" bash /c/Users/user/tfb/repro/clone_check.sh >> $L 2>&1
echo "WINDOWS exit $? $(date)" >> $L
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/user/tfb/repro/recheck_wsl.sh >> $L 2>&1
echo "LINUX exit $? $(date)" >> $L
echo ALLDONE >> $L
