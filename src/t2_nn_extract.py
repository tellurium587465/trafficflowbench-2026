"""Stream the members of nn_data.npz / nn_test.npz to E:/tfb_nn/<name>/<key>.npy (no RAM copy).

Plain .npy files load with one large read (npz members are read in small zip chunks, ~30 MB/s over WSL 9P),
and E: keeps the reads off the C: SSD.
"""
import os, shutil, sys, time, zipfile
from common import NN                                        # 10/22: data root off the failing E: (TFB_NN)

for name in sys.argv[1:] or ["nn_data", "nn_test"]:
    t = time.time(); out = f"{NN}/{name}"; os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(f"C:/Users/user/tfb/work/t2/{name}.npz") as z:
        for m in z.namelist():
            with z.open(m) as src, open(f"{out}/{m}.part", "wb") as dst:
                shutil.copyfileobj(src, dst, 64 << 20)
            os.replace(f"{out}/{m}.part", f"{out}/{m}")
    print(name, sorted(os.listdir(out)), f"{time.time()-t:.0f}s", flush=True)
