#!/bin/bash
# after a reboot: WSL + ROCm visible, a small forward pass of the Linear-only ops
source ~/rocm721venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 MIOPEN_USER_DB_PATH=/home/user/miopen721 MIOPEN_CUSTOM_CACHE_DIR=/home/user/miopen721
python - <<'EOF'
import torch, time
print(torch.__version__, "gpu", torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
x = torch.randn(256, 200, 128, device="cuda"); m = torch.nn.Sequential(torch.nn.Linear(128, 128), torch.nn.GELU(), torch.nn.Linear(128, 6)).to("cuda")
t = time.time(); y = m(x); y.sum().backward(); print("step", round(time.time() - t, 3), "s", y.shape)
EOF
