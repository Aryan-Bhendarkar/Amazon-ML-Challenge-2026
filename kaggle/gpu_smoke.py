"""Kaggle GPU smoke test (~2-4 min): proves push -> GPU run -> pull works end to end.
Run:  python scripts/kaggle_gpu.py run kaggle/gpu_smoke.py --slug amlc-smoke --wait"""
import glob
import json
import subprocess
import time

import torch

t0 = time.time()
info = {
    "cuda": torch.cuda.is_available(),
    "n_gpu": torch.cuda.device_count(),
    "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "torch": torch.__version__,
    "inputs": sorted(glob.glob("/kaggle/input/**/*", recursive=True))[:50],
}
if info["cuda"]:
    x = torch.randn(8192, 8192, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize(); t = time.time()
    for _ in range(20):
        y = x @ x
    torch.cuda.synchronize()
    info["fp16_tflops"] = round(20 * 2 * 8192**3 / (time.time() - t) / 1e12, 1)
try:  # HF hub reachable? (needs internet on)
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("microsoft/Multilingual-MiniLM-L12-H384", "config.json")
    info["hf_ok"] = True
except Exception as e:
    info["hf_ok"] = f"FAIL {type(e).__name__}"
info["nvidia_smi"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
info["secs"] = round(time.time() - t0, 1)
print(json.dumps(info, indent=2))
json.dump(info, open("/kaggle/working/smoke.json", "w"), indent=2)
