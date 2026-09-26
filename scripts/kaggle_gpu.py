"""Use Kaggle's free GPUs (2x T4) as a remote GPU, driven from the AWS box / VS Code terminal.

AWS GPU quota is 0 on our accounts, so GPU jobs run as private Kaggle "script" kernels:
upload inputs as a PRIVATE dataset -> push a self-contained script -> it runs in the background on
Kaggle GPUs -> poll -> download /kaggle/working outputs here.

  python scripts/kaggle_gpu.py whoami
  python scripts/kaggle_gpu.py data   <dir> --slug amlc-xenc-data [-m "msg"]
  python scripts/kaggle_gpu.py run    <script.py> --slug amlc-xenc [--data amlc-xenc-data ...] [--kernels amlc-xenc]
                                      [--gpu t4x2] [--no-internet] [--wait] [--timeout SEC]
  python scripts/kaggle_gpu.py status --slug amlc-xenc
  python scripts/kaggle_gpu.py wait   --slug amlc-xenc          # poll until done, then pull
  python scripts/kaggle_gpu.py pull   --slug amlc-xenc [--out artifacts/kaggle/amlc-xenc]

Auth (one-time, human): kaggle.com -> Settings -> API -> "Create New Token" (kaggle.json) ->
put it at ~/.kaggle/kaggle.json with chmod 600. Account must be phone-verified for GPU + internet.
Everything this script creates is PRIVATE. Never pass --public anywhere (competition data).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GPU = {"t4x2": "NvidiaTeslaT4", "t4": "NvidiaTeslaT4", "l4": "NvidiaL4", "l4x1": "NvidiaL4X1"}


def kaggle_bin() -> str:
    for c in (shutil.which("kaggle"), str(Path.home() / ".local/bin/kaggle")):
        if c and Path(c).exists():
            return c
    sys.exit("kaggle CLI not found. Install: uv tool install kaggle")


def sh(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    cmd = [kaggle_bin(), *args]
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    out = (r.stdout + r.stderr).strip()
    if out:
        print(out, flush=True)
    if check and r.returncode != 0:
        sys.exit(f"kaggle command failed ({r.returncode})")
    return r


def username(cli_user: str | None) -> str:
    if cli_user:
        return cli_user
    if os.environ.get("KAGGLE_USERNAME"):
        return os.environ["KAGGLE_USERNAME"]
    kj = Path.home() / ".kaggle/kaggle.json"
    if kj.exists():
        return json.loads(kj.read_text())["username"]
    r = subprocess.run([kaggle_bin(), "config", "view"], capture_output=True, text=True)  # access_token auth
    m = re.search(r"username:\s*(\S+)", r.stdout)
    if m and m.group(1) != "None":
        return m.group(1)
    sys.exit("Unknown Kaggle username: set KAGGLE_USERNAME or put kaggle.json in ~/.kaggle/")


def cmd_data(a):
    user, d = username(a.user), Path(a.dir).resolve()
    if not d.is_dir():
        sys.exit(f"not a directory: {d}")
    ref = f"{user}/{a.slug}"
    (d / "dataset-metadata.json").write_text(json.dumps(
        {"title": a.slug, "id": ref, "licenses": [{"name": "unknown"}]}, indent=2))
    size = sum(p.stat().st_size for p in d.rglob("*") if p.is_file()) / 1e9
    print(f"[data] {ref}: {size:.2f} GB from {d}")
    exists = sh("datasets", "status", ref, check=False).returncode == 0
    if exists:
        sh("datasets", "version", "-p", str(d), "-m", a.message, "--dir-mode", "zip")
    else:  # private by default; never add --public
        sh("datasets", "create", "-p", str(d), "--dir-mode", "zip")
    for _ in range(120):  # wait until processed so a kernel can mount it
        r = sh("datasets", "status", ref, check=False)
        if "ready" in (r.stdout + r.stderr).lower():
            print(f"[data] ready: {ref}")
            return
        time.sleep(15)
    print("[data] still processing - check `kaggle datasets status` before pushing a kernel")


def build_kernel_dir(a, user: str) -> Path:
    src = Path(a.script).resolve()
    kdir = ROOT / "build/kaggle" / a.slug
    if kdir.exists():
        shutil.rmtree(kdir)
    kdir.mkdir(parents=True)
    shutil.copy(src, kdir / src.name)
    meta = {
        "id": f"{user}/{a.slug}", "title": a.slug, "code_file": src.name,
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": a.gpu != "none", "enable_internet": not a.no_internet,
        "dataset_sources": [x if "/" in x else f"{user}/{x}" for x in a.data],
        "competition_sources": [], "kernel_sources": [x if "/" in x else f"{user}/{x}" for x in a.kernels],
    }
    if a.gpu != "none":
        meta["machine_shape"] = GPU.get(a.gpu, a.gpu)
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return kdir


def cmd_run(a):
    user = username(a.user)
    kdir = build_kernel_dir(a, user)
    args = ["kernels", "push", "-p", str(kdir)]
    if a.gpu != "none":
        args += ["--accelerator", GPU.get(a.gpu, a.gpu)]
    if a.timeout:
        args += ["-t", str(a.timeout)]
    r = sh(*args)
    o = (r.stdout + r.stderr).lower()                      # the CLI exits 0 on "Maximum batch GPU session count" and
    if "error" in o or "not valid" in o:                   # on missing sources ("not valid dataset sources")
        sys.exit(f"[run] push FAILED for {user}/{a.slug}")
    print(f"[run] pushed {user}/{a.slug}  (https://www.kaggle.com/code/{user}/{a.slug})")
    if a.wait:
        cmd_wait(a)


def status_of(user: str, slug: str) -> str:
    r = sh("kernels", "status", f"{user}/{slug}", check=False)
    m = re.search(r'status "?([A-Za-z_.]+)"?', r.stdout + r.stderr)
    return (m.group(1) if m else (r.stdout + r.stderr)).split(".")[-1].lower()


def cmd_status(a):
    print(status_of(username(a.user), a.slug))


def cmd_wait(a):
    user, t0 = username(a.user), time.time()
    while True:
        s = status_of(user, a.slug)
        print(f"[wait] {time.strftime('%H:%M:%S')} {s} ({(time.time()-t0)/60:.0f} min)", flush=True)
        if s in ("complete", "error", "cancelacknowledged", "cancelrequested"):
            break
        time.sleep(a.poll)
    cmd_pull(a)
    if s != "complete":
        sys.exit(f"kernel ended with status {s} - read the .log in the pulled folder")


def cmd_pull(a):
    user = username(a.user)
    out = Path(a.out or ROOT / "artifacts/kaggle" / a.slug)
    out.mkdir(parents=True, exist_ok=True)
    sh("kernels", "output", f"{user}/{a.slug}", "-p", str(out), "-o", check=False)
    print(f"[pull] -> {out}")
    for p in sorted(out.iterdir()):
        print(f"   {p.name}  {p.stat().st_size/1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", help="Kaggle username (default: kaggle.json / $KAGGLE_USERNAME)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("whoami")
    p = sp.add_parser("data"); p.add_argument("dir"); p.add_argument("--slug", required=True)
    p.add_argument("-m", "--message", default="update")
    p = sp.add_parser("run"); p.add_argument("script"); p.add_argument("--slug", required=True)
    p.add_argument("--data", nargs="*", default=[]); p.add_argument("--gpu", default="t4x2")
    p.add_argument("--kernels", nargs="*", default=[], help="kernel outputs to mount (e.g. checkpoints)")
    p.add_argument("--no-internet", action="store_true"); p.add_argument("--timeout", type=int)
    p.add_argument("--wait", action="store_true"); p.add_argument("--poll", type=int, default=60)
    p.add_argument("--out")
    for name in ("status", "wait", "pull"):
        p = sp.add_parser(name); p.add_argument("--slug", required=True)
        p.add_argument("--out"); p.add_argument("--poll", type=int, default=60)
    a = ap.parse_args()
    if a.cmd == "whoami":
        print("user:", username(a.user)); sh("kernels", "list", "--mine", "--page-size", "5", check=False)
        return
    {"data": cmd_data, "run": cmd_run, "status": cmd_status, "wait": cmd_wait, "pull": cmd_pull}[a.cmd](a)


if __name__ == "__main__":
    main()
