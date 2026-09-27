import re, json, pathlib
src = pathlib.Path("kaggle/xenc_g2.py").read_text(encoding="utf-8")
# robust pair mode: try mp.spawn, fall back to one thread per GPU (Kaggle may run the script as a notebook)
old = "        mp.spawn(_pair_worker, args=(halves,), nprocs=2, join=True)\n"
assert old in src
new = ("        try:\n"
       "            mp.spawn(_pair_worker, args=(halves,), nprocs=2, join=True)\n"
       "        except Exception as e:\n"
       "            import threading\n"
       "            print(f\"mp.spawn failed ({type(e).__name__}: {e}); using 1 thread per GPU\", flush=True)\n"
       "            th = [threading.Thread(target=train_half, args=(h, [i])) for i, h in enumerate(halves)]\n"
       "            [t.start() for t in th]; [t.join() for t in th]\n")
src = src.replace(old, new)
pathlib.Path("kaggle/xenc_g2.py").write_text(src, encoding="utf-8")
m = re.search(r"^CONFIG = \{.*?\}[^\n]*\n", src, flags=re.S | re.M)
assert m, "CONFIG not found"
jobs = {"xenc_g2_xlmr_h0.py": {"model": "FacebookAI/xlm-roberta-base", "mode": "dp", "halves": [0], "lr": 2e-05, "budget_min": 420, "bs_per_gpu": 64},
        "xenc_g2_xlmr_h1.py": {"model": "FacebookAI/xlm-roberta-base", "mode": "dp", "halves": [1], "lr": 2e-05, "budget_min": 420, "bs_per_gpu": 64},
        "xenc_g2_minilm.py": {"model": "microsoft/Multilingual-MiniLM-L12-H384", "mode": "pair", "halves": [0, 1], "lr": 5e-05, "budget_min": 300, "bs_per_gpu": 128}}
for f, c in jobs.items():
    out = src[:m.start()] + f"CONFIG = {json.dumps(c)}  # @CONFIG@ (generated from xenc_g2.py)\n" + src[m.end():]
    compile(out, f, "exec")
    pathlib.Path("kaggle", f).write_text(out, encoding="utf-8")
    print(f, len(out.splitlines()), "lines OK")
