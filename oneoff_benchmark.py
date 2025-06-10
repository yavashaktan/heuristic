
# oneoff_benchmark.py — Windows‑uyumlu, izleme + watch‑dog + result logging + extended summary
from __future__ import annotations

import csv
import importlib
import itertools
import json
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from logger import log

#################### parameters ####################
SCORER_MOD = "itc2007"          # scorer module (projedir)
TIME_LIMIT = 300                 # s – algorithm time budget
TIME_PAD   = 120                 # s – watchdog cushion
SEEDS      = list(range(10))

ALG_SPECS = [
    {"name": "GA",         "module": "timetabelingGA", "kwargs": {"pop_size": 40}},
    {"name": "BaselineGA", "module": "baselineGA",     "kwargs": {"pop_size": 40}},
    {"name": "SimAnneal",  "module": "simpleSA",       "kwargs": {}},
]
####################################################


def progress_bar(done: int, total: int, barlen: int = 30):
    pct = done / total
    filled = int(barlen * pct)
    bar = "#" * filled + "-" * (barlen - filled)
    sys.stdout.write(f"\r[{bar}] {pct*100:5.1f}% ({done}/{total})\x1b[K")
    sys.stdout.flush()


def load_skiplist() -> set[Tuple[str, str, int]]:
    if not Path("runs.csv").exists():
        return set()
    with Path("runs.csv").open(newline="") as f:
        return {(r["instance"], r["algorithm"], int(r["seed"])) for r in csv.DictReader(f)}

# ───────────────────────── worker (spawn‑safe) ─────────────────────────

def _worker_run_alg(q: mp.Queue,
                    inst_str: str,
                    alg_name: str,
                    alg_module: str,
                    alg_kwargs: Dict,
                    seed: int,
                    scorer_mod: str,
                    time_limit: int):
    """Run a single (instance, algorithm, seed) in a child process."""

    import importlib
    import time
    from pathlib import Path

    # ensure project root on PYTHONPATH
    proj_root = Path(__file__).parent
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    scorer = importlib.import_module(scorer_mod)
    import types
    if "scorers" not in sys.modules:
        sys.modules["scorers"] = types.ModuleType("scorers")
    sys.modules["scorers.itc2007"] = scorer
    alg_mod = importlib.import_module(alg_module)

    problem = scorer.load_instance(Path(inst_str))

    # progress callback — logs every 50 generations / 500 SA steps
    t0 = time.perf_counter()

    def progress_cb(gen: int, best_pen: int, diversity: int | None = None):
        if gen % 50 == 0:
            msg = (f"{inst_str} | {alg_name} | seed={seed} | gen={gen} | "
                   f"best={best_pen} | t={int(time.perf_counter()-t0)}s")
            if diversity is not None:
                msg += f" | div={diversity}"
            log(msg, level="DBG")

    # run algorithm
    start = time.perf_counter()
    result = alg_mod.solve(
        problem=problem,
        time_limit=time_limit,
        seed=seed,
        progress_cb=progress_cb,
        **alg_kwargs,
    )
    runtime = time.perf_counter() - start

    solution, stats = result if isinstance(result, tuple) else (result, {})

    penalty, feasible = scorer.evaluate(problem, solution)
    row = {
        "instance": Path(inst_str).name,
        "algorithm": alg_name,
        "seed": seed,
        "penalty": penalty,
        "feasible": feasible,
        "runtime_s": runtime,
        **stats,
    }

    log(f"RESULT | {row}", level="RESULT")

    # save solution JSON
    json_path = Path(f"solution_{Path(inst_str).stem}_{alg_name}_{seed}.json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(solution, f)

    q.put(row)

# ───────────────────────── watchdog wrapper ─────────────────────────

def run_alg_safe(inst_path: Path, alg_spec: Dict, seed: int) -> Dict:
    q = mp.Queue()
    proc = mp.Process(target=_worker_run_alg,
                      args=(q, str(inst_path), alg_spec["name"], alg_spec["module"],
                            alg_spec.get("kwargs", {}), seed, SCORER_MOD, TIME_LIMIT))
    proc.start()
    proc.join(TIME_LIMIT + TIME_PAD)

    if proc.is_alive():
        proc.terminate(); proc.join()
        log(f"TIMEOUT: {inst_path.name} | {alg_spec['name']} | seed={seed}", "WARN")
        return {"instance": inst_path.name, "algorithm": alg_spec["name"], "seed": seed, "timeout": True}
    try:
        return q.get_nowait()
    except queue.Empty:
        return {"instance": inst_path.name, "algorithm": alg_spec["name"], "seed": seed, "timeout": True}

# ───────────────────────── main driver ─────────────────────────

def main():
    mp.set_start_method("spawn", force=True)

    inst_paths = sorted(Path(__file__).parent.rglob("*.ctt"))
    skip = load_skiplist()
    total = len(inst_paths) * len(ALG_SPECS) * len(SEEDS) - len(skip)
    done = 0
    log(f"Çalışma başlatıldı — hedef {total} koşu")

    for inst, alg, seed in itertools.product(inst_paths, ALG_SPECS, SEEDS):
        if (inst.name, alg["name"], seed) in skip:
            continue
        done += 1
        progress_bar(done, total)
        print(f"\n>> STARTING {inst.name} | {alg['name']} | seed={seed}", flush=True)
        row = run_alg_safe(inst, alg, seed)
        pd.DataFrame([row]).to_csv("runs.csv", mode="a", index=False,
                                   header=not Path("runs.csv").exists())
    print(f"✅ Completed: {inst.name} | {alg['name']} | seed={seed} → "
              f"penalty={row['penalty']} feasible={row['feasible']} "
              f"time={row['runtime_s']:.1f}s", flush=True)
    # summary csv
    df = pd.read_csv("runs.csv")
    summary = (
        df.groupby(["instance", "algorithm"])
          .agg(min_penalty=("penalty", "min"),
               max_penalty=("penalty", "max"),
               median_penalty=("penalty", "median"),
               mean_penalty=("penalty", "mean"),
               std_penalty=("penalty", "std"),
               feasible_rate=("feasible", "mean"),
               mean_runtime_s=("runtime_s", "mean"))
          .reset_index()
    )
    summary.to_csv("summary.csv", index=False)
    log("Tamamlandı ➜ runs.csv, summary.csv")

# ───────────────────────── entry ─────────────────────────
if __name__ == "__main__":
    import os
    from pathlib import Path
    # 1) Force cwd → script folder
    script_dir = Path(__file__).parent
    print(f"DEBUG: Changing cwd to script folder: {script_dir}", flush=True)
    os.chdir(script_dir)

    # 2) Report what CTT files we actually see
    ctts = list(script_dir.rglob("*.ctt"))
    print(f"DEBUG: Found {len(ctts)} .ctt file(s): {[p.name for p in ctts]}", flush=True)

    # 3) Unbuffered prints from now on
    sys.stdout.reconfigure(line_buffering=True)

    main()
