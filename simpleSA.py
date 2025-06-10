"""
simpleSA.py – yalın Simüle Tavlama
progress_cb(step, best_pen)  → her 500 adımda çağrılır (opsiyonel)
"""
from __future__ import annotations
import math, random, time
from typing import Callable, Optional
import numpy as np
from timetabelingGA import CTTIndividual, mutate

BIG_PENALTY = 100_000

def neighbour(ind: CTTIndividual, rng: random.Random):
    child = ind.clone(); mutate(child, 1/ind.problem.total_lectures); return child

def solve(problem, time_limit: int, seed: int,
          progress_cb: Optional[Callable[[int,int],None]] = None,
          T0: float = 50.0, alpha: float = .95):
    rng = random.Random(seed); np.random.seed(seed)
    cur = best = CTTIndividual(problem, rng)
    pen_best = pen_cur = -cur.fitness()
    T, eval_calls, step = T0, 1, 0
    start = time.perf_counter()

    while time.perf_counter()-start < time_limit:
        step += 1
        nxt = neighbour(cur, rng); eval_calls += 1
        pen_nxt = -nxt.fitness(); delta = pen_nxt-pen_cur
        if delta < 0 or rng.random() < math.exp(-delta/T):
            cur, pen_cur = nxt, pen_nxt
            if pen_cur < pen_best: best, pen_best = cur, pen_cur
        T *= alpha
        if progress_cb and step % 500 == 0: progress_cb(step, pen_best)

    stats = dict(init_penalty=-best.fitness(), best_penalty=pen_best,
                 gen_found=0, div0=-1, div_final=-1,
                 impr_ratio=0.0, stagn_last=-1,
                 feas_gen=-1 if pen_best>=BIG_PENALTY else 0,
                 eval_calls=eval_calls,
                 runtime_s=time.perf_counter()-start)
    return best.to_solution_dict(), stats
