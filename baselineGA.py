"""
baselineGA.py – sabit mutasyonlu “vanilla” GA
progress_cb(gen, best_pen) -> her 50 nesilde çağrılır (opsiyonel)
"""
import random, time, heapq
from typing import Optional, Callable
import numpy as np
from timetablingGA import CTTIndividual, diversity, mutate, crossover

# smaller default population speeds up search
POP_SIZE_DEFAULT, BIG_PENALTY = 20, 100_000

def run_baseline_ga(problem, time_limit: int, seed: int, pop_size: int,
                    progress_cb: Optional[Callable[[int,int],None]] = None):
    rng = random.Random(seed); np.random.seed(seed)
    pop = [CTTIndividual(problem, rng) for _ in range(pop_size)]
    eval_calls, best = len(pop), max(pop, key=lambda i: i.fitness())
    best_pen, gen, gen_found = -best.fitness(), 0, 0
    start = time.perf_counter()

    while time.perf_counter()-start < time_limit:
        gen += 1
        pop.sort(key=lambda x: x.fitness(), reverse=True)
        elite_cnt = max(1, int(.05*pop_size)); elite = pop[:elite_cnt]
        parents = random.sample(pop[elite_cnt:], k=pop_size//2)
        children = [crossover(random.choice(parents), random.choice(parents))
                    for _ in range(pop_size-elite_cnt)]
        for ch in children:
            mutate(ch, rate=0.01)
        pop, eval_calls = elite+children, eval_calls+len(pop)
        cur_best = max(pop, key=lambda i: i.fitness()); cur_pen = -cur_best.fitness()
        if cur_pen < best_pen: best_pen, best, gen_found = cur_pen, cur_best, gen
        if progress_cb and (gen == 1 or gen % 50 == 0):
            progress_cb(gen, best_pen)

    stats = dict(init_penalty=-max(pop, key=lambda i:i.fitness()).fitness(),
                 best_penalty=best_pen, gen_found=gen_found,
                 div0=diversity(pop), div_final=diversity(pop),
                 impr_ratio=0 if best_pen==BIG_PENALTY else None,
                 stagn_last=gen-gen_found,
                 feas_gen=-1 if best_pen>=BIG_PENALTY else gen_found,
                 eval_calls=eval_calls,
                 runtime_s=time.perf_counter()-start)
    return best.to_solution_dict(), stats

def solve(problem, time_limit: int, seed: int, pop_size: int = POP_SIZE_DEFAULT,
          progress_cb: Optional[Callable[[int,int],None]] = None):
    return run_baseline_ga(problem, time_limit, seed, pop_size, progress_cb)
