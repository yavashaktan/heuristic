"""
timetablingGA.py – Adapted CTT Genetic Algorithm with violation-based fitness
and constructive seeding
"""
from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple, Callable, Optional, cast
import multiprocessing as mp
import numpy as np
from itc2007 import evaluate, evaluate_vectorized

# Constants
# default population size was somewhat high; reduce for faster runs
POP_SIZE_DEFAULT = 20

# probability of applying the constructive repair after genetic operators
CONSTRUCTIVE_PROB = 0.5
TIME_LIMIT_DEFAULT = 300  # seconds
BIG_PENALTY = 100_000

# Individual representation
class CTTIndividual:
    def __init__(self, problem, rng: random.Random):
        self.problem = problem
        self.rng = rng
        self.genes = np.zeros((problem.total_lectures, 3), dtype=np.int16)
        self._fitness: Optional[float] = None

        # resource index maps
        teachers = sorted({c.teacher for c in problem.courses})
        curricula = [c.name for c in problem.curricula]
        self.teacher_idx = {t: i for i, t in enumerate(teachers)}
        self.curr_idx = {c: i for i, c in enumerate(curricula)}

        D, P, R = problem.days, problem.periods_per_day, len(problem.rooms)
        self.room_busy = np.zeros((D, P, R), dtype=bool)
        self.teacher_busy = np.zeros((D, P, len(self.teacher_idx)), dtype=bool)
        self.curr_busy = np.zeros((D, P, len(self.curr_idx)), dtype=bool)

        # mapping from lecture to teacher/curricula indices
        self.lec_teacher = np.empty(problem.total_lectures, dtype=np.int16)
        self.lec_currs: List[List[int]] = []
        for lec, cid in enumerate(problem.lec_to_course):
            teacher = problem.course_by_id[cid].teacher
            self.lec_teacher[lec] = self.teacher_idx[teacher]
            self.lec_currs.append([self.curr_idx[c] for c in problem.course_to_curricula.get(cid, [])])

        # unavailability matrix per lecture
        self.unavail = np.zeros((problem.total_lectures, D, P), dtype=bool)
        for lec, cid in enumerate(problem.lec_to_course):
            for d, p in problem.unavailability.get(cid, set()):
                if d < D and p < P:
                    self.unavail[lec, d, p] = True

        # constructive or random init
        # self.constructive()

    def constructive(self) -> None:
        """Greedy construct using occupancy matrices."""
        D, P, R = self.problem.days, self.problem.periods_per_day, len(self.problem.rooms)

        self.room_busy.fill(False)
        self.teacher_busy.fill(False)
        self.curr_busy.fill(False)

        for lec in range(self.problem.total_lectures):
            t_idx = self.lec_teacher[lec]
            cur_idxs = self.lec_currs[lec]
            placed = False
            for d in range(D):
                for p in range(P):
                    if self.unavail[lec, d, p]:
                        continue
                    if self.teacher_busy[d, p, t_idx]:
                        continue
                    if any(self.curr_busy[d, p, ci] for ci in cur_idxs):
                        continue
                    for r in range(R):
                        if self.room_busy[d, p, r]:
                            continue
                        # place lecture
                        self.genes[lec] = (d, p, r)
                        self.room_busy[d, p, r] = True
                        self.teacher_busy[d, p, t_idx] = True
                        for ci in cur_idxs:
                            self.curr_busy[d, p, ci] = True
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                d = self.rng.randrange(D)
                p = self.rng.randrange(P)
                r = self.rng.randrange(R)
                self.genes[lec] = (d, p, r)
                self.room_busy[d, p, r] = True
                self.teacher_busy[d, p, t_idx] = True
                for ci in cur_idxs:
                    self.curr_busy[d, p, ci] = True
        self._fitness = None

    def to_solution_dict(self) -> Dict[int, Tuple[int,int,int]]:
        return {i: cast(Tuple[int, int, int], tuple(map(int, self.genes[i])))
                for i in range(len(self.genes))}

    def fitness(self) -> float:
        if self._fitness is None:
            pen, feas = evaluate(self.problem, self.to_solution_dict())
            self._fitness = -pen
        return self._fitness

    def clone(self) -> "CTTIndividual":
        child = CTTIndividual(self.problem, self.rng)
        child.genes = self.genes.copy()
        child._fitness = self._fitness
        return child

# Operators

def crossover(p1: CTTIndividual, p2: CTTIndividual) -> CTTIndividual:
    child = p1.clone()
    mask = np.random.rand(len(p1.genes)) < 0.5
    child.genes[~mask] = p2.genes[~mask]
    child._fitness = None
    if random.random() < CONSTRUCTIVE_PROB:
        child.constructive()
    return child


def mutate(ind: CTTIndividual, rate: float) -> None:
    D, P, R = ind.problem.days, ind.problem.periods_per_day, len(ind.problem.rooms)
    mask = np.random.rand(len(ind.genes)) < rate
    if mask.any():
        n = mask.sum()
        ind.genes[mask, 0] = np.random.randint(0, D, size=n)
        ind.genes[mask, 1] = np.random.randint(0, P, size=n)
        ind.genes[mask, 2] = np.random.randint(0, R, size=n)
        ind._fitness = None
        if ind.rng.random() < CONSTRUCTIVE_PROB:
            ind.constructive()

# Helper

def diversity(pop: List[CTTIndividual]) -> int:
    return len({tuple(ind.genes.flatten()) for ind in pop})


def _eval_chunk(args):
    problem, arr = args
    pens, _ = evaluate_vectorized(problem, arr)
    return pens


def compute_population_fitness(pop: List[CTTIndividual], problem, processes: int | None = None) -> None:
    """Evaluate all individuals, optionally in parallel."""
    arr = np.stack([ind.genes for ind in pop])
    if processes and processes > 1:
        chunks = np.array_split(arr, processes)
        with mp.Pool(processes) as pool:
            results = pool.map(_eval_chunk, [(problem, c) for c in chunks])
        pens = np.concatenate(results)
    else:
        pens, _ = evaluate_vectorized(problem, arr)
    for ind, pen in zip(pop, pens):
        ind._fitness = -float(pen)

# GA loop

def run_ctt_ga(
    problem,
    time_limit: int,
    seed: int,
    pop_size: int,
    progress_cb: Optional[Callable[[int, float], None]] = None,
    processes: int | None = None,
):
    
    rng = random.Random(seed)
    np.random.seed(seed)
    pop = [CTTIndividual(problem, rng) for _ in range(pop_size)]
    compute_population_fitness(pop, problem, processes)
    eval_calls = len(pop)
    start = time.time()

    best = max(pop, key=lambda i: i.fitness())
    best_pen = -best.fitness()
    gen_found = 0
    gen = 0
    stats = {"div0": diversity(pop), "init_penalty": best_pen, "eval_calls": eval_calls}
    print(f"[GA DEBUG] initial best_pen={best_pen}", flush=True)

    while time.time() - start < time_limit:
        # ensure we log at least once
        gen += 1

        pop.sort(key=lambda i: i.fitness(), reverse=True)
        retain = int(len(pop) * 0.6)
        parents = pop[:retain] + [p for p in pop[retain:] if rng.random() < 0.1]
        children = []
        while len(children) < pop_size - len(parents):
            p1, p2 = rng.choice(parents), rng.choice(parents)
            children.append(crossover(p1, p2))
        mut_rate = 0.20 if diversity(pop) < 3 else 0.01
        for ch in children:
            mutate(ch, mut_rate)
        pop = parents + children
        compute_population_fitness(pop, problem, processes)
        eval_calls += len(pop)

        cur = max(pop, key=lambda i: i.fitness())
        cur_pen = -cur.fitness()
        if cur_pen < best_pen:
            best, best_pen, gen_found = cur, cur_pen, gen
        if progress_cb:
            progress_cb(gen, best_pen)
        # only break on perfect zero if you really want to stop early:
        if best_pen == 0 and gen >= 10:
            break

    runtime = time.time() - start
    stats.update(
        best_penalty=best_pen,
        gen_found=gen_found,
        div_final=diversity(pop),
        impr_ratio=(stats['init_penalty'] - best_pen)/stats['init_penalty'] if stats['init_penalty']>0 else 0,
        stagn_last=gen - gen_found,
        feas_gen=gen_found,
        runtime_s=runtime,
        eval_calls=eval_calls
    )
    return best, stats

# Solve interface

def solve(
    problem,
    time_limit: int = TIME_LIMIT_DEFAULT,
    seed: int = 0,
    pop_size: int = POP_SIZE_DEFAULT,
    progress_cb: Optional[Callable[[int, float], None]] = None,
    processes: int | None = None,
) -> Tuple[Dict[int, Tuple[int,int,int]], Dict]:
    best, stats = run_ctt_ga(problem, time_limit, seed, pop_size, progress_cb, processes)
    return best.to_solution_dict(), stats
