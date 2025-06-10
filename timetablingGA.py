"""
timetablingGA.py – Adapted CTT Genetic Algorithm with violation-based fitness
and constructive seeding
"""
from __future__ import annotations
import random
import time
from typing import Dict, List, Tuple, Callable, Optional, cast
import numpy as np
from itc2007 import evaluate

# Constants
POP_SIZE_DEFAULT = 40
TIME_LIMIT_DEFAULT = 300  # seconds
BIG_PENALTY = 100_000

# Individual representation
class CTTIndividual:
    def __init__(self, problem, rng: random.Random):
        self.problem = problem
        self.rng = rng
        self.genes = np.zeros((problem.total_lectures, 3), dtype=np.int16)
        self._fitness: Optional[float] = None
        # constructive or random init
        #self.constructive()
        self._fitness = None

    def constructive(self) -> None:
        """Greedy construct: assign each lecture to first non-violating slot."""
        sol: Dict[int, Tuple[int,int,int]] = {}
        D, P, R = self.problem.days, self.problem.periods_per_day, len(self.problem.rooms)
        for lec in range(self.problem.total_lectures):
            assigned = False
            for d in range(D):
                for p in range(P):
                    for r in range(R):
                        sol[lec] = (d, p, r)
                        pen, _ = evaluate(self.problem, sol)
                        # treat initial feasibility: pen < BIG_PENALTY means <=0 violations
                        if pen < BIG_PENALTY:
                            assigned = True
                            break
                    if assigned: break
                if assigned: break
            if not assigned:
                # fallback random slot
                sol[lec] = (self.rng.randrange(D), self.rng.randrange(P), self.rng.randrange(R))
        # write to genes
        for i, slot in sol.items():
            self.genes[i] = slot

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
    child.constructive()
    return child


def mutate(ind: CTTIndividual, rate: float) -> None:
    D, P, R = ind.problem.days, ind.problem.periods_per_day, len(ind.problem.rooms)
    for lec in range(len(ind.genes)):
        if ind.rng.random() < rate:
            ind.genes[lec] = (ind.rng.randrange(D), ind.rng.randrange(P), ind.rng.randrange(R))
    ind._fitness = None
    ind.constructive()

# Helper

def diversity(pop: List[CTTIndividual]) -> int:
    return len({tuple(ind.genes.flatten()) for ind in pop})

# GA loop

def run_ctt_ga(
    problem,
    time_limit: int,
    seed: int,
    pop_size: int,
    progress_cb: Optional[Callable[[int, float], None]] = None
):
    
    rng = random.Random(seed)
    np.random.seed(seed)
    pop = [CTTIndividual(problem, rng) for _ in range(pop_size)]
    eval_calls = pop_size
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
    progress_cb: Optional[Callable[[int, float], None]] = None
) -> Tuple[Dict[int, Tuple[int,int,int]], Dict]:
    best, stats = run_ctt_ga(problem, time_limit, seed, pop_size, progress_cb)
    return best.to_solution_dict(), stats
