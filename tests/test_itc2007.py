import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from itc2007 import load_instance, evaluate

def dataset_path():
    return Path(__file__).resolve().parents[1] / 'datasets/ITC2007/Track2/comp01.ctt'

def test_load_instance_values():
    problem = load_instance(dataset_path())
    assert problem.total_lectures == 160
    assert len(problem.rooms) == 6

def test_evaluate_types():
    problem = load_instance(dataset_path())
    solution = {i: (0, 0, 0) for i in range(problem.total_lectures)}
    penalty, feasible = evaluate(problem, solution)
    assert isinstance(penalty, int)
    assert isinstance(feasible, bool)

