import sys
from pathlib import Path
import random
import pytest

pytest.importorskip("numpy")
sys.path.append(str(Path(__file__).resolve().parents[1]))

import timetablingGA as ga
from itc2007 import load_instance

def dataset_path():
    return Path(__file__).resolve().parents[1] / 'datasets/ITC2007/Track2/comp01.ctt'


def test_constructive_without_evaluate(monkeypatch):
    problem = load_instance(dataset_path())
    ind = ga.CTTIndividual(problem, random.Random(0))
    called = False

    def fake_eval(*args, **kwargs):
        nonlocal called
        called = True
        return 0, True

    monkeypatch.setattr(ga, "evaluate", fake_eval)
    ind.constructive()
    assert called is False


def test_constructive_calls_evaluate_once(monkeypatch):
    problem = load_instance(dataset_path())
    ind = ga.CTTIndividual(problem, random.Random(1))
    calls = []
    orig = ga.evaluate

    def wrapped(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(ga, "evaluate", wrapped)
    ind.constructive()
    ind.fitness()
    assert len(calls) == 1
