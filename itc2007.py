"""
scorers/itc2007.py
=================
Robust loader & scorer for ITC-2007 Track-2 CB-CTT problems.
Includes hard-constraint enforcement and soft-penalty calculation.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import inf
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

@dataclass
class Course:
    cid: str
    teacher: str
    lectures: int
    min_days: int
    students: int

@dataclass
class Curriculum:
    name: str
    courses: List[str]

@dataclass
class Problem:
    name: str
    days: int
    periods_per_day: int
    rooms: List[Tuple[str, int]]            # list of (room_id, capacity)
    courses: List[Course]
    course_by_id: Dict[str, Course]
    curricula: List[Curriculum]
    total_lectures: int
    unavailability: Dict[str, set[Tuple[int,int]]]
    lec_to_course: List[str]
    course_to_curricula: Dict[str, List[str]]

# --------------------------------------------------------------------- #

def load_instance(path: Path) -> Problem:
    """Load a Track-2 CTT instance from file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    # parse header key-values
    header = {}
    for ln in lines:
        if ':' not in ln:
            continue
        key, val = ln.split(':', 1)
        key = key.strip()
        val = val.strip()
        if key in ("Name", "Courses", "Rooms", "Days", "Periods_per_day", "Curricula", "Constraints"):
            header[key] = val
    # required fields
    name = header.get("Name", "")
    courses_n = int(header.get("Courses", 0))
    rooms_n = int(header.get("Rooms", 0))
    days_n = int(header.get("Days", 0))
    periods_n = int(header.get("Periods_per_day", 0))
    curricula_n = int(header.get("Curricula", 0))
    constraints_n = int(header.get("Constraints", 0))

    # helper to locate section
    def find_section(label: str) -> int:
        """Locate a section header line exactly matching the label."""
        for idx, ln in enumerate(lines):
            if ln.strip().upper() == label.upper():
                return idx + 1
        raise ValueError(f"Section '{label}' not found.")

    # parse courses
    idx = find_section("COURSES:")
    courses: List[Course] = []
    course_by_id: Dict[str, Course] = {}
    total_lectures = 0
    for i in range(courses_n):
        parts = lines[idx + i].split()
        if len(parts) < 5:
            raise ValueError(f"Malformed course line: '{lines[idx+i]}'")
        cid, teacher, lec, minw, stud = parts[:5]
        c = Course(cid=cid,
                   teacher=teacher,
                   lectures=int(lec),
                   min_days=int(minw),
                   students=int(stud))
        courses.append(c)
        course_by_id[cid] = c
        total_lectures += c.lectures
    # parse rooms
    idx = find_section("ROOMS:")
    rooms: List[Tuple[str,int]] = []
    for i in range(rooms_n):
        parts = lines[idx + i].split()
        if len(parts) < 2:
            raise ValueError(f"Malformed room line: '{lines[idx+i]}'")
        rid, cap = parts[:2]
        rooms.append((rid, int(cap)))
    # parse curricula
    idx = find_section("CURRICULA:")
    curricula: List[Curriculum] = []
    for i in range(curricula_n):
        parts = lines[idx + i].split()
        if len(parts) < 3:
            raise ValueError(f"Malformed curriculum line: '{lines[idx+i]}'")
        cname = parts[0]
        course_list = parts[2:]
        curricula.append(Curriculum(name=cname, courses=course_list))
    # parse unavailability
    try:
        idx = find_section("UNAVAILABILITY_CONSTRAINTS:")
    except ValueError:
        idx = find_section("UNAVAILABILITY:")
    unavailability: Dict[str, set[Tuple[int,int]]] = defaultdict(set)
    j = idx
    while j < len(lines) and lines[j].strip() and not lines[j].strip().upper().startswith("END"):
        parts = lines[j].split()
        if len(parts) >= 3:
            unavailability[parts[0]].add((int(parts[1]), int(parts[2])))
        j += 1

    course_to_curricula: Dict[str, List[str]] = defaultdict(list)
    for cur in curricula:
        for cid in cur.courses:
            course_to_curricula[cid].append(cur.name)

    lec_to_course: List[str] = []
    for c in courses:
        lec_to_course.extend([c.cid] * c.lectures)

    return Problem(
        name=name,
        days=days_n,
        periods_per_day=periods_n,
        rooms=rooms,
        courses=courses,
        course_by_id=course_by_id,
        curricula=curricula,
        total_lectures=total_lectures,
        unavailability=unavailability,
        lec_to_course=lec_to_course,
        course_to_curricula=course_to_curricula,
    )

# --------------------------------------------------------------------- #

def _prepare_eval_cache(problem: Problem):
    """Create numpy helper arrays for fast evaluation."""
    if hasattr(problem, "_eval_cache"):
        return problem._eval_cache

    L = problem.total_lectures
    D = problem.days
    P = problem.periods_per_day

    teacher_names = sorted({c.teacher for c in problem.courses})
    t_idx = {t: i for i, t in enumerate(teacher_names)}

    curr_names = [c.name for c in problem.curricula]
    c_idx = {c: i for i, c in enumerate(curr_names)}

    lec_teacher = np.empty(L, dtype=np.int16)
    lec_curr = []
    lec_students = np.empty(L, dtype=np.int32)
    lec_min_days = np.empty(L, dtype=np.int16)

    course_lecs: Dict[str, List[int]] = defaultdict(list)
    for i, cid in enumerate(problem.lec_to_course):
        course = problem.course_by_id[cid]
        lec_teacher[i] = t_idx[course.teacher]
        lec_students[i] = course.students
        lec_min_days[i] = course.min_days
        lec_curr.append([c_idx[c] for c in problem.course_to_curricula.get(cid, [])])
        course_lecs[cid].append(i)

    unavail = np.zeros((L, D, P), dtype=bool)
    for cid, slots in problem.unavailability.items():
        for li in course_lecs[cid]:
            for d, p in slots:
                if d < D and p < P:
                    unavail[li, d, p] = True

    problem._eval_cache = (
        lec_teacher,
        lec_curr,
        lec_students,
        lec_min_days,
        unavail,
        len(teacher_names),
        len(curr_names),
    )
    return problem._eval_cache


def evaluate(problem: Problem, solution: Dict[int, Tuple[int, int, int]] | np.ndarray) -> Tuple[int, bool]:
    """Return (combined_penalty, feasible). Accepts dict or array."""

    if isinstance(solution, dict):
        arr = np.zeros((problem.total_lectures, 3), dtype=np.int16)
        for i in range(problem.total_lectures):
            arr[i] = solution[i]
    else:
        arr = np.asarray(solution, dtype=np.int16)
        if arr.shape[0] != problem.total_lectures:
            raise ValueError("Invalid solution shape")

    penalties, feas = evaluate_vectorized(problem, arr[None, :, :])
    return int(penalties[0]), bool(feas[0])


def evaluate_vectorized(problem: Problem, solutions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized evaluation for a batch of solutions."""

    lec_teacher, lec_curr, lec_students, lec_min_days, unavail, n_teachers, n_curr = _prepare_eval_cache(problem)

    n, L, _ = solutions.shape
    D, P, R = problem.days, problem.periods_per_day, len(problem.rooms)
    BIG_PENALTY = 100_000

    d = solutions[:, :, 0]
    p = solutions[:, :, 1]
    r = solutions[:, :, 2]

    penalties = np.zeros(n, dtype=np.int64)
    feasible = np.ones(n, dtype=bool)

    for idx in range(n):
        dv = d[idx]
        pv = p[idx]
        rv = r[idx]

        valid = (
            (dv >= 0)
            & (dv < D)
            & (pv >= 0)
            & (pv < P)
            & (rv >= 0)
            & (rv < R)
        )
        viol = (~valid).sum()

        dv_clipped = dv.clip(0, D - 1)
        pv_clipped = pv.clip(0, P - 1)

        viol += unavail[np.arange(L), dv_clipped, pv_clipped][valid].sum()

        room_count = np.zeros((D, P, R), dtype=np.int16)
        np.add.at(room_count, (dv_clipped[valid], pv_clipped[valid], rv[valid]), 1)
        viol += (room_count[room_count > 1] - 1).sum()

        teach_count = np.zeros((D, P, n_teachers), dtype=np.int16)
        np.add.at(teach_count, (dv_clipped[valid], pv_clipped[valid], lec_teacher[valid]), 1)
        viol += (teach_count[teach_count > 1] - 1).sum()

        curr_count = np.zeros((D, P, n_curr), dtype=np.int16)
        valid_idx = np.nonzero(valid)[0]
        for li in valid_idx:
            for ci in lec_curr[li]:
                curr_count[dv_clipped[li], pv_clipped[li], ci] += 1
        viol += (curr_count[curr_count > 1] - 1).sum()

        soft_pen = 0
        room_cap = np.array([cap for _, cap in problem.rooms])
        soft_pen += np.maximum(lec_students[valid] - room_cap[rv[valid]], 0).sum()

        rs_map = defaultdict(set)
        days_map = defaultdict(set)
        cc_map = defaultdict(list)
        for li in valid_idx:
            cid = problem.lec_to_course[li]
            rs_map[cid].add(rv[li])
            days_map[cid].add(dv_clipped[li])
            for ci in lec_curr[li]:
                cc_map[(ci, dv_clipped[li])].append(pv_clipped[li])

        for cid, ds in days_map.items():
            deficit = problem.course_by_id[cid].min_days - len(ds)
            if deficit > 0:
                soft_pen += 5 * deficit
        soft_pen += sum(max(0, len(rset) - 1) for rset in rs_map.values())
        for periods in cc_map.values():
            ps = sorted(periods)
            for i, per in enumerate(ps):
                left = per - (ps[i - 1] if i > 0 else per)
                right = (ps[i + 1] if i + 1 < len(ps) else per) - per
                if left > 1 and right > 1:
                    soft_pen += 2

        penalties[idx] = viol * BIG_PENALTY + soft_pen
        feasible[idx] = viol == 0

    return penalties, feasible

