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
    )

# --------------------------------------------------------------------- #

def evaluate(problem: Problem, solution: Dict[int, Tuple[int,int,int]]) -> Tuple[int,bool]:
    """Return (combined_penalty, feasible)."""
    BIG_PENALTY = 100_000

    # 1) Count violations
    violations = 0
    #if len(solution) != problem.total_lectures:
    #    violations += abs(len(solution) - problem.total_lectures)

    # Lecture→Course map
    lec_to_course: List[str] = []
    for c in problem.courses:
        lec_to_course.extend([c.cid] * c.lectures)

    # Maps for clashes
    room_map  = defaultdict(list)
    curr_map  = defaultdict(list)
    teach_map = defaultdict(list)

    for lec_id, (d,p,r) in solution.items():
        # Range and unavailability
        if not (0 <= d < problem.days and 0 <= p < problem.periods_per_day and 0 <= r < len(problem.rooms)):
            violations += 1
            continue
        cid = lec_to_course[lec_id]
        if (d,p) in problem.unavailability.get(cid, set()):
            violations += 1

        room_map[(d,p,r)].append(lec_id)
        for cur in problem.curricula:
            if cid in cur.courses:
                curr_map[(cur.name,d,p)].append(lec_id)
        teacher = problem.course_by_id[cid].teacher
        teach_map[(teacher,d,p)].append(lec_id)

    # Count multi‐person clashes
    for m in (room_map, curr_map, teach_map):
        for lecs in m.values():
            if len(lecs) > 1:
                violations += (len(lecs) - 1)

    # 2) Compute soft penalties
    soft_pen = 0
    rs_map = defaultdict(set)
    days_map = defaultdict(set)
    cc_map = defaultdict(list)

    for lec_id,(d,p,r) in solution.items():
        cid = lec_to_course[lec_id]
        course = problem.course_by_id[cid]
        room_cap = problem.rooms[r][1]
        # Room capacity
        if course.students > room_cap:
            soft_pen += (course.students - room_cap)
        rs_map[cid].add(r)
        days_map[cid].add(d)
        for cur in problem.curricula:
            if cid in cur.courses:
                cc_map[(cur.name,d)].append(p)

    # Minimum Working Days
    for cid, ds in days_map.items():
        deficit = problem.course_by_id[cid].min_days - len(ds)
        if deficit > 0:
            soft_pen += 5 * deficit
    # Room Stability
    soft_pen += sum(max(0, len(rset)-1) for rset in rs_map.values())
    # Curriculum Compactness
    for periods in cc_map.values():
        ps = sorted(periods)
        for i, per in enumerate(ps):
            left = per - (ps[i-1] if i>0 else per)
            right = (ps[i+1] if i+1 < len(ps) else per) - per
            if left > 1 and right > 1:
                soft_pen += 2

    total_pen = violations * BIG_PENALTY + soft_pen
    feasible = (violations == 0)
    return total_pen, feasible

