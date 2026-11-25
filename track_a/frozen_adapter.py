"""Synthetic fixture adapter and explicit binding to the frozen wind scorer.

The standalone implementation is a local engineering shim. Direct binding uses
isolated source modules, native Instance/CostAnchor objects and the full frozen
solution schema; neither mode establishes a server run or a scientific result.
"""
from __future__ import annotations

import hashlib
import builtins
import importlib.util
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple

FROZEN_SLOT_MODELS = ("LD", "SD")
FROZEN_SLOT_CAPS = (1.00, 0.81)
STATION_TYPE_CODES = {"CLOSED": 0, "SABS": 1, "DABS": 2}
STATION_FIXED_COST_RMB = {"CLOSED": 0.0, "SABS": 27.8, "DABS": 43.4}
STATION_COVERAGE_RADIUS_M = {"CLOSED": 0.0, "SABS": 4000.0, "DABS": 5000.0}
AIRSPEED_MPS = 15.0
AIRSPEED_M_PER_MIN = 900.0
DUTY_COST_RMB_PER_MIN = 0.5
R_FLY_PER_MIN = 0.018
DEFAULT_TOWER_SRV_RATE_PER_MIN = 0.020
ENERGY_TOL = 1e-12
COST_TOL = 1e-7
COVERAGE_TOL_M = 1e-7
PHYSICS_TOL = 1e-7
DEFAULT_PARAMETERS = dict(
    speed=900.0, flight_energy=0.018, small_battery=0.81, large_battery=1.0,
    small_radius=4000.0, large_radius=5000.0, small_duration=480.0,
    large_duration=600.0, small_station_cost=22.8, large_station_cost=30.4,
    small_drone_cost=5.0, large_drone_cost=8.0, time_cost=0.5,
)


def _finite(value: Any, *, nonnegative: bool = False) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value) and (not nonnegative or value >= 0)
    except (ValueError, OverflowError):
        return False


def _xy(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 2 and all(_finite(x) for x in value)


def wind_mps_to_m_per_min(wind_mps: float) -> float:
    if not _finite(wind_mps):
        raise ValueError("wind component must be a finite number")
    value = wind_mps * 60.0
    if not math.isfinite(value):
        raise ValueError("converted wind component must be finite")
    return value


def _standalone_heading_leg(p1: Sequence[float], p2: Sequence[float],
                            wind_speed_m_per_min: float = 540.0,
                            wind_dir_rad: float = 0.0,
                            air_speed_m_per_min: float = AIRSPEED_M_PER_MIN) -> float:
    """Independent heading formula, with the frozen scorer's m/min units."""
    if (not _xy(p1) or not _xy(p2) or not _finite(wind_speed_m_per_min, nonnegative=True)
            or not _finite(wind_dir_rad) or not _finite(air_speed_m_per_min)
            or air_speed_m_per_min <= wind_speed_m_per_min):
        raise ValueError("finite coordinates and wind strictly below positive airspeed required")
    wind = (wind_speed_m_per_min * math.cos(wind_dir_rad),
            wind_speed_m_per_min * math.sin(wind_dir_rad))
    return _heading_vector_leg(p1, p2, wind, air_speed_m_per_min)


def _heading_vector_leg(p1, p2, wind, airspeed):
    """Preserve the frozen source's component values and operation order."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy)
    if not math.isfinite(length):
        raise ValueError("nonfinite geometry")
    if length == 0:
        return 0.0
    course, angle = math.atan2(dy, dx), math.atan2(wind[1], wind[0])
    heading = course - math.asin(math.hypot(*wind) / airspeed * math.sin(angle - course))
    vx, vy = airspeed * math.cos(heading) + wind[0], airspeed * math.sin(heading) + wind[1]
    along = vx * math.cos(course) + vy * math.sin(course)
    if along <= 0 or not math.isfinite(along):
        raise ValueError("no finite forward ground speed")
    return length / along


def _load_frozen(root: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load only the three required sources without changing sys.path.

    wind_scorer uses an absolute ``env.constant_audit`` import. A module-local
    import hook resolves only that dependency to this same root, without
    replacing global env modules or executing package initialization code.
    """
    frozen = root if (root / "env").is_dir() and (root / "baselines").is_dir() else root / "production" / "frozen"
    paths = [frozen / "env" / "constant_audit.py", frozen / "baselines" / "problem.py",
             frozen / "baselines" / "wind_scorer.py"]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("frozen root must contain env/constant_audit.py and baselines/{problem,wind_scorer}.py")
    lock_path = Path(__file__).resolve().parents[1] / "frozen_source_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    required = {"env/__init__.py", "env/constant_audit.py", "baselines/__init__.py",
                "baselines/problem.py", "baselines/wind_scorer.py"}
    if set(lock) != {"production/frozen/" + key for key in required}:
        raise ValueError("incomplete frozen source lock")
    for key in required:
        path = frozen / key
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != lock["production/frozen/" + key]:
            raise ValueError("frozen source lock mismatch")
    sources = [path.read_bytes() for path in paths]
    if any(hashlib.sha256(source).hexdigest() != lock["production/frozen/" + str(path.relative_to(frozen))]
           for path, source in zip(paths, sources)):
        raise ValueError("frozen source changed while binding")
    source_id = hashlib.sha256(str(frozen.resolve()).encode() + b"".join(sources)).hexdigest()
    prefix = "_track_a_frozen_" + source_id
    loaded = []
    try:
        for name, path, source in zip(("constant_audit", "problem", "wind_scorer"), paths, sources):
            full_name = prefix + "_" + name
            spec = importlib.util.spec_from_file_location(full_name, path)
            if spec is None or spec.loader is None:
                raise ImportError("missing frozen source loader")
            module = importlib.util.module_from_spec(spec)
            module.binding_source_sha256 = hashlib.sha256(source).hexdigest()
            sys.modules[full_name] = module  # Required while dataclass decorators run.
            loaded.append(full_name)
            if name == "wind_scorer":
                def local_import(module_name, globals=None, locals=None, fromlist=(), level=0):
                    if module_name == "env.constant_audit" and level == 0 and fromlist:
                        return modules[0]
                    return builtins.__import__(module_name, globals, locals, fromlist, level)
                module.__dict__["__builtins__"] = {**vars(builtins), "__import__": local_import}
                exec(compile(source, str(path), "exec"), module.__dict__)
            else:
                exec(compile(source, str(path), "exec"), module.__dict__)
            if name == "constant_audit":
                modules = [module]
            else:
                modules.append(module)
        return tuple(modules)
    finally:
        # Bindings retain their module objects, so no global import cache is needed.
        for name in loaded:
            sys.modules.pop(name, None)


class FrozenEvaluatorBinding:
    """An optional local frozen-source binding; no implicit remote execution."""

    def __init__(self, hn01_repo_root: Optional[str] = None) -> None:
        root = hn01_repo_root
        self.hn01_repo_root = str(Path(root).resolve()) if root else None
        self.mode = "STANDALONE_SYNTHETIC_ENGINEERING_SHIM"
        self._frozen_heading_leg = None
        self._frozen_validate_candidate = None
        self._frozen_problem_mod = None
        self._frozen_scorer_mod = None
        self.frozen_validation_calls = 0
        self.source_sha256 = {}
        if root:
            ca, problem, scorer = _load_frozen(Path(root).resolve())
            for mod, required in ((ca, ("heading_leg",)),
                                  (problem, ("Parameters", "Node", "Instance")),
                                  (scorer, ("score_solution", "validate_candidate", "CostAnchor", "physics_identity"))):
                if any(not callable(getattr(mod, name, None)) for name in required):
                    raise ImportError("incomplete frozen API")
            self._frozen_heading_leg = ca.heading_leg
            self._frozen_validate_candidate = scorer.validate_candidate
            self._frozen_problem_mod, self._frozen_scorer_mod = problem, scorer
            self.source_sha256 = {"env/constant_audit.py": ca.binding_source_sha256,
                                  "baselines/problem.py": problem.binding_source_sha256,
                                  "baselines/wind_scorer.py": scorer.binding_source_sha256}
            # A real-signature probe prevents fake alternate-signature hooks from
            # being labelled a compatible direct binding.
            if ca.heading_leg((0., 0.), (900., 0.), (0., 0.), 900.) != 1.0:
                raise ImportError("incompatible frozen heading API or units")
            self.mode = "DIRECT_FROZEN_IMPORT"

    @staticmethod
    def parameters(instance: Dict[str, Any]) -> Dict[str, float]:
        supplied = instance.get("parameters", {})
        if not isinstance(supplied, dict) or set(supplied) - set(DEFAULT_PARAMETERS):
            raise ValueError("invalid parameters schema")
        p = {**DEFAULT_PARAMETERS, **supplied}
        if not all(_finite(v, nonnegative=True) for v in p.values()):
            raise ValueError("parameters must be finite and nonnegative")
        for name in ("speed", "small_battery", "large_battery", "small_radius", "large_radius", "small_duration", "large_duration"):
            if p[name] <= 0:
                raise ValueError("physical parameters must be positive")
        if any(p["small_" + name] > p["large_" + name] for name in ("battery", "radius", "duration")):
            raise ValueError("small/large parameter ordering")
        return p

    def validate_instance(self, instance: Dict[str, Any]) -> Dict[str, float]:
        if not isinstance(instance, dict):
            raise ValueError("instance must be a dictionary")
        p = self.parameters(instance)
        wind = instance.get("wind_xy_mps")
        if not _xy(wind) or math.hypot(*(wind_mps_to_m_per_min(w) for w in wind)) >= p["speed"]:
            raise ValueError("finite wind below airspeed required")
        stations, towers = instance.get("stations"), instance.get("towers")
        if not isinstance(stations, dict) or not stations or not isinstance(towers, dict):
            raise ValueError("stations/towers dictionaries required")
        tower_ids = set()
        for key, tower in towers.items():
            try:
                tid = int(key)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("tower keys must be canonical integer strings") from exc
            if type(key) is not str or str(tid) != key or tid in tower_ids:
                raise ValueError("tower keys must be unique canonical integer strings")
            if (not isinstance(tower, dict) or not _xy(tower.get("xy_m"))
                    or not _finite(tower.get("energy"), nonnegative=True)
                    or not _finite(tower.get("t_srv_min"), nonnegative=True)):
                raise ValueError("tower coordinates, explicit energy and service must be finite and nonnegative")
            tower_ids.add(tid)
        assigned = []
        for sid, cfg in stations.items():
            if (type(sid) is not str or not sid or not isinstance(cfg, dict)
                    or cfg.get("station_type") not in STATION_TYPE_CODES or not _xy(cfg.get("xy_m"))):
                raise ValueError("invalid station schema")
            ids = cfg.get("assigned_towers")
            if not isinstance(ids, list) or any(type(tid) is not int or tid not in tower_ids for tid in ids):
                raise ValueError("assigned_towers must contain exact integer tower IDs")
            if cfg["station_type"] == "CLOSED" and ids:
                raise ValueError("closed station owns targets")
            assigned.extend(ids)
        if len(assigned) != len(set(assigned)) or set(assigned) != tower_ids:
            raise ValueError("target ownership must be a complete disjoint partition")
        return p

    def leg_minutes_xy(self, p1, p2, wind_xy_mps, *, airspeed_m_per_min=AIRSPEED_M_PER_MIN) -> float:
        if not _xy(p1) or not _xy(p2) or not _xy(wind_xy_mps) or not _finite(airspeed_m_per_min):
            raise ValueError("finite two-dimensional coordinates, wind and airspeed required")
        wind = tuple(wind_mps_to_m_per_min(w) for w in wind_xy_mps)
        speed = math.hypot(*wind)
        if speed >= airspeed_m_per_min:
            raise ValueError("wind must be strictly below positive airspeed")
        if self._frozen_heading_leg is not None:
            result = self._frozen_heading_leg(p1, p2, wind, airspeed_m_per_min)
        else:
            result = _heading_vector_leg(p1, p2, wind, airspeed_m_per_min)
        if not _finite(result, nonnegative=True):
            raise ValueError("nonfinite leg time")
        return result

    @staticmethod
    def cost_within_cap(cost: float, budget: float, tolerance: float = COST_TOL) -> bool:
        return all(_finite(x, nonnegative=True) for x in (cost, budget, tolerance)) and cost - budget <= tolerance

    def to_native_instance(self, instance):
        """Convert public fixture data to native objects, including stable identity."""
        if self._frozen_problem_mod is None:
            raise RuntimeError("native conversion requires an explicit frozen binding")
        p = self.validate_instance(instance)
        prob = self._frozen_problem_mod
        targets = tuple(prob.Node(int(key), *obj["xy_m"], obj["energy"], obj["t_srv_min"])
                        for key, obj in instance["towers"].items())
        used_ids = {node.id for node in targets}
        station_nodes = []
        next_id = min(used_ids | {0}) - 1
        for cfg in instance["stations"].values():
            station_nodes.append(prob.Node(next_id, *cfg["xy_m"], 0., 0.))
            next_id -= 1
        physical = {"wind_xy_mps": instance["wind_xy_mps"], "parameters": p,
                    "stations": instance["stations"], "towers": instance["towers"],
                    "station_order": list(instance["stations"]), "tower_order": list(instance["towers"])}
        raw = json.dumps(physical, sort_keys=True, separators=(",", ":"), allow_nan=False)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return prob.Instance("synthetic_track_a_fixture", targets, tuple(station_nodes), digest, prob.Parameters(**p))

    def to_native_solution(self, instance, solution, native_instance=None):
        if not isinstance(solution, dict) or set(solution) != set(instance["stations"]):
            raise ValueError("station key mismatch in native solution conversion")
        native = native_instance or self.to_native_instance(instance)
        wind = tuple(wind_mps_to_m_per_min(v) for v in instance["wind_xy_mps"])
        return {"schema_version": 1, "instance": native.name, "instance_sha256": native.sha256,
                "physics_sha256": self._frozen_scorer_mod.physics_identity(native, wind),
                "station_types": [STATION_TYPE_CODES[cfg["station_type"]] for cfg in instance["stations"].values()],
                "routes": [solution[sid] for sid in instance["stations"]]}

    def _evaluate_fixture(self, instance, solution):
        try:
            p = self.validate_instance(instance)
        except (TypeError, ValueError, OverflowError) as exc:
            return {"valid": False, "reason": "INVALID_INSTANCE:" + str(exc)}
        if not isinstance(solution, dict) or set(solution) != set(instance["stations"]):
            return {"valid": False, "reason": "STATION_KEY_MISMATCH"}
        fixed = total_flight = total_service = maximum = operating = 0.0
        visited, details = set(), {}
        for sid, cfg in instance["stations"].items():
            kind, depot = cfg["station_type"], cfg["xy_m"]
            small = kind == "SABS"
            fixed_here = 0. if kind == "CLOSED" else (p["small_station_cost"] + p["small_drone_cost"] if small else p["large_station_cost"] + p["small_drone_cost"] + p["large_drone_cost"])
            fixed += fixed_here
            slots = solution[sid]
            if not isinstance(slots, list) or len(slots) != 2 or any(not isinstance(trips, list) for trips in slots):
                return {"valid": False, "reason": "TWO_SLOT_SCHEMA_REQUIRED:" + sid}
            if small and any(trip for trip in slots[0]):
                return {"valid": False, "reason": "SABS_SLOT0_NONEMPTY:" + sid}
            duties, flights, energies = [], [], []
            for slot, trips in enumerate(slots):
                cap = p["large_battery"] if slot == 0 else p["small_battery"]
                duty = flight = 0.
                sortie_energies = []
                for trip in trips:
                    if not isinstance(trip, list):
                        return {"valid": False, "reason": f"TRIP_SCHEMA_REQUIRED:{sid}:slot{slot}"}
                    if not trip:  # Matches original frozen semantics.
                        continue
                    if kind == "CLOSED":
                        return {"valid": False, "reason": "CLOSED_STATION_NONEMPTY:" + sid}
                    current = depot
                    leg_times, services, service_energies = [], [], []
                    for tid in trip:
                        if type(tid) is not int:
                            return {"valid": False, "reason": "EXACT_INTEGER_TOWER_ID_REQUIRED"}
                        if tid not in cfg["assigned_towers"]:
                            return {"valid": False, "reason": f"UNASSIGNED_TOWER:{sid}:{tid}"}
                        if tid in visited:
                            return {"valid": False, "reason": f"DUPLICATE_TOWER:{tid}"}
                        visited.add(tid)
                        tower = instance["towers"][str(tid)]
                        radius = p["small_radius"] if small else p["large_radius"]
                        if math.hypot(tower["xy_m"][0] - depot[0], tower["xy_m"][1] - depot[1]) > radius + COVERAGE_TOL_M:
                            return {"valid": False, "reason": f"COVERAGE_EXCEEDED:{sid}:{tid}"}
                        try:
                            dt = self.leg_minutes_xy(current, tower["xy_m"], instance["wind_xy_mps"], airspeed_m_per_min=p["speed"])
                        except ValueError:
                            return {"valid": False, "reason": "NONFINITE_GEOMETRY"}
                        leg_times.append(dt)
                        services.append(tower["t_srv_min"])
                        service_energies.append(tower["energy"])
                        current = tower["xy_m"]
                    try:
                        dt = self.leg_minutes_xy(current, depot, instance["wind_xy_mps"], airspeed_m_per_min=p["speed"])
                    except ValueError:
                        return {"valid": False, "reason": "NONFINITE_GEOMETRY"}
                    travel = sum(leg_times + [dt])
                    service = sum(services)
                    energy = travel * p["flight_energy"] + sum(service_energies)
                    if not _finite(energy, nonnegative=True) or energy > cap + ENERGY_TOL:
                        return {"valid": False, "reason": f"BATTERY_EXCEEDED:{sid}:slot{slot}({FROZEN_SLOT_MODELS[slot]}):{energy:.12f}>{cap:.2f}"}
                    duty += travel + service
                    flight += travel
                    total_flight += travel
                    total_service += service
                    sortie_energies.append(energy)
                limit = p["small_duration"] if small else p["large_duration"]
                if not _finite(duty, nonnegative=True) or duty > limit + PHYSICS_TOL:
                    return {"valid": False, "reason": f"CUMULATIVE_DUTY_EXCEEDED:{sid}:slot{slot}"}
                duties.append(duty)
                operating += duty
                flights.append(flight)
                energies.append(sortie_energies)
            maximum = max(maximum, *duties)
            details[sid] = {"station_type": kind, "fixed_cost_rmb": fixed_here,
                            "slot_models": list(FROZEN_SLOT_MODELS), "slot_duties_min": duties,
                            "slot_flights_min": flights, "slot_sortie_energies": energies}
        if visited != {int(key) for key in instance["towers"]}:
            return {"valid": False, "reason": "INCOMPLETE_TOWER_COVERAGE"}
        total_duty = operating
        cost = fixed + p["time_cost"] * total_duty
        if not _finite(cost, nonnegative=True):
            return {"valid": False, "reason": "NONFINITE_TOTAL_COST"}
        return {"valid": True, "reason": "VALID", "makespan_min": maximum,
                "flight_min": total_flight, "service_min": total_service,
                "total_duty_min": total_duty, "fixed_cost_rmb": fixed,
                "raw_cost_rmb": cost, "per_station": details}

    def evaluate_solution(self, instance, solution):
        result = self._evaluate_fixture(instance, solution)
        if result["valid"] and self._frozen_scorer_mod is not None:
            native = self.to_native_instance(instance)
            sol = self.to_native_solution(instance, solution, native)
            wind = tuple(wind_mps_to_m_per_min(v) for v in instance["wind_xy_mps"])
            score = self._frozen_scorer_mod.score_solution(native, sol, wind=wind)
            if not score["original_feasible"]:
                return {"valid": False, "reason": "FROZEN_REJECT:" + ",".join(score["issues"]), "frozen_score": score}
            result.update(raw_cost_rmb=score["cost"], makespan_min=score["M_return"],
                          total_duty_min=score["operating_minutes"], fixed_cost_rmb=score["fixed_cost"])
            result["frozen_score"] = score
        return result

    def validate_candidate(self, instance, solution, anchor, wall_elapsed_s=0.0, wall_limit_s=6.0):
        """Full-map cap; caller elapsed plus validator time includes certification.

        Elapsed arguments are engineering fixture inputs. A real caller must
        measure all work before entering this function, including construction.
        """
        started = perf_counter()
        if not all(_finite(value, nonnegative=True) for value in (wall_elapsed_s, wall_limit_s)):
            return {"admissible": False, "reason": "INVALID_WALL_CLOCK"}
        anc = self.evaluate_solution(instance, anchor)
        if not anc["valid"]:
            raise ValueError("Anchor/reference solution itself is invalid: " + anc["reason"])
        cand = self.evaluate_solution(instance, solution)
        common = {"anchor_cost_rmb": anc["raw_cost_rmb"], "anchor_makespan_min": anc["makespan_min"]}
        if not cand["valid"]:
            result = {"admissible": False, "reason": cand["reason"], **common}
        else:
            admissible = self.cost_within_cap(cand["raw_cost_rmb"], anc["raw_cost_rmb"])
            native_result = None
            if self._frozen_scorer_mod is not None:
                native = self.to_native_instance(instance)
                wind = tuple(wind_mps_to_m_per_min(v) for v in instance["wind_xy_mps"])
                native_anchor = self._frozen_scorer_mod.CostAnchor.from_solution(
                    native, self.to_native_solution(instance, anchor, native), wind=wind)
                native_result = self._frozen_validate_candidate(native, self.to_native_solution(instance, solution, native), native_anchor)
                self.frozen_validation_calls += 1
                admissible = native_result["admissible"]
            result = {"admissible": admissible, "reason": "ADMISSIBLE" if admissible else "REJECT_COST_CAP_FROZEN_SUBTRACT",
                      **common, "raw_cost_rmb": cand["raw_cost_rmb"],
                      "cost_diff_rmb": cand["raw_cost_rmb"] - anc["raw_cost_rmb"],
                      "makespan_min": cand["makespan_min"], "flight_min": cand["flight_min"],
                      "service_min": cand["service_min"], "total_duty_min": cand["total_duty_min"],
                      "per_station": cand["per_station"]}
            if native_result is not None:
                result["frozen_validation"] = native_result
        if wall_elapsed_s + perf_counter() - started > wall_limit_s:
            return {"admissible": False, "reason": "WALL_CLOCK_EXCEEDED_AT_VALIDATOR_FINISH", **common}
        return result


def _check_station_types(station_types):
    if not isinstance(station_types, dict) or any(kind not in STATION_TYPE_CODES for kind in station_types.values()):
        raise ValueError("invalid station types")


def model_keyed_to_frozen_solution(station_types, by_model):
    _check_station_types(station_types)
    if not isinstance(by_model, dict) or set(by_model) != set(station_types):
        raise ValueError("station key mismatch")
    out = {}
    for sid, kind in station_types.items():
        models = by_model[sid]
        if not isinstance(models, dict) or set(models) - set(FROZEN_SLOT_MODELS):
            raise ValueError("unknown model key")
        slots = [models.get("LD", []), models.get("SD", [])]
        out[sid] = _copy_slots(kind, slots)
    return out


def _copy_slots(kind, slots):
    if not isinstance(slots, list) or len(slots) != 2:
        raise ValueError("two-slot schema required")
    copied = []
    for trips in slots:
        if not isinstance(trips, list) or any(not isinstance(trip, list) or any(type(tid) is not int for tid in trip) for trip in trips):
            raise ValueError("exact integer IDs and list trip schema required")
        copied.append([list(trip) for trip in trips])
    if kind == "SABS" and any(copied[0]):
        raise ValueError("SABS slot 0 must be empty")
    if kind == "CLOSED" and any(trip for trips in copied for trip in trips):
        raise ValueError("closed station must have empty routes")
    return copied


def frozen_solution_to_model_keyed(station_types, frozen_sol):
    _check_station_types(station_types)
    if not isinstance(frozen_sol, dict) or set(frozen_sol) != set(station_types):
        raise ValueError("station key mismatch")
    out = {}
    for sid, kind in station_types.items():
        ld, sd = _copy_slots(kind, frozen_sol[sid])
        out[sid] = {"LD": ld, "SD": sd}
    return out
