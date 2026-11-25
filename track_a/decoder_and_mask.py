"""Track A Unified Policy Action Mask, Canonical State Key & Equivalence Classes.

Strictly adheres to v5 Admission Review §2.A.4:
1. Slot indices follow frozen wind_scorer.py everywhere: 0 = LD (1.00), 1 = SD (0.81).
   On SABS, slot 0 is permanently inactive (SABS serializes as [[], SD_trips]).
2. Single unified policy action mask: ALL physically & makespan-admissible (slot_idx, target)
   pairs are simultaneously valid in the policy mask (no 'min-workload-slot-first' lock).
3. Unproven flight-time hard pruning (FLIGHT_LB) is DISABLED for hard deletion; flight lower bound
   (F_acc + f_rem_lb) is computed strictly for candidate ordering / tie-breaking.
4. Truncated (k1=1) and dead-end (k1=2) witnesses are evaluated in a separate trajectory harness
   and NEVER submitted to validate_candidate as complete routes.
"""
from __future__ import annotations

from collections import deque
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .frozen_adapter import ENERGY_TOL, PHYSICS_TOL, COVERAGE_TOL_M, FrozenEvaluatorBinding, _finite

# Frozen rejection-reason priority order:
REJECTION_PRIORITY: Tuple[str, ...] = (
    "INACTIVE_SABS_SLOT0",
    "EMPTY_RETURN",
    "ALREADY_VISITED",
    "BATT_RET",
    "BATT_CAP",
    "MAKESPAN_LB",
)

StateTuple = Tuple[
    Tuple[int, ...],          # U: sorted unvisited assigned tower IDs
    Tuple[int, int],          # p: current node per slot (0 = depot)
    Tuple[float, float],      # e: current sortie accumulated energy per slot
    Tuple[float, float],      # W: accumulated duty minutes per slot
    float,                    # F_acc: accumulated flight minutes across station
]


def canonical_state_key(st: StateTuple) -> Tuple[Any, ...]:
    """Exact binary64 state key; a tolerance never justifies merging states.

    Rounding can merge prefixes on opposite sides of a future energy boundary.
    The resource values used by the mask therefore remain unrounded.
    """
    u_rem, pos, eng, duty, f_acc = st
    return (
        u_rem,
        pos,
        eng,
        duty,
        f_acc,
    )


class StationCoSortieMDP:
    """Single-station co-sortie MDP with unified policy mask and zero hard flight pruning."""

    def __init__(
        self,
        instance: Dict[str, Any],
        station_id: str,
        global_makespan_cap_min: float,
        binding: Optional[FrozenEvaluatorBinding] = None,
    ) -> None:
        self.binding = binding or FrozenEvaluatorBinding()
        self.params = self.binding.validate_instance(instance)
        if (type(global_makespan_cap_min) not in (int, float)
                or not math.isfinite(global_makespan_cap_min) or global_makespan_cap_min < 0):
            raise ValueError("makespan cap must be finite and nonnegative")
        self.instance = instance
        self.station_id = station_id
        self.s_cfg = instance["stations"][station_id]
        self.station_type = self.s_cfg["station_type"]
        self.global_makespan_cap_min = float(global_makespan_cap_min)
        self.assigned_towers: Tuple[int, ...] = tuple(sorted(self.s_cfg["assigned_towers"]))
        if 0 in self.assigned_towers:
            raise ValueError("target ID 0 conflicts with the public RETURN action sentinel")
        self.slot_caps = (self.params["large_battery"], self.params["small_battery"])
        self.duty_limit = self.params["small_duration"] if self.station_type == "SABS" else self.params["large_duration"]
        self.radius = self.params["small_radius"] if self.station_type == "SABS" else self.params["large_radius"]
        self.coords: Dict[int, Tuple[float, float]] = {0: tuple(self.s_cfg["xy_m"])}
        self.srv_min: Dict[int, float] = {}
        self.srv_energy: Dict[int, float] = {}

        for u in self.assigned_towers:
            t_obj = instance["towers"][str(u)]
            if "energy" not in t_obj:
                raise KeyError(f"Tower {u} missing explicit 'energy' field")
            self.coords[u] = tuple(t_obj["xy_m"])
            self.srv_min[u] = float(t_obj["t_srv_min"])
            self.srv_energy[u] = float(t_obj["energy"])

        wind_xy = tuple(instance["wind_xy_mps"])
        nodes = (0,) + self.assigned_towers
        self.tau: Dict[Tuple[int, int], float] = {
            (i, j): self.binding.leg_minutes_xy(self.coords[i], self.coords[j], wind_xy,
                                              airspeed_m_per_min=self.params["speed"])
            for i in nodes
            for j in nodes
        }

    def initial_state(self) -> StateTuple:
        return (self.assigned_towers, (0, 0), (0.0, 0.0), (0.0, 0.0), 0.0)

    def active_slots(self) -> Tuple[int, ...]:
        """0 = LD, 1 = SD; SABS only activates slot 1 (slot 0 stays empty)."""
        return () if self.station_type == "CLOSED" else (1,) if self.station_type == "SABS" else (0, 1)

    def remaining_flight_lb(self, u_rem: Tuple[int, ...], pos: Tuple[int, int]) -> float:
        ret_sum = sum(self.tau[(p, 0)] for p in pos if p != 0)
        if not u_rem:
            return ret_sum
        best_single_insertion = max(
            min(
                [self.tau[(0, u)] + self.tau[(u, 0)]]
                + [self.tau[(p, u)] + self.tau[(u, 0)] - self.tau[(p, 0)] for p in pos if p != 0]
            )
            for u in u_rem
        )
        return ret_sum + best_single_insertion

    def makespan_and_flight_lb(
        self, u_rem: Tuple[int, ...], pos: Tuple[int, int], duty: Tuple[float, float]
    ) -> Tuple[float, float]:
        f_rem = self.remaining_flight_lb(u_rem, pos)
        act_slots = self.active_slots()
        if not act_slots:
            return 0.0, 0.0
        max_ret = max(duty[d] + self.tau[(pos[d], 0)] for d in act_slots)
        n_active = float(len(act_slots))
        avg_work = (
            sum(duty[d] for d in act_slots) + sum(self.srv_min[u] for u in u_rem) + f_rem
        ) / n_active
        return max(max_ret, avg_work), f_rem

    def step(self, st: StateTuple, action: Tuple[int, int]) -> Tuple[Optional[StateTuple], str]:
        """Evaluate (slot_idx, target) under frozen rejection priority and ENERGY_TOL = 1e-12."""
        if (not isinstance(action, (tuple, list)) or len(action) != 2
                or any(type(x) is not int for x in action) or action[0] not in (0, 1)):
            return None, "INVALID_ACTION"
        if not self._state_valid(st):
            return None, "INVALID_STATE"
        u_rem, pos, eng, duty, f_acc = st
        slot_idx, target = action

        if self.station_type == "CLOSED":
            return None, "INACTIVE_CLOSED_STATION"

        if self.station_type == "SABS" and slot_idx == 0:
            return None, "INACTIVE_SABS_SLOT0"

        b_cap = self.slot_caps[slot_idx]
        cur = pos[slot_idx]

        if target == 0:
            if cur == 0:
                return None, "EMPTY_RETURN"
            dt = self.tau[(cur, 0)]
            if eng[slot_idx] + self.params["flight_energy"] * dt > b_cap + ENERGY_TOL:
                return None, "BATT_RET"
            p2 = list(pos)
            p2[slot_idx] = 0
            e2 = list(eng)
            e2[slot_idx] = 0.0
            w2 = list(duty)
            w2[slot_idx] += dt
            f2 = f_acc + dt
            u2 = u_rem
        else:
            if target not in u_rem:
                return None, "ALREADY_VISITED"
            if math.hypot(self.coords[target][0] - self.coords[0][0],
                          self.coords[target][1] - self.coords[0][1]) > self.radius + COVERAGE_TOL_M:
                return None, "COVERAGE_EXCEEDED"
            dt = self.tau[(cur, target)]
            e_after = eng[slot_idx] + self.params["flight_energy"] * dt + self.srv_energy[target]
            e_with_ret = e_after + self.params["flight_energy"] * self.tau[(target, 0)]
            if e_with_ret > b_cap + ENERGY_TOL:
                return None, "BATT_CAP"
            p2 = list(pos)
            p2[slot_idx] = target
            e2 = list(eng)
            e2[slot_idx] = e_after
            w2 = list(duty)
            w2[slot_idx] += dt + self.srv_min[target]
            f2 = f_acc + dt
            u2 = tuple(x for x in u_rem if x != target)

        if w2[slot_idx] + self.tau[(p2[slot_idx], 0)] > self.duty_limit + PHYSICS_TOL:
            return None, "CUMULATIVE_DUTY"
        mlb, _ = self.makespan_and_flight_lb(u2, (p2[0], p2[1]), (w2[0], w2[1]))
        if (mlb - self.global_makespan_cap_min) > ENERGY_TOL:
            return None, "MAKESPAN_LB"

        return (u2, (p2[0], p2[1]), (e2[0], e2[1]), (w2[0], w2[1]), f2), "OK"

    def _state_valid(self, st) -> bool:
        if not isinstance(st, tuple) or len(st) != 5:
            return False
        u_rem, pos, eng, duty, f_acc = st
        if (not isinstance(u_rem, tuple) or any(type(u) is not int or u not in self.assigned_towers for u in u_rem)
                or tuple(sorted(set(u_rem))) != u_rem):
            return False
        if (not isinstance(pos, tuple) or len(pos) != 2
                or any(type(p) is not int or p not in self.coords or p in u_rem for p in pos)):
            return False
        if any(not isinstance(pair, tuple) or len(pair) != 2 for pair in (eng, duty)):
            return False
        if any(not _finite(x, nonnegative=True) for x in (*eng, *duty, f_acc)):
            return False
        if any(eng[d] > self.slot_caps[d] + ENERGY_TOL or duty[d] > self.duty_limit + PHYSICS_TOL for d in (0, 1)):
            return False
        return all(pos[d] == 0 and eng[d] == 0 and duty[d] == 0 for d in (0, 1) if d not in self.active_slots())

    def probe_candidates_deterministic(self, st: StateTuple) -> Tuple[List[Tuple[int, int]], Dict[str, str]]:
        """Deterministic probe traversal order: slot_idx ASC over (0, 1), target ASC over [0] + list(U).

        Note: Policy action mask contains ALL valid actions returned in `valid_actions`;
        flight lower bound is NEVER used for hard rejection (only ranking if needed).
        """
        u_rem = st[0]
        valid_actions: List[Tuple[int, int]] = []
        rejections: Dict[str, str] = {}
        for slot_idx in (0, 1):
            for target in (0,) + u_rem:
                ns, rsn = self.step(st, (slot_idx, target))
                if ns is not None:
                    valid_actions.append((slot_idx, target))
                else:
                    rejections[f"({slot_idx},{target})"] = rsn
        return valid_actions, rejections

    def rank_valid_actions_by_soft_skyline(self, st: StateTuple) -> List[Tuple[Tuple[int, int], float, float]]:
        """Soft ordering key (mlb ASC, F_acc + f_rem_lb ASC, slot_idx ASC, target ASC) — ZERO hard deletion."""
        valid_actions, _ = self.probe_candidates_deterministic(st)
        scored = []
        for act in valid_actions:
            ns, _ = self.step(st, act)
            assert ns is not None
            mlb, frem = self.makespan_and_flight_lb(ns[0], ns[1], ns[3])
            scored.append((act, mlb, ns[4] + frem))
        scored.sort(key=lambda item: (round(item[1], 12), round(item[2], 12), item[0][0], item[0][1]))
        return scored

    def actions_to_frozen_station_routes(self, actions: Sequence[Tuple[int, int]]) -> List[List[List[int]]]:
        """Convert complete closed trajectory into frozen 2-slot [[LD_trips], [SD_trips]]."""
        slots: List[List[List[int]]] = [[], []]
        open_trip: List[List[int]] = [[], []]
        st = self.initial_state()
        for act in actions:
            ns, reason = self.step(st, act)
            if ns is None:
                raise ValueError(f"Illegal physical action {act}: {reason}")
            st = ns
            slot_idx, target = act
            if target == 0:
                slots[slot_idx].append(list(open_trip[slot_idx]))
                open_trip[slot_idx] = []
            else:
                open_trip[slot_idx].append(target)
        if st[0] or st[1] != (0, 0) or open_trip[0] or open_trip[1]:
            raise ValueError("Incomplete/unclosed trajectory cannot be serialized as complete solution!")
        return slots

    def evaluate_trajectory_witness(
        self, actions: Sequence[Tuple[int, int]], truncated: bool = False
    ) -> Dict[str, Any]:
        """Separate witness harness for complete (k1=0), truncated (k1=1), and dead-end (k1=2) trajectories."""
        st = self.initial_state()
        n_active_decision_steps = 0
        for raw_act in actions:
            if not isinstance(raw_act, (tuple, list)) or len(raw_act) != 2 or any(type(v) is not int for v in raw_act):
                raise ValueError("exact integer physical action required")
            act = tuple(raw_act)
            valids, rejs = self.probe_candidates_deterministic(st)
            if act not in valids:
                raise AssertionError(f"Trajectory action {act} rejected ({rejs.get(f'({act[0]},{act[1]})')})")
            if len(valids) >= 2:
                n_active_decision_steps += 1
            ns, _ = self.step(st, act)
            assert ns is not None
            st = ns

        term_valids, term_rejs = self.probe_candidates_deterministic(st)
        u_rem, pos, _, duty, f_acc = st
        mlb, frem = self.makespan_and_flight_lb(u_rem, pos, duty)
        n_unret = sum(1 for p in pos if p != 0)
        if not u_rem and n_unret == 0 and not truncated:
            k1 = 0
        elif truncated and len(term_valids) >= 1:
            k1 = 1
        else:
            if len(term_valids) != 0:
                raise AssertionError(f"Expected dead-end (0 valid actions), found {term_valids}")
            k1 = 2
        return {
            "k1": k1,
            "k2": 2 * len(u_rem) + n_unret,
            "v3_makespan_lb_min": round(mlb, 12),
            "v4_flight_lb_min": round(f_acc + frem, 12),
            "n_active_steps": n_active_decision_steps,
            "terminal_valid_count": len(term_valids),
            "terminal_rejections": term_rejs if k1 == 2 else {},
            "can_submit_to_validator": (k1 == 0),
        }

    def enumerate_canonical_state_space(self) -> Dict[str, Any]:
        """Exact BFS census of reachable canonical states under the frozen unified mask."""
        init = self.initial_state()
        seen = {canonical_state_key(init): init}
        q = deque([init])
        complete_keys = []
        dead_end_keys = []
        while q:
            st = q.popleft()
            if not st[0] and st[1] == (0, 0):
                complete_keys.append(canonical_state_key(st))
                continue
            valids, _ = self.probe_candidates_deterministic(st)
            if not valids:
                dead_end_keys.append(canonical_state_key(st))
                continue
            for act in valids:
                ns, _ = self.step(st, act)
                assert ns is not None
                ck = canonical_state_key(ns)
                if ck not in seen:
                    seen[ck] = ns
                    q.append(ns)
        return {
            "reachable_canonical_states": len(seen),
            "complete_terminal_states": len(complete_keys),
            "dead_end_states": len(dead_end_keys),
            "complete_keys": complete_keys,
        }


def canonicalize_station_solution_equivalence(
    station_slots: Sequence[Sequence[Sequence[int]]],
) -> Dict[str, Any]:
    """Static closed-tour equivalence, not an online policy-state equivalence:
    1. ordered_representation: exact [[LD_trips], [SD_trips]]
    2. intra_slot_sortie_multiset: sorts closed sorties within each slot (since sortie order on same drone commutes for total duty, per-sortie energy, and total cost)
    3. undirected_sortie_partition: additionally canonicalizes reversible symmetric loops (min(trip, trip[::-1]))

    These are algebraic grouping keys. Binary64 summation order can change a
    literal scorer boundary, so every delivered route still requires its own
    full certification; the keys do not authorize feasibility or cap deduping.
    """
    if (not isinstance(station_slots, (list, tuple)) or len(station_slots) != 2
            or any(not isinstance(slot, (list, tuple)) or any(not isinstance(tr, (list, tuple))
                   or any(type(x) is not int for x in tr) for tr in slot) for slot in station_slots)):
        raise ValueError("exact two-slot integer-ID schema required")
    ordered = tuple(tuple(tuple(tr) for tr in slot) for slot in station_slots)
    multiset_key = tuple(tuple(sorted(slot)) for slot in ordered)
    partition_key = tuple(
        tuple(sorted(min(tr, tr[::-1]) for tr in slot))
        for slot in ordered
    )
    return {
        "ordered_key": ordered,
        "intra_slot_sortie_multiset_key": multiset_key,
        "undirected_partition_key": partition_key,
    }
