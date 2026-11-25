"""Track B Common Non-Learning CAPE Public Executor & Shared Witness Completer.

Engineering prototype for the synthetic v5 Admission Review §2.B fixtures.
This is not a trained policy, a production CAPE implementation, or scientific evidence.
1. Frozen 8-row physical mode transition table: every drone starts on ground at t=0
   with epoch=0; EVERY physical mode transition (including t=0 takeoff GROUND_READY -> FLY_*)
   increments drone_epoch += 1 (so B3 Sortie 1 takeoff/arrive/depart/land produces epoch=4).
2. Explicit separation between `tower_catalog` (all physical towers eligible for urgent
   re-inspection tickets within 4km/5km station coverage) and `mandatory_routine_tasks`
   (this shift's required patrol tasks, verified 100% completed independently at SHIFT_END).
3. Physical action interface in `SharedWitnessCompleter`: attributable policy actions
   map strictly to `(station_id, slot_idx, target_or_RETURN)` (0=LD, 1=SD, SABS slot 0 empty);
   `SharedWitnessCompleter` only completes a deterministic residual witness after the policy's
   explicit next physical waypoint, with unique tie-breaking, explicit `NO_FEASIBLE_WITNESS`,
   and end-to-end wall-clock measurement (`snapshot -> enumerate -> mask -> complete -> CAS`).
4. Executor recursively rejects `expected*` keys (blind execution).
"""
from __future__ import annotations

import copy
import itertools
import math
import time
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence, Tuple

STATION_RULES_B: Dict[str, Dict[str, Any]] = {
    "DABS": {"slot_models": ("LD", "SD"), "coverage_radius_m": 5000.0},
    "SABS": {"slot_models": ("NONE", "SD"), "coverage_radius_m": 4000.0},
}
DRONE_BATTERY_CAPS: Dict[str, Fraction] = {
    "LD": Fraction(1, 1),
    "SD": Fraction(81, 100),
}


def reject_expected_keys(value: Any, path: str = "inputs") -> None:
    """Reject expected-output contamination at every nesting level."""
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower().startswith("expected"):
                raise ValueError(f"Blind input violation: {path}.{key}")
            reject_expected_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            reject_expected_keys(child, f"{path}[{i}]")


def _ms(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return value


def _indexed(rows: Any, key: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{key} rows must be a list")
    out = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get(key), str) or not row[key] or row[key] in out:
            raise ValueError(f"Missing or duplicate {key}")
        out[row[key]] = row
    return out

# Frozen 8-row physical mode transition table (every transition increments epoch += 1):
FROZEN_MODE_TRANSITION_TABLE: Tuple[Dict[str, str], ...] = (
    {"row": "T1", "from": "GROUND_READY", "to": "FLY_ROUTINE|FLY_OUT", "trigger": "Takeoff at t >= 0 (t_dep == now)"},
    {"row": "T2", "from": "GROUND_IDLE", "to": "GROUND_READY", "trigger": "CAS arms future sortie (t_dep > now)"},
    {"row": "T3", "from": "GROUND_IDLE", "to": "FLY_ROUTINE|FLY_OUT", "trigger": "CAS launches immediate sortie (t_dep == now)"},
    {"row": "T4", "from": "FLY_ROUTINE|FLY_OUT", "to": "SERVING", "trigger": "Arrive at tower stop"},
    {"row": "T5", "from": "SERVING", "to": "HOVER", "trigger": "Service ends with hover_timeout>0 & hold_rem>0 & B_liquid>0"},
    {"row": "T6", "from": "SERVING", "to": "FLY_ROUTINE|FLY_OUT", "trigger": "Service ends & departs immediately without hover"},
    {"row": "T7", "from": "HOVER", "to": "FLY_ROUTINE|FLY_OUT", "trigger": "Hover hold alarm, global budget alarm, or CAS continuation"},
    {"row": "T8", "from": "FLY_ROUTINE|FLY_OUT", "to": "GROUND_READY|GROUND_IDLE", "trigger": "Return to home station & battery swap"},
)


def quantize_flight_ms(p1: Sequence[float], p2: Sequence[float]) -> int:
    """Canonical 1 ms ceiling flight quantization: ceil(200 * hypot(dx, dy) / 3)."""
    dx = float(p2[0]) - float(p1[0])
    dy = float(p2[1]) - float(p1[1])
    if dx == 0.0 and dy == 0.0:
        return 0
    dist_sq = int(round(dx)) ** 2 + int(round(dy)) ** 2
    root = math.isqrt(dist_sq)
    if root * root == dist_sq and dx == round(dx) and dy == round(dy):
        return (200 * root + 2) // 3
    dist = math.hypot(dx, dy)
    return int(math.ceil((200.0 * dist) / 3.0))


class OriginalDomainCapeExecutor:
    """Local deterministic physical simulator for the synthetic acceptance fixtures."""

    def __init__(self, blind_trace_spec: Dict[str, Any]) -> None:
        reject_expected_keys(blind_trace_spec)

        self.spec = copy.deepcopy(blind_trace_spec)
        init = self.spec["initial_state"]
        self.t_ms = _ms(init["t_start_ms"], "t_start_ms")
        self.shift_end_ms = _ms(init.get("shift_end_ms", 6_000_000), "shift_end_ms")
        if self.t_ms >= self.shift_end_ms:
            raise ValueError("shift_end_ms must follow t_start_ms")

        # Exact rational rates per millisecond
        self.c_fly = Fraction(1, 60_000)       # 1.0 RMB/min
        self.c_srv = Fraction(1, 300_000)      # 0.2 RMB/min
        self.c_hover = Fraction(1, 120_000)    # 0.5 RMB/min
        self.r_fly = Fraction(9, 30_000_000)   # 0.018 /min
        self.r_srv = Fraction(1, 3_000_000)    # 0.020 /min
        self.r_hover = Fraction(1, 3_000_000)  # 0.020 /min

        self.stations = _indexed(init["stations"], "station_id")
        self.tower_catalog = _indexed(init["tower_catalog"], "node_id")
        if not self.stations or set(self.stations) & set(self.tower_catalog):
            raise ValueError("Station/tower identifiers must be disjoint and stations nonempty")
        self.mandatory_routine_tasks = _indexed(init["mandatory_routine_tasks"], "task_id")
        # Explicit task identity preserves multiple jobs at one physical tower.
        self.mandatory_tasks_by_station_node: Dict[Tuple[str, str], List[str]] = {}
        self.mandatory_by_station_node: Dict[Tuple[str, str], str] = {}
        for tid, m in self.mandatory_routine_tasks.items():
            key = (m["station_id"], m["node_id"])
            if key[0] not in self.stations or key[1] not in self.tower_catalog:
                raise ValueError(f"Unknown mandatory task station/tower: {tid}")
            _ms(m["srv_ms"], "mandatory srv_ms")
            self.mandatory_tasks_by_station_node.setdefault(key, []).append(tid)
        self.mandatory_by_station_node = {key: tids[0] for key, tids in self.mandatory_tasks_by_station_node.items() if len(tids) == 1}
        self.completed_mandatory_tasks: set[str] = set()

        self.nodes: Dict[str, Tuple[float, float]] = {}
        for s_id, s in self.stations.items():
            if s["station_type"] not in STATION_RULES_B:
                raise ValueError(f"Unknown station_type: {s['station_type']}")
            self.nodes[s_id] = (float(s["x_m"]), float(s["y_m"]))
        for n_id, t in self.tower_catalog.items():
            self.nodes[n_id] = (float(t["x_m"]), float(t["y_m"]))
        if any(not math.isfinite(v) for xy in self.nodes.values() for v in xy):
            raise ValueError("Coordinates must be finite")

        self.B_cost = Fraction(init["budget_total_yuan"])
        if self.B_cost < 0:
            raise ValueError("budget_total_yuan must be nonnegative")
        self.B_spent = Fraction(0)
        self.B_witness = Fraction(0)
        self.B_hover = Fraction(0)
        self.B_free = Fraction(0)
        self.global_budget_alarm_ms: Optional[int] = None
        self.ledger_ver = 0
        self.tickets: Dict[str, Dict[str, Any]] = {}
        self.urgent_pub = 0
        self.urgent_scored = 0
        self.hard_violations = 0
        self.logs: List[Dict[str, Any]] = []

        # Validate 2-slot convention (0=LD, 1=SD; SABS only has slot_idx=1 SD)
        station_slots_seen: Dict[str, Dict[int, str]] = {s_id: {} for s_id in self.stations}
        self.drones: Dict[str, Dict[str, Any]] = {}
        covered_mandatory_at_init: set[str] = set()

        for d in _indexed(init["drones"], "drone_id").values():
            s_id = d["station_id"]
            stype = self.stations[s_id]["station_type"]
            slot_idx = _ms(d["slot_idx"], "slot_idx")
            if slot_idx not in (0, 1):
                raise ValueError("slot_idx must be 0 or 1")
            expected_model = STATION_RULES_B[stype]["slot_models"][slot_idx]
            if expected_model == "NONE" or d["model"] != expected_model:
                raise ValueError(
                    f"Slot convention violation on {s_id} ({stype}): slot_idx={slot_idx} "
                    f"must be {expected_model}, got {d['model']}"
                )
            if slot_idx in station_slots_seen[s_id]:
                raise ValueError(f"Duplicate slot_idx {slot_idx} on station {s_id}")
            station_slots_seen[s_id][slot_idx] = d["drone_id"]

            r_cov = STATION_RULES_B[stype]["coverage_radius_m"]
            sorties = copy.deepcopy(d["initial_witness_sorties"])
            init_mode = "GROUND_READY" if sorties else "GROUND_IDLE"
            init_step_end = max(self.t_ms, int(sorties[0]["t_arm_ms"])) if sorties else None

            drone = {
                "id": d["drone_id"],
                "station_id": s_id,
                "slot_idx": slot_idx,
                "model": d["model"],
                "r_cov_m": r_cov,
                "b_cap": DRONE_BATTERY_CAPS[d["model"]],
                "epoch": _ms(d.get("initial_epoch", 0), "initial_epoch"),
                "witness_ver": d["initial_witness_ver"],
                "mode": init_mode,
                "pos_node": s_id,
                "battery": Fraction(0),
                "sorties": sorties,
                "unserved_routine": set(),
                "unserved_tickets": set(),
                "onboard_served_tickets": [],
                "step_end_ms": init_step_end,
                "hover_limit_ms": 0,
                "hover_entered_ms": None,
            }

            if drone["epoch"] != 0:
                raise ValueError("Every initial drone must start at epoch 0")
            error, _, routine_tasks = self._validate_sorties(drone, sorties, initial=True)
            if error:
                raise ValueError(f"Invalid initial witness: {error}")
            if covered_mandatory_at_init & routine_tasks:
                raise ValueError("A mandatory task is assigned more than once")
            rollout = self.evaluate_candidate_rollout_metrics(drone, sorties)
            if rollout["batt_over_sortie_idx"] is not None or rollout["final_return_ms"] > self.shift_end_ms:
                raise ValueError("Initial witness exceeds battery capacity or shift end")

            for srt in drone["sorties"]:
                for stp in srt["stops"]:
                    nid = stp["node_id"]
                    if nid != s_id:
                        if nid not in self.tower_catalog:
                            raise ValueError(f"Initial witness references unknown tower {nid}")
                        sx, sy = self.nodes[s_id]
                        nx, ny = self.nodes[nid]
                        if math.hypot(nx - sx, ny - sy) > r_cov + 1e-9:
                            raise ValueError(f"Initial coverage violation: {nid} on {s_id}")
                        task_id = self._routine_task_id(s_id, stp)
                        expected_srv = int(self.mandatory_routine_tasks[task_id]["srv_ms"])
                        if int(stp["srv_ms"]) != expected_srv:
                            raise ValueError(f"Initial routine srv_ms mismatch on {task_id}")
                        covered_mandatory_at_init.add(task_id)
                        drone["unserved_routine"].add(task_id)

            self.drones[d["drone_id"]] = drone

        if covered_mandatory_at_init != set(self.mandatory_routine_tasks.keys()):
            missing = set(self.mandatory_routine_tasks.keys()) - covered_mandatory_at_init
            raise ValueError(f"Initial witnesses fail to cover mandatory_routine_tasks: {sorted(missing)}")

        for s_id, stype_obj in self.stations.items():
            stype = stype_obj["station_type"]
            req_slots = {0, 1} if stype == "DABS" else {1}
            if set(station_slots_seen[s_id].keys()) != req_slots:
                raise ValueError(f"Station {s_id} ({stype}) slots {station_slots_seen[s_id]} != {req_slots}")

        self.B_witness = sum(self._remaining_witness_cost(d) for d in self.drones.values())
        if self.B_witness > self.B_cost:
            raise ValueError("Initial witness exceeds the total budget")
        self._exo_events = self._validate_events(self.spec["exogenous_events"])
        self._exo_idx = 0
        self._finished = False
        self._recompute_hover_and_free()
        self._record("SHIFT_INIT")

    def _routine_task_id(self, station_id: str, stop: Dict[str, Any]) -> str:
        tid = stop.get("task_id")
        choices = self.mandatory_tasks_by_station_node.get((station_id, stop["node_id"]), [])
        if tid is None:
            if len(choices) != 1:
                raise ValueError("REJECT_AMBIGUOUS_OR_UNREGISTERED_ROUTINE_TASK")
            return choices[0]
        if tid not in choices:
            raise ValueError("REJECT_ROUTINE_TASK_ID_MISMATCH")
        return tid

    def _validate_sorties(self, d: Dict[str, Any], sorties: Any, initial: bool = False) -> Tuple[Optional[str], set[str], set[str]]:
        tickets: set[str] = set()
        routines: set[str] = set()
        try:
            if not isinstance(sorties, list) or (not sorties and not initial):
                raise ValueError("REJECT_MALFORMED_SORTIE")
            sortie_ids = set()
            for srt in sorties:
                sid = srt["sortie_id"]
                if not isinstance(sid, str) or not sid or sid in sortie_ids:
                    raise ValueError("REJECT_DUPLICATE_OR_INVALID_SORTIE_ID")
                sortie_ids.add(sid)
                _ms(srt["t_arm_ms"], "t_arm_ms")
                stops = srt["stops"]
                if not isinstance(stops, list) or not stops:
                    raise ValueError("REJECT_MALFORMED_SORTIE")
                ret = stops[-1]
                if ret["node_id"] != d["station_id"] or _ms(ret["srv_ms"], "srv_ms") != 0 or "ticket_id" in ret or "task_id" in ret or ret.get("hover_timeout_ms", 0) != 0:
                    raise ValueError("REJECT_MALFORMED_SORTIE")
                for stop in stops[:-1]:
                    nid = stop["node_id"]
                    if nid not in self.tower_catalog:
                        raise ValueError("REJECT_UNKNOWN_TOWER")
                    srv = _ms(stop["srv_ms"], "srv_ms")
                    _ms(stop.get("hover_timeout_ms", 0), "hover_timeout_ms")
                    if "ticket_id" in stop:
                        tid = stop["ticket_id"]
                        if "task_id" in stop or initial:
                            raise ValueError("REJECT_INVALID_TICKET_TASK_FIELDS")
                        if tid in tickets:
                            raise ValueError("REJECT_DUPLICATE_TICKET")
                        if tid not in self.tickets:
                            raise ValueError("REJECT_UNPUBLISHED_TICKET")
                        ticket = self.tickets[tid]
                        if ticket["node_id"] != nid:
                            raise ValueError("REJECT_TICKET_NODE_MISMATCH")
                        if srv != ticket["srv_ms"]:
                            raise ValueError("REJECT_TAMPERED_SERVICE_DURATION")
                        if tid not in d["unserved_tickets"] and ticket["status"] != "UNCLAIMED":
                            raise ValueError("REJECT_TICKET_ALREADY_CLAIMED")
                        tickets.add(tid)
                    else:
                        tid = self._routine_task_id(d["station_id"], stop)
                        if srv != self.mandatory_routine_tasks[tid]["srv_ms"]:
                            raise ValueError("REJECT_TAMPERED_SERVICE_DURATION")
                        if tid in routines or tid in self.completed_mandatory_tasks:
                            raise ValueError("REJECT_DUPLICATE_OR_COMPLETED_ROUTINE_TASK")
                        routines.add(tid)
            return None, tickets, routines
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            reason = str(exc)
            return (reason if reason.startswith("REJECT_") else "REJECT_MALFORMED_SORTIE"), set(), set()

    def _validate_events(self, events: Any) -> List[Dict[str, Any]]:
        if not isinstance(events, list):
            raise ValueError("exogenous_events must be a list")
        ticket_ids, proposal_ids = set(), set()
        for event in events:
            when = _ms(event["t_ms"], "event t_ms")
            if not self.t_ms <= when <= self.shift_end_ms:
                raise ValueError("Event lies outside the shift")
            if event["type"] == "PUBLISH_TICKET":
                tk = event["ticket"]
                tid = tk["ticket_id"]
                if not isinstance(tid, str) or not tid or tid in ticket_ids or tk["node_id"] not in self.tower_catalog:
                    raise ValueError("Duplicate or invalid published ticket")
                ticket_ids.add(tid)
                _ms(tk["srv_ms"], "ticket srv_ms")
                if _ms(tk["sla_ms"], "sla_ms") < when:
                    raise ValueError("Ticket SLA precedes publication")
            elif event["type"] == "PROPOSE_WITNESS_CAS":
                pid = event["proposal_id"]
                if not isinstance(pid, str) or not pid or pid in proposal_ids:
                    raise ValueError("Duplicate or invalid proposal_id")
                proposal_ids.add(pid)
                _ms(event["proposal_seq"], "proposal_seq")
                if not isinstance(event.get("station_id"), str) or not isinstance(event.get("drone_id"), str):
                    raise ValueError("Proposal station_id and drone_id must be strings")
            else:
                raise ValueError(f"Unknown exogenous event type: {event['type']}")
        return sorted(events, key=lambda e: e["t_ms"])

    def _transition_mode(self, d: Dict[str, Any], new_mode: str, transition_row: str) -> None:
        """Single gated transition helper enforcing epoch += 1 on EVERY mode change."""
        old_mode = d["mode"]
        if new_mode == old_mode:
            return
        d["mode"] = new_mode
        d["epoch"] += 1
        d["last_transition_row"] = transition_row

    def _sortie_rem_cost_and_energy(
        self, srt: Dict[str, Any], start_pos: str
    ) -> Tuple[Fraction, Fraction]:
        c = Fraction(0)
        e = Fraction(0)
        cur = start_pos
        for stp in srt["stops"]:
            nxt = stp["node_id"]
            f_ms = quantize_flight_ms(self.nodes[cur], self.nodes[nxt])
            s_ms = int(stp["srv_ms"])
            c += self.c_fly * f_ms + self.c_srv * s_ms
            e += self.r_fly * f_ms + self.r_srv * s_ms
            cur = nxt
        return c, e

    def _remaining_witness_cost(self, d: Dict[str, Any]) -> Fraction:
        total = Fraction(0)
        if not d["sorties"]:
            return total
        for idx, srt in enumerate(d["sorties"]):
            if idx == 0:
                if d["mode"] in ("FLY_ROUTINE", "FLY_OUT"):
                    rem_seg = d["step_end_ms"] - self.t_ms
                    total += self.c_fly * rem_seg
                    stp0 = srt["stops"][0]
                    total += self.c_srv * int(stp0["srv_ms"])
                    cur = stp0["node_id"]
                    for stp in srt["stops"][1:]:
                        f_ms = quantize_flight_ms(self.nodes[cur], self.nodes[stp["node_id"]])
                        total += self.c_fly * f_ms + self.c_srv * int(stp["srv_ms"])
                        cur = stp["node_id"]
                elif d["mode"] == "SERVING":
                    rem_srv = d["step_end_ms"] - self.t_ms
                    total += self.c_srv * rem_srv
                    cur = srt["stops"][0]["node_id"]
                    for stp in srt["stops"][1:]:
                        f_ms = quantize_flight_ms(self.nodes[cur], self.nodes[stp["node_id"]])
                        total += self.c_fly * f_ms + self.c_srv * int(stp["srv_ms"])
                        cur = stp["node_id"]
                elif d["mode"] == "HOVER":
                    c_s, _ = self._sortie_rem_cost_and_energy(srt, d["pos_node"])
                    total += c_s
                else:
                    c_s, _ = self._sortie_rem_cost_and_energy(srt, d["station_id"])
                    total += c_s
            else:
                c_s, _ = self._sortie_rem_cost_and_energy(srt, d["station_id"])
                total += c_s
        return total

    def _compute_hover_hold_rem_ms(self, d: Dict[str, Any]) -> int:
        assert d["mode"] == "HOVER"
        rem_timeout = d["hover_limit_ms"] - (self.t_ms - d["hover_entered_ms"])
        srt0 = d["sorties"][0]
        _, e_rem0 = self._sortie_rem_cost_and_energy(srt0, d["pos_node"])
        batt_slack_ms = int((d["b_cap"] - d["battery"] - e_rem0) / self.r_hover)
        t_cur = self.t_ms
        min_sla_slack = self.shift_end_ms
        cur = d["pos_node"]
        for stp in srt0["stops"]:
            t_cur += quantize_flight_ms(self.nodes[cur], self.nodes[stp["node_id"]])
            if stp.get("ticket_id"):
                sla = self.tickets[stp["ticket_id"]]["sla_ms"]
                min_sla_slack = min(min_sla_slack, sla - t_cur)
            t_cur += int(stp["srv_ms"])
            cur = stp["node_id"]
        shift_slack = self._max_initial_delay_before_shift_or_sla_violation(d, t_cur, min_sla_slack)
        return max(0, min(rem_timeout, batt_slack_ms, shift_slack))

    def _max_initial_delay_before_shift_or_sla_violation(
        self, d: Dict[str, Any], t_after_s0_now: int, sla_slack_s0: int
    ) -> int:
        hi = max(0, min(sla_slack_s0, self.shift_end_ms - t_after_s0_now))
        lo = 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            t_c = t_after_s0_now + mid
            ok = True
            for srt in d["sorties"][1:]:
                t_c = max(t_c, int(srt["t_arm_ms"]))
                cur = d["station_id"]
                for stp in srt["stops"]:
                    t_c += quantize_flight_ms(self.nodes[cur], self.nodes[stp["node_id"]])
                    if stp.get("ticket_id") and t_c > self.tickets[stp["ticket_id"]]["sla_ms"]:
                        ok = False
                        break
                    t_c += int(stp["srv_ms"])
                    cur = stp["node_id"]
                if not ok:
                    break
            if ok and t_c <= self.shift_end_ms:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _recompute_hover_and_free(self) -> None:
        hovering = [d for d in self.drones.values() if d["mode"] == "HOVER"]
        B_liquid = self.B_cost - self.B_spent - self.B_witness
        if B_liquid < 0:
            raise ValueError(f"Negative B_liquid: {B_liquid}")
        if not hovering:
            self.B_hover = Fraction(0)
            self.B_free = B_liquid
            self.global_budget_alarm_ms = None
            return
        e_req = sum(self.c_hover * self._compute_hover_hold_rem_ms(d) for d in hovering)
        self.B_hover = min(B_liquid, e_req)
        self.B_free = B_liquid - self.B_hover
        r_tot = self.c_hover * len(hovering)
        dur_ms = int(self.B_hover / r_tot)
        self.global_budget_alarm_ms = self.t_ms + dur_ms

    def _drone_epochs_snapshot(self) -> Dict[str, int]:
        return {d_id: d["epoch"] for d_id, d in sorted(self.drones.items())}

    def _drone_modes_snapshot(self) -> Dict[str, str]:
        return {d_id: d["mode"] for d_id, d in sorted(self.drones.items())}

    def _check_ledger(self) -> None:
        buckets = (self.B_spent, self.B_witness, self.B_hover, self.B_free)
        if sum(buckets) != self.B_cost or min(buckets) < 0:
            raise ValueError(f"Budget invariant violated: {buckets}, total={self.B_cost}")

    def _record(self, event_type: str, **extra: Any) -> None:
        self._check_ledger()
        rec: Dict[str, Any] = {
            "t_ms": self.t_ms,
            "event_type": event_type,
            "ledger_ver": self.ledger_ver,
            "buckets_exact": [
                str(self.B_spent),
                str(self.B_witness),
                str(self.B_hover),
                str(self.B_free),
            ],
            "drone_epochs": self._drone_epochs_snapshot(),
            "drone_modes": self._drone_modes_snapshot(),
            "urgent_scored": self.urgent_scored,
            "urgent_pub": self.urgent_pub,
            "hard_violations": self.hard_violations,
        }
        rec.update(extra)
        self.logs.append(rec)

    def _settle_to(self, target_ms: int) -> None:
        _ms(target_ms, "target_ms")
        if target_ms < self.t_ms or target_ms > self.shift_end_ms:
            raise ValueError("Cannot settle backwards or outside the shift")
        if target_ms > self._next_internal_ms():
            self.advance_to(target_ms)
            return
        dt = target_ms - self.t_ms
        if dt == 0:
            return
        for d in self.drones.values():
            if d["mode"] in ("FLY_ROUTINE", "FLY_OUT"):
                self.B_spent += self.c_fly * dt
                self.B_witness -= self.c_fly * dt
                d["battery"] += self.r_fly * dt
            elif d["mode"] == "SERVING":
                self.B_spent += self.c_srv * dt
                self.B_witness -= self.c_srv * dt
                d["battery"] += self.r_srv * dt
            elif d["mode"] == "HOVER":
                self.B_spent += self.c_hover * dt
                self.B_hover -= self.c_hover * dt
                d["battery"] += self.r_hover * dt
        self.t_ms = target_ms
        self._check_ledger()

    def _next_internal_ms(self) -> int:
        nxt = self.shift_end_ms
        if self.global_budget_alarm_ms is not None:
            nxt = min(nxt, self.global_budget_alarm_ms)
        for d in self.drones.values():
            if d["mode"] == "HOVER":
                nxt = min(nxt, self.t_ms + self._compute_hover_hold_rem_ms(d))
            elif d["step_end_ms"] is not None:
                nxt = min(nxt, d["step_end_ms"])
        return nxt

    def _process_internal_at_now(self) -> None:
        """Deterministic same-ms internal event loop:
        Stage 1: Flight/Service completions (sorted by drone_id ASC)
        Stage 2: Per-drone hover hold deadline alarms (sorted by drone_id ASC)
        Stage 3: Global shared hover budget alarm
        Stage 4: Ground default takeoff alarms (sorted by drone_id ASC)
        """
        progressed = True
        while progressed:
            progressed = False
            # 1. Flight & Service completions
            for d in sorted(self.drones.values(), key=lambda x: x["id"]):
                if d["mode"] in ("FLY_ROUTINE", "FLY_OUT") and d["step_end_ms"] == self.t_ms:
                    progressed = True
                    srt0 = d["sorties"][0]
                    stp0 = srt0["stops"][0]
                    d["pos_node"] = stp0["node_id"]
                    if stp0.get("ticket_id"):
                        if self.t_ms > self.tickets[stp0["ticket_id"]]["sla_ms"]:
                            self.hard_violations += 1
                    if stp0["node_id"] == d["station_id"] and len(srt0["stops"]) == 1:
                        # Row T8: Return to home station & swap battery -> GROUND_READY or GROUND_IDLE
                        if d["battery"] > d["b_cap"]:
                            self.hard_violations += 1
                        self.urgent_scored += len(d["onboard_served_tickets"])
                        d["onboard_served_tickets"].clear()
                        d["battery"] = Fraction(0)
                        d["sorties"].pop(0)
                        if d["sorties"]:
                            self._transition_mode(d, "GROUND_READY", "T8")
                            d["step_end_ms"] = max(self.t_ms, int(d["sorties"][0]["t_arm_ms"]))
                        else:
                            self._transition_mode(d, "GROUND_IDLE", "T8")
                            d["step_end_ms"] = None
                        self._recompute_hover_and_free()
                        self._record("SORTIE_RETURN_AND_SWAP", drone_id=d["id"])
                    else:
                        # Row T4: Arrive at tower -> SERVING
                        s_ms = int(stp0["srv_ms"])
                        self._transition_mode(d, "SERVING", "T4")
                        d["step_end_ms"] = self.t_ms + s_ms
                        self._recompute_hover_and_free()
                        self._record("ARRIVE_SERVING", drone_id=d["id"], node_id=stp0["node_id"])
                elif d["mode"] == "SERVING" and d["step_end_ms"] == self.t_ms:
                    progressed = True
                    srt0 = d["sorties"][0]
                    stp0 = srt0["stops"].pop(0)
                    if stp0.get("ticket_id"):
                        d["unserved_tickets"].discard(stp0["ticket_id"])
                        d["onboard_served_tickets"].append(stp0["ticket_id"])
                    else:
                        task_id = self._routine_task_id(d["station_id"], stp0)
                        d["unserved_routine"].discard(task_id)
                        self.completed_mandatory_tasks.add(task_id)

                    h_max = int(stp0.get("hover_timeout_ms", 0))
                    B_liq = self.B_cost - self.B_spent - self.B_witness
                    can_hover = False
                    if h_max > 0 and B_liq > 0:
                        prev_mode = d["mode"]
                        d["mode"] = "HOVER"
                        d["hover_limit_ms"] = h_max
                        d["hover_entered_ms"] = self.t_ms
                        hold_rem = self._compute_hover_hold_rem_ms(d)
                        d["mode"] = prev_mode
                        if hold_rem > 0:
                            can_hover = True
                    if can_hover:
                        # Row T5: SERVING -> HOVER
                        d["hover_limit_ms"] = h_max
                        d["hover_entered_ms"] = self.t_ms
                        self._transition_mode(d, "HOVER", "T5")
                        d["step_end_ms"] = None
                        self._recompute_hover_and_free()
                        self._record("ENTER_HOVER", drone_id=d["id"], node_id=d["pos_node"])
                    else:
                        # Row T6: SERVING -> FLY_*
                        nxt_stp = srt0["stops"][0]
                        f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[nxt_stp["node_id"]])
                        nxt_mode = "FLY_OUT" if nxt_stp.get("ticket_id") else "FLY_ROUTINE"
                        self._transition_mode(d, nxt_mode, "T6")
                        d["step_end_ms"] = self.t_ms + f_ms
                        self._recompute_hover_and_free()
                        self._record(
                            "DEPART_AFTER_SERVICE",
                            drone_id=d["id"],
                            zero_hover_forced=(h_max > 0),
                        )

            # 2. Individual hover hold deadline alarms (Row T7)
            for d in sorted(self.drones.values(), key=lambda x: x["id"]):
                if d["mode"] == "HOVER" and self._compute_hover_hold_rem_ms(d) == 0:
                    progressed = True
                    nxt_stp = d["sorties"][0]["stops"][0]
                    f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[nxt_stp["node_id"]])
                    nxt_mode = "FLY_OUT" if nxt_stp.get("ticket_id") else "FLY_ROUTINE"
                    self._transition_mode(d, nxt_mode, "T7")
                    d["step_end_ms"] = self.t_ms + f_ms
                    self._recompute_hover_and_free()
                    self._record("HOLD_ALARM_FIRED", drone_id=d["id"])

            # 3. Global shared hover budget alarm (Row T7)
            if self.global_budget_alarm_ms == self.t_ms:
                hovering = [
                    d for d in sorted(self.drones.values(), key=lambda x: x["id"]) if d["mode"] == "HOVER"
                ]
                if hovering:
                    progressed = True
                    for d in hovering:
                        nxt_stp = d["sorties"][0]["stops"][0]
                        f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[nxt_stp["node_id"]])
                        nxt_mode = "FLY_OUT" if nxt_stp.get("ticket_id") else "FLY_ROUTINE"
                        self._transition_mode(d, nxt_mode, "T7")
                        d["step_end_ms"] = self.t_ms + f_ms
                    self._recompute_hover_and_free()
                    self._record("GLOBAL_HOVER_BUDGET_ALARM_FIRED")

            # 4. Ground default takeoff alarms (Row T1, including t=0 launch!)
            for d in sorted(self.drones.values(), key=lambda x: x["id"]):
                if d["mode"] == "GROUND_READY" and d["step_end_ms"] == self.t_ms:
                    progressed = True
                    stp0 = d["sorties"][0]["stops"][0]
                    f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[stp0["node_id"]])
                    nxt_mode = "FLY_OUT" if stp0.get("ticket_id") else "FLY_ROUTINE"
                    self._transition_mode(d, nxt_mode, "T1")
                    d["step_end_ms"] = self.t_ms + f_ms
                    self._recompute_hover_and_free()
                    self._record("DEFAULT_TAKEOFF_FIRED", drone_id=d["id"])

    def evaluate_candidate_rollout_metrics(
        self, d: Dict[str, Any], cand_sorties: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Pure forward physical evaluator returning exact per-sortie energies, arrival times, and return ms."""
        s_id = d["station_id"]
        t_sim = self.t_ms
        new_cost = Fraction(0)
        sortie_metrics = []
        sla_late_ticket: Optional[str] = None
        batt_over_idx: Optional[int] = None

        for idx, srt in enumerate(cand_sorties):
            if idx == 0 and d["mode"] == "HOVER":
                cur = d["pos_node"]
                e_sim = d["battery"]
            else:
                t_sim = max(t_sim, int(srt["t_arm_ms"]))
                cur = s_id
                e_sim = Fraction(0)
            last_srv_end_ms = t_sim
            for stp in srt["stops"]:
                nxt = stp["node_id"]
                f_ms = quantize_flight_ms(self.nodes[cur], self.nodes[nxt])
                s_ms = int(stp["srv_ms"])
                t_sim += f_ms
                if stp.get("ticket_id") and stp["ticket_id"] in self.tickets:
                    if t_sim > self.tickets[stp["ticket_id"]]["sla_ms"] and sla_late_ticket is None:
                        sla_late_ticket = stp["ticket_id"]
                t_sim += s_ms
                if nxt != s_id:
                    last_srv_end_ms = t_sim
                e_sim += self.r_fly * f_ms + self.r_srv * s_ms
                new_cost += self.c_fly * f_ms + self.c_srv * s_ms
                cur = nxt
            if e_sim > d["b_cap"] and batt_over_idx is None:
                batt_over_idx = idx
            sortie_metrics.append(
                {
                    "sortie_id": srt["sortie_id"],
                    "energy_float": round(float(e_sim), 6),
                    "energy_exact": str(e_sim),
                    "last_service_end_ms": last_srv_end_ms,
                    "return_station_ms": t_sim,
                }
            )
        return {
            "total_cost": new_cost,
            "final_return_ms": t_sim,
            "sla_late_ticket": sla_late_ticket,
            "batt_over_sortie_idx": batt_over_idx,
            "sortie_metrics": sortie_metrics,
        }

    def atomic_cas(self, prop: Dict[str, Any]) -> Dict[str, Any]:
        """Execute atomic CAS proposal and emit exact audit log record."""
        proposal_id = prop.get("proposal_id", "INVALID_PROPOSAL")
        def reject(reason):
            self._record("CAS_DECISION", proposal_id=proposal_id, verdict=reason)
            return self.logs[-1]
        if self._finished or self.t_ms >= self.shift_end_ms:
            return reject("REJECT_SHIFT_ALREADY_ENDED")
        if prop.get("drone_id") not in self.drones:
            return reject("REJECT_UNKNOWN_DRONE")
        d = self.drones[prop["drone_id"]]
        s_id = d["station_id"]
        if prop.get("station_id") != s_id:
            return reject("REJECT_STATION_DRONE_MISMATCH")
        try:
            _ms(prop["snapshot_epoch"], "snapshot_epoch")
            _ms(prop["proposal_seq"], "proposal_seq")
            if not isinstance(prop["new_witness_ver"], str) or not prop["new_witness_ver"] or prop["new_witness_ver"] == d["witness_ver"]:
                return reject("REJECT_INVALID_WITNESS_VERSION")
            prop["snapshot_witness_ver"]
        except (KeyError, TypeError, ValueError):
            return reject("REJECT_MALFORMED_PROPOSAL")
        cand_sorties = prop.get("candidate_sorties")
        error, cand_tickets, cand_routines = self._validate_sorties(d, cand_sorties)
        if error:
            return reject(error)

        if not d["unserved_tickets"].issubset(cand_tickets) or not d["unserved_routine"].issubset(cand_routines):
            self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_COMMITTED_WORK_DROPPED")
            return self.logs[-1]

        # Step 2: Non-diversion mode check & optimistic epoch/witness lock
        if d["mode"] not in ("GROUND_IDLE", "GROUND_READY", "HOVER"):
            self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_DRONE_IN_FLIGHT_NO_DIVERSION")
            return self.logs[-1]
        if d["epoch"] != int(prop["snapshot_epoch"]) or d["witness_ver"] != prop["snapshot_witness_ver"]:
            self._record(
                "CAS_DECISION",
                proposal_id=prop["proposal_id"],
                verdict="REJECT_STALE_EPOCH_OR_WITNESS",
                actual_epoch=d["epoch"],
                submitted_epoch=int(prop["snapshot_epoch"]),
            )
            return self.logs[-1]

        # Step 3: Station-type coverage radius check
        sx, sy = self.nodes[s_id]
        for srt in cand_sorties:
            for stp in srt["stops"][:-1]:
                nx, ny = self.nodes[stp["node_id"]]
                if math.hypot(nx - sx, ny - sy) > d["r_cov_m"] + 1e-9:
                    self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_COVERAGE_RADIUS_EXCEEDED")
                    return self.logs[-1]

        # Step 4: Live forward physical rollout
        rollout = self.evaluate_candidate_rollout_metrics(d, cand_sorties)
        diag = {
            "sortie_metrics": rollout["sortie_metrics"],
            "final_return_ms": rollout["final_return_ms"],
        }
        if rollout["sla_late_ticket"] is not None:
            self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_SLA_ARRIVAL_LATE", rollout_diag=diag)
            return self.logs[-1]
        if rollout["batt_over_sortie_idx"] is not None:
            self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_BATTERY_CAP_EXCEEDED", rollout_diag=diag)
            return self.logs[-1]
        if rollout["final_return_ms"] > self.shift_end_ms:
            self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_SHIFT_END_EXCEEDED", rollout_diag=diag)
            return self.logs[-1]

        # Step 5: Zero-positive-slack budget check protecting other hovering drones
        old_cost = self._remaining_witness_cost(d)
        delta_c = rollout["total_cost"] - old_cost
        B_liquid = self.B_cost - self.B_spent - self.B_witness
        other_hover_req = sum(
            self.c_hover * self._compute_hover_hold_rem_ms(o)
            for o in self.drones.values()
            if o["mode"] == "HOVER" and o["id"] != d["id"]
        )
        B_avail = B_liquid - min(B_liquid, other_hover_req)
        if delta_c > B_avail:
            self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict="REJECT_INSUFFICIENT_FREE_BUDGET", rollout_diag=diag)
            return self.logs[-1]

        # Step 6: Atomic state & mode transition commit
        new_tickets = cand_tickets - d["unserved_tickets"]
        for tid in new_tickets:
            self.tickets[tid]["status"] = "CLAIMED"
        d["unserved_tickets"] = cand_tickets
        d["unserved_routine"] = cand_routines
        d["sorties"] = copy.deepcopy(list(cand_sorties))
        d["witness_ver"] = prop["new_witness_ver"]
        self.B_witness += delta_c
        self.ledger_ver += 1

        if d["mode"] == "HOVER":
            stp0 = d["sorties"][0]["stops"][0]
            f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[stp0["node_id"]])
            nxt_mode = "FLY_OUT" if stp0.get("ticket_id") else "FLY_ROUTINE"
            self._transition_mode(d, nxt_mode, "T7")
            d["step_end_ms"] = self.t_ms + f_ms
        elif d["mode"] == "GROUND_IDLE":
            t_dep = max(self.t_ms, int(d["sorties"][0]["t_arm_ms"]))
            if t_dep > self.t_ms:
                self._transition_mode(d, "GROUND_READY", "T2")
                d["step_end_ms"] = t_dep
            else:
                stp0 = d["sorties"][0]["stops"][0]
                f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[stp0["node_id"]])
                nxt_mode = "FLY_OUT" if stp0.get("ticket_id") else "FLY_ROUTINE"
                self._transition_mode(d, nxt_mode, "T3")
                d["step_end_ms"] = self.t_ms + f_ms
        elif d["mode"] == "GROUND_READY":
            t_dep = max(self.t_ms, int(d["sorties"][0]["t_arm_ms"]))
            if t_dep > self.t_ms:
                # Remains GROUND_READY (no mode change -> epoch unchanged!)
                d["step_end_ms"] = t_dep
            else:
                stp0 = d["sorties"][0]["stops"][0]
                f_ms = quantize_flight_ms(self.nodes[d["pos_node"]], self.nodes[stp0["node_id"]])
                nxt_mode = "FLY_OUT" if stp0.get("ticket_id") else "FLY_ROUTINE"
                self._transition_mode(d, nxt_mode, "T1")
                d["step_end_ms"] = self.t_ms + f_ms

        self._recompute_hover_and_free()
        verdict = (
            "PASS_EXACT_EXHAUST"
            if (delta_c > 0 and self.B_free == 0)
            else ("PASS_ZERO_NET_COST" if delta_c == 0 else "PASS_COMMIT_ACCEPTED")
        )
        self._record("CAS_DECISION", proposal_id=prop["proposal_id"], verdict=verdict, rollout_diag=diag)
        return self.logs[-1]

    def _process_exogenous_at_now(self) -> None:
        batch = []
        while self._exo_idx < len(self._exo_events) and self._exo_events[self._exo_idx]["t_ms"] == self.t_ms:
            batch.append(self._exo_events[self._exo_idx])
            self._exo_idx += 1
        # The frozen protocol orders existing-work CAS before new ticket publication
        # at the same ms; a proposal cannot claim a not-yet-published same-ms ticket.
        def cas_sort_key(p):
            d = self.drones.get(p.get("drone_id"), {})
            sorties = p.get("candidate_sorties")
            if not isinstance(sorties, list):
                sorties = []
            tids = [stp.get("ticket_id") for srt in sorties if isinstance(srt, dict) and isinstance(srt.get("stops"), list)
                    for stp in srt["stops"] if isinstance(stp, dict) and isinstance(stp.get("ticket_id"), str) and stp["ticket_id"]
                    and stp["ticket_id"] not in d.get("unserved_tickets", set())]
            priority = min(((self.tickets[t]["sla_ms"], t) for t in tids if t in self.tickets), default=(10**15, ""))
            return (*priority, str(p.get("station_id", "")), str(p.get("drone_id", "")), p["proposal_seq"])
        for prop in sorted((e for e in batch if e["type"] == "PROPOSE_WITNESS_CAS"), key=cas_sort_key):
            self.atomic_cas(prop)
            self._process_internal_at_now()
        for event in sorted((e for e in batch if e["type"] == "PUBLISH_TICKET"), key=lambda e: e["ticket"]["ticket_id"]):
            ticket = copy.deepcopy(event["ticket"])
            ticket["status"] = "UNCLAIMED"
            self.tickets[ticket["ticket_id"]] = ticket
            self.urgent_pub += 1
            self._record("TICKET_PUBLISHED", ticket_id=ticket["ticket_id"], node_id=ticket["node_id"])

    def advance_to(self, target_ms: int) -> None:
        """Advance the event clock through every physical and exogenous event."""
        _ms(target_ms, "target_ms")
        if not self.t_ms <= target_ms <= self.shift_end_ms:
            raise ValueError("Cannot advance backwards or outside the shift")
        self._process_internal_at_now()
        self._process_exogenous_at_now()
        while self.t_ms < target_ms:
            next_exo = self._exo_events[self._exo_idx]["t_ms"] if self._exo_idx < len(self._exo_events) else target_ms
            next_time = min(self._next_internal_ms(), next_exo, target_ms)
            self._settle_to(next_time)
            self._process_internal_at_now()
            self._process_exogenous_at_now()

    def run_to_shift_end(self) -> List[Dict[str, Any]]:
        if self._finished:
            return self.logs
        self.advance_to(self.shift_end_ms)

        # Independent SHIFT_END audit: all drones GROUND_IDLE, zero unserved tickets, 100% mandatory_routine_tasks done!
        mandatory_all_done = self.completed_mandatory_tasks == set(self.mandatory_routine_tasks.keys())
        if not mandatory_all_done:
            self.hard_violations += 1
        for d in self.drones.values():
            if d["mode"] != "GROUND_IDLE" or d["unserved_routine"] or d["unserved_tickets"]:
                self.hard_violations += 1
        self._record(
            "SHIFT_END",
            mandatory_routine_completed_count=len(self.completed_mandatory_tasks),
            mandatory_routine_total_count=len(self.mandatory_routine_tasks),
            mandatory_routine_all_done=mandatory_all_done,
            compliant_score=0 if self.hard_violations > 0 else self.urgent_scored,
        )
        self._finished = True
        return self.logs


# ============================================================================
# Quotient Physical Action Interface & Shared Deterministic Witness Completer
# ============================================================================
class SharedWitnessCompleter:
    """Bounded deterministic completer available to physical policy callbacks.

    This prototype includes no trained RL policy or implemented baseline benchmark.

    In physical-route CAPE-RL (Review §2.B.3), the policy MUST choose the explicit immediate
    physical action `(station_id, slot_idx, target_spec)` where `target_spec` is either:
      - `("RETURN", station_id)` (return to home station now, only valid in HOVER)
      - `("TICKET", ticket_id, node_id)` (fly immediately to `node_id` to serve `ticket_id`)
      - `("ROUTINE", node_id)` (fly immediately to `node_id` to serve committed routine task)
    `SharedWitnessCompleter` NEVER overrides the policy's chosen immediate physical waypoint;
    it only constructs a unique canonical residual witness for any remaining committed tasks
    after that waypoint, breaking ties strictly by:
      `(delta_cost_exact ASC, final_return_ms ASC, canonical_repr ASC)`
    and returning `(None, "NO_FEASIBLE_WITNESS")` if no completion exists in the
    bounded no-optional-hover domain. More than six required stops return
    `COMPLETION_LIMIT_EXCEEDED`, which is not an infeasibility proof.
    """

    @staticmethod
    def enumerate_quotient_physical_actions(executor: OriginalDomainCapeExecutor) -> List[Tuple[Any, ...]]:
        """Quotient action deduplication (replaces broken v5 §3.2 cross-ref):
        1. Only controllable modes (`GROUND_IDLE`, `GROUND_READY`, `HOVER`) expose physical actions;
        2. SABS slot 0 is permanently masked (`slot_idx == 1` only on SABS, `(0, 1)` on DABS);
        3. Multiple unclaimed tickets on different towers remain distinct physical targets, while
           empty ground return is masked out (`RETURN` only valid in `HOVER`).
        """
        actions: List[Tuple[Any, ...]] = []
        unclaimed = [
            tk for tk in sorted(executor.tickets.values(), key=lambda x: x["ticket_id"])
            if tk["status"] == "UNCLAIMED"
        ]
        for d in sorted(executor.drones.values(), key=lambda x: (x["station_id"], x["slot_idx"])):
            if d["mode"] not in ("GROUND_IDLE", "GROUND_READY", "HOVER"):
                continue
            s_id = d["station_id"]
            slot_idx = d["slot_idx"]
            sx, sy = executor.nodes[s_id]
            if d["mode"] == "HOVER":
                actions.append(("STEP", s_id, slot_idx, ("RETURN", s_id)))
            for task_id in sorted(d["unserved_routine"]):
                task = executor.mandatory_routine_tasks[task_id]
                actions.append(("STEP", s_id, slot_idx, ("ROUTINE", task["node_id"], task_id)))
            eligible_tickets = unclaimed + [executor.tickets[tid] for tid in sorted(d["unserved_tickets"])]
            for tk in eligible_tickets:
                nx, ny = executor.nodes[tk["node_id"]]
                if math.hypot(nx - sx, ny - sy) <= d["r_cov_m"] + 1e-9:
                    actions.append(("STEP", s_id, slot_idx, ("TICKET", tk["ticket_id"], tk["node_id"])))
        return actions

    @staticmethod
    def complete_after_explicit_next_step(
        executor: OriginalDomainCapeExecutor,
        station_id: str,
        slot_idx: int,
        target_spec: Tuple[Any, ...],
    ) -> Tuple[Optional[List[Dict[str, Any]]], str]:
        matching = [
            d for d in executor.drones.values()
            if d["station_id"] == station_id and d["slot_idx"] == slot_idx
        ]
        if not matching:
            return None, "INVALID_STATION_OR_SLOT"
        d = matching[0]
        if d["mode"] not in ("GROUND_IDLE", "GROUND_READY", "HOVER"):
            return None, "DRONE_NOT_CONTROLLABLE"

        req_stops: List[Dict[str, Any]] = []
        for task_id in sorted(d["unserved_routine"]):
            task = executor.mandatory_routine_tasks[task_id]
            req_stops.append({"node_id": task["node_id"], "srv_ms": task["srv_ms"], "task_id": task_id})
        for tid in sorted(d["unserved_tickets"]):
            tk = executor.tickets[tid]
            req_stops.append({"node_id": tk["node_id"], "srv_ms": tk["srv_ms"], "ticket_id": tid})
        if not isinstance(target_spec, (tuple, list)) or not target_spec:
            return None, "INVALID_TARGET_SPEC"
        kind = target_spec[0]
        first_stop = None
        if kind == "RETURN":
            if len(target_spec) != 2 or target_spec[1] != station_id:
                return None, "INVALID_RETURN_TARGET"
            if d["mode"] != "HOVER":
                return None, "EMPTY_GROUND_RETURN_FORBIDDEN"
        elif kind == "ROUTINE":
            if len(target_spec) not in (2, 3):
                return None, "INVALID_TARGET_SPEC"
            matches = [i for i, stop in enumerate(req_stops) if stop["node_id"] == target_spec[1] and "task_id" in stop
                       and (len(target_spec) == 2 or stop["task_id"] == target_spec[2])]
            if len(matches) != 1:
                return None, "ROUTINE_TARGET_NOT_PENDING_OR_AMBIGUOUS"
            first_stop = req_stops.pop(matches[0])
        elif kind == "TICKET":
            if len(target_spec) != 3:
                return None, "INVALID_TARGET_SPEC"
            tid, nid = target_spec[1:]
            tk = executor.tickets.get(tid)
            if tk is None or (tk["status"] != "UNCLAIMED" and tid not in d["unserved_tickets"]):
                return None, "TICKET_NOT_UNCLAIMED_OR_OWN_COMMITTED"
            if tk["node_id"] != nid:
                return None, "TICKET_NODE_MISMATCH"
            first_stop = {"node_id": nid, "srv_ms": tk["srv_ms"], "ticket_id": tid}
            req_stops = [stop for stop in req_stops if stop.get("ticket_id") != tid]
        else:
            return None, f"UNKNOWN_TARGET_KIND:{kind}"

        total_stops = len(req_stops) + (first_stop is not None)
        if total_stops > 6:
            return None, "COMPLETION_LIMIT_EXCEEDED"
        # The exhaustive *bounded* domain is all orderings and contiguous sortie
        # partitions of <=6 stops, with the requested physical first step fixed.
        # No optional hovering is inserted. A limit result is not infeasibility.
        candidates_scored = []
        old_cost = executor._remaining_witness_cost(d)
        liquid = executor.B_cost - executor.B_spent - executor.B_witness
        other_hover = sum(executor.c_hover * executor._compute_hover_hold_rem_ms(o)
                          for o in executor.drones.values() if o["mode"] == "HOVER" and o["id"] != d["id"])
        available = liquid - min(liquid, other_hover)
        sx, sy = executor.nodes[station_id]
        for perm in itertools.permutations(req_stops):
            ordered = ([first_stop] if first_stop is not None else []) + list(perm)
            for cut_mask in range(1 << max(0, len(ordered) - 1)):
                groups = []
                if ordered:
                    groups = [[ordered[0]]]
                    for i, stop in enumerate(ordered[1:]):
                        if cut_mask & (1 << i):
                            groups.append([])
                        groups[-1].append(stop)
                if kind == "RETURN":
                    groups.insert(0, [])
                candidate = [{"sortie_id": f"CMP_{i}", "t_arm_ms": executor.t_ms,
                              "stops": group + [{"node_id": station_id, "srv_ms": 0}]}
                             for i, group in enumerate(groups)]
                if any(math.hypot(executor.nodes[stop["node_id"]][0] - sx,
                                  executor.nodes[stop["node_id"]][1] - sy) > d["r_cov_m"] + 1e-9 for stop in ordered):
                    continue
                roll = executor.evaluate_candidate_rollout_metrics(d, candidate)
                if roll["sla_late_ticket"] is not None or roll["batt_over_sortie_idx"] is not None or roll["final_return_ms"] > executor.shift_end_ms:
                    continue
                delta = roll["total_cost"] - old_cost
                if delta > available:
                    continue
                candidates_scored.append((delta, roll["final_return_ms"], repr(candidate), candidate))
        if not candidates_scored:
            return None, "NO_FEASIBLE_WITNESS"
        return min(candidates_scored, key=lambda item: item[:3])[3], "OK"

    @classmethod
    def execute_timed_decision_step(
        cls,
        executor: OriginalDomainCapeExecutor,
        policy_selector,
        proposal_id: str,
        proposal_seq: int,
    ) -> Dict[str, Any]:
        """Measure full wall-clock from snapshot -> enumerate -> mask -> complete -> CAS commit."""
        t0_ns = time.perf_counter_ns()
        executor.advance_to(executor.t_ms)
        snapshot = {
            "t_ms": executor.t_ms,
            "epochs": executor._drone_epochs_snapshot(),
            "witness_vers": {k: d["witness_ver"] for k, d in executor.drones.items()},
        }
        raw_actions = cls.enumerate_quotient_physical_actions(executor)
        masked_feasible: List[Tuple[Tuple[Any, ...], Optional[List[Dict[str, Any]]]]] = []
        completion_status_counts: Dict[str, int] = {}
        for act in raw_actions:
            _, s_id, slot_idx, target_spec = act
            comp, status = cls.complete_after_explicit_next_step(executor, s_id, slot_idx, target_spec)
            completion_status_counts[status] = completion_status_counts.get(status, 0) + 1
            if status == "OK" and comp is not None:
                masked_feasible.append((act, comp))

        # The policy owns only the physical action. Completion candidates are
        # private; the legacy pair return is accepted only if it is unchanged.
        chosen_act, chosen_sorties = None, None
        if masked_feasible:
            selected = policy_selector(copy.deepcopy(snapshot), copy.deepcopy(masked_feasible))
            selected_pair = None
            if isinstance(selected, (tuple, list)) and len(selected) == 2 and isinstance(selected[0], (tuple, list)):
                selected_pair = selected
                chosen_act = tuple(selected[0])
            else:
                chosen_act = tuple(selected) if isinstance(selected, (tuple, list)) else selected
            matches = [(action, completion) for action, completion in masked_feasible if action == chosen_act]
            if len(matches) != 1:
                raise ValueError("Selector must choose an offered physical action")
            chosen_act, chosen_sorties = matches[0]
            if selected_pair is not None and selected_pair[1] != chosen_sorties:
                raise ValueError("Selector must not alter the shared completion witness")
        decision_elapsed_ms = (time.perf_counter_ns() - t0_ns + 999_999) // 1_000_000
        ready_ms = min(executor.shift_end_ms, snapshot["t_ms"] + decision_elapsed_ms)
        executor.advance_to(ready_ms)
        cas_rec = None
        verdict = "COMPLETION_LIMIT_EXCEEDED" if completion_status_counts.get("COMPLETION_LIMIT_EXCEEDED") else "NO_EXECUTOR_ACTION"
        if chosen_act is not None and chosen_sorties is not None:
            _, s_id, slot_idx, _ = chosen_act
            d = next(x for x in executor.drones.values() if x["station_id"] == s_id and x["slot_idx"] == slot_idx)
            prop = {
                "proposal_id": proposal_id,
                "proposal_seq": proposal_seq,
                "station_id": s_id,
                "drone_id": d["id"],
                "snapshot_epoch": snapshot["epochs"][d["id"]],
                "snapshot_witness_ver": snapshot["witness_vers"][d["id"]],
                "new_witness_ver": f"{snapshot['witness_vers'][d['id']]}_p{proposal_seq}",
                "candidate_sorties": chosen_sorties,
            }
            if snapshot["t_ms"] + decision_elapsed_ms >= executor.shift_end_ms:
                verdict = "REJECT_DECISION_READY_AFTER_SHIFT_END"
            else:
                cas_rec = executor.atomic_cas(prop)
                verdict = cas_rec["verdict"]
        wall_ms = (time.perf_counter_ns() - t0_ns) / 1_000_000.0
        return {
            "chosen_action": chosen_act,
            "feasible_action_count": len(masked_feasible),
            "completion_status_counts": completion_status_counts,
            "cas_verdict": verdict,
            "decision_ready_ms": ready_ms,
            "simulated_decision_elapsed_ms": decision_elapsed_ms,
            "wall_decision_to_commit_ms": round(wall_ms, 6),
        }
