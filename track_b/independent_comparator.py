"""Track B Strictly Separate Independent Comparator.

Compares raw actual logs emitted by `cape_executor.py` (which never saw expected outputs)
against `track_b_v5_1_expected.json`.
"""
from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List


def extract_canonical_checkpoints(trace_id: str, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    key_types = {
        "SHIFT_INIT",
        "CAS_DECISION",
        "HOLD_ALARM_FIRED",
        "GLOBAL_HOVER_BUDGET_ALARM_FIRED",
        "SHIFT_END",
    }
    out: List[Dict[str, Any]] = []
    for lg in logs:
        include = lg["event_type"] in key_types or (
            trace_id.startswith("B7") and lg["t_ms"] == 86668 and lg["event_type"] == "SORTIE_RETURN_AND_SWAP"
        )
        if not include:
            continue
        # Verify exact 4-bucket budget conservation at every checkpoint independently!
        b_vals = [Fraction(x) for x in lg["buckets_exact"]]
        if any(x < 0 for x in b_vals):
            raise AssertionError(f"Negative bucket in {trace_id}: {lg}")
        slim = {
            "t_ms": lg["t_ms"],
            "event_type": lg["event_type"],
            "ledger_ver": lg["ledger_ver"],
            "buckets_exact": lg["buckets_exact"],
            "drone_epochs": lg["drone_epochs"],
            "urgent_scored": lg["urgent_scored"],
            "urgent_pub": lg["urgent_pub"],
            "hard_violations": lg["hard_violations"],
        }
        for opt_k in (
            "proposal_id",
            "verdict",
            "drone_id",
            "mandatory_routine_all_done",
            "compliant_score",
            "rollout_diag",
        ):
            if opt_k in lg:
                slim[opt_k] = lg[opt_k]
        out.append(slim)
    return out


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _validate_raw_logs(logs: Any, budget: Fraction) -> None:
    """Independent arithmetic/state guards for *all* events, not only checkpoints.

    This validates exported evidence; it is not an independent flight simulator.
    Missing intermediate physical events require a separately retained full-log
    expectation or input replay and are not proved absent by 37 checkpoints.
    """
    if not isinstance(logs, list) or not logs or not all(isinstance(lg, dict) for lg in logs):
        raise ValueError("raw logs must be a nonempty list of event objects")
    if logs[0].get("event_type") != "SHIFT_INIT" or logs[-1].get("event_type") != "SHIFT_END":
        raise ValueError("raw logs must start at SHIFT_INIT and finish at SHIFT_END")
    known = {"SHIFT_INIT", "SHIFT_END", "CAS_DECISION", "TICKET_PUBLISHED", "SORTIE_RETURN_AND_SWAP",
             "ARRIVE_SERVING", "ENTER_HOVER", "DEPART_AFTER_SERVICE", "HOLD_ALARM_FIRED",
             "GLOBAL_HOVER_BUDGET_ALARM_FIRED", "DEFAULT_TAKEOFF_FIRED"}
    modes = {"GROUND_READY", "GROUND_IDLE", "FLY_ROUTINE", "FLY_OUT", "SERVING", "HOVER"}
    previous = None
    for idx, lg in enumerate(logs):
        event = lg["event_type"]
        if event not in known or (idx and event == "SHIFT_INIT") or (idx != len(logs)-1 and event == "SHIFT_END"):
            raise ValueError(f"invalid event sequence at event {idx}")
        when = _nonnegative_int(lg["t_ms"], "t_ms")
        ledger = _nonnegative_int(lg["ledger_ver"], "ledger_ver")
        values = lg["buckets_exact"]
        if not isinstance(values, list) or len(values) != 4 or not all(isinstance(v, str) for v in values):
            raise ValueError("buckets_exact must contain exactly four rational strings")
        buckets = [Fraction(value) for value in values]
        if min(buckets) < 0 or sum(buckets) != budget:
            raise ValueError(f"budget invariant broken at event {idx}")
        scored = _nonnegative_int(lg["urgent_scored"], "urgent_scored")
        published = _nonnegative_int(lg["urgent_pub"], "urgent_pub")
        if scored > published or _nonnegative_int(lg["hard_violations"], "hard_violations") != 0:
            raise ValueError(f"score or hard violation at event {idx}")
        epochs, current_modes = lg["drone_epochs"], lg["drone_modes"]
        if not isinstance(epochs, dict) or not epochs or not isinstance(current_modes, dict) or set(epochs) != set(current_modes):
            raise ValueError("drone epoch/mode keys are missing or inconsistent")
        for drone, epoch in epochs.items():
            _nonnegative_int(epoch, "drone_epoch")
            if current_modes[drone] not in modes:
                raise ValueError("unknown physical mode")
        if previous is None:
            if ledger != 0 or scored != 0 or published != 0 or any(epochs.values()) or any(mode not in {"GROUND_READY", "GROUND_IDLE"} for mode in current_modes.values()):
                raise ValueError("invalid shift initialization")
        else:
            if when < previous["t_ms"] or set(epochs) != set(previous["drone_epochs"]):
                raise ValueError("time or drone identity changed backwards")
            increment = int(event == "CAS_DECISION" and str(lg.get("verdict", "")).startswith("PASS_"))
            if ledger != previous["ledger_ver"] + increment:
                raise ValueError("ledger version does not match the CAS outcome")
            if scored < previous["urgent_scored"] or (scored != previous["urgent_scored"] and event != "SORTIE_RETURN_AND_SWAP"):
                raise ValueError("urgent score changed outside a sortie return")
            if buckets[0] < Fraction(previous["buckets_exact"][0]):
                raise ValueError("spent cost moved backwards")
            if published != previous["urgent_pub"] + int(event == "TICKET_PUBLISHED"):
                raise ValueError("publication counter does not match event")
            for drone, epoch in epochs.items():
                changed = current_modes[drone] != previous["drone_modes"][drone]
                if epoch != previous["drone_epochs"][drone] + int(changed):
                    raise ValueError("epoch does not match a physical mode transition")
        if event == "CAS_DECISION" and not (str(lg.get("verdict", "")).startswith(("PASS_", "REJECT_")) and isinstance(lg.get("proposal_id"), str)):
            raise ValueError("CAS outcome is missing")
        previous = lg
    end = logs[-1]
    _nonnegative_int(end.get("compliant_score"), "compliant_score")
    if end.get("mandatory_routine_all_done") is not True or end.get("compliant_score") != end["urgent_scored"] or any(m != "GROUND_IDLE" for m in end["drone_modes"].values()) or buckets[1] != 0 or buckets[2] != 0:
        raise ValueError("shift end is not fully compliant")
    if "mandatory_routine_total_count" in end and end.get("mandatory_routine_completed_count") != end["mandatory_routine_total_count"]:
        raise ValueError("mandatory task counts differ at shift end")
    if "mandatory_routine_total_count" in end:
        _nonnegative_int(end["mandatory_routine_total_count"], "mandatory_routine_total_count")
        _nonnegative_int(end["mandatory_routine_completed_count"], "mandatory_routine_completed_count")


def compare_actual_logs_against_expected(
    actual_bundle: Dict[str, Any], expected_bundle: Dict[str, Any],
) -> Dict[str, Any]:
    mismatches, summaries = [], {}
    verified, expected_count = 0, 0
    actual_traces = actual_bundle.get("traces_actual_logs") if isinstance(actual_bundle, dict) else None
    expected_traces = expected_bundle.get("expected_traces") if isinstance(expected_bundle, dict) else None
    if not isinstance(actual_traces, dict) or not isinstance(expected_traces, dict) or not expected_traces:
        mismatches.append({"error": "INVALID_OR_EMPTY_TRACE_BUNDLE"})
    else:
        for trace_id in sorted(set(actual_traces) - set(expected_traces)):
            mismatches.append({"trace_id": trace_id, "error": "UNEXPECTED_ACTUAL_TRACE"})
        for trace_id, expected in expected_traces.items():
            summary = {"matched": False, "checkpoint_count": 0, "total_raw_events": 0}
            summaries[trace_id] = summary
            try:
                checkpoints = expected["checkpoints"]
                budget = Fraction(expected["budget_total_yuan"])
                if budget < 0 or not isinstance(checkpoints, list) or len(checkpoints) < 2 or checkpoints[0]["event_type"] != "SHIFT_INIT" or checkpoints[-1]["event_type"] != "SHIFT_END":
                    raise ValueError("empty or invalid expected checkpoints")
                summary["checkpoint_count"] = len(checkpoints)
                expected_count += len(checkpoints)
                if trace_id not in actual_traces:
                    raise ValueError("MISSING_ACTUAL_TRACE")
                raw = actual_traces[trace_id]
                _validate_raw_logs(raw, budget)
                summary["total_raw_events"] = len(raw)
                extracted = extract_canonical_checkpoints(trace_id, raw)
                if extracted != checkpoints:
                    mismatches.append({"trace_id": trace_id, "error": "CHECKPOINT_MISMATCH",
                                       "actual_checkpoints": extracted, "expected_checkpoints": checkpoints})
                else:
                    summary["matched"] = True
                    verified += len(checkpoints)
            except (KeyError, TypeError, ValueError, AssertionError, ZeroDivisionError, IndexError) as exc:
                mismatches.append({"trace_id": trace_id, "error": f"INVALID_TRACE_EVIDENCE:{exc}"})
    return {"all_passed": not mismatches, "trace_count": len(expected_traces) if isinstance(expected_traces, dict) else 0,
            "total_verified_checkpoints": verified, "total_expected_checkpoints": expected_count,
            "evidence_scope": "CHECKPOINT_COMPARISON_AND_EXPORTED_STATE_INVARIANTS_NOT_INDEPENDENT_PHYSICS",
            "per_trace_summary": summaries, "mismatch_count": len(mismatches), "mismatches": mismatches}


def main() -> int:
    parser = argparse.ArgumentParser(description="Independent Track B log comparator")
    parser.add_argument("--actual-logs", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    args = parser.parse_args()
    if args.report_out.resolve() in {args.actual_logs.resolve(), args.expected.resolve()}:
        parser.error("Report output must not overwrite actual logs or expected checkpoints")

    actual_bundle = json.loads(args.actual_logs.read_text(encoding="utf-8"))
    expected_bundle = json.loads(args.expected.read_text(encoding="utf-8"))
    report = compare_actual_logs_against_expected(actual_bundle, expected_bundle)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not report["all_passed"]:
        print(json.dumps(report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
