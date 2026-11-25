"""Track B v5.1 Blind Executor Runner & Separate Comparator Invocation."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track_b.cape_executor import (
    FROZEN_MODE_TRANSITION_TABLE,
    OriginalDomainCapeExecutor,
    SharedWitnessCompleter,
    reject_expected_keys,
)
from track_b.independent_comparator import (
    compare_actual_logs_against_expected,
)


def run_blind_executor_all_traces(inputs_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    raw_inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    reject_expected_keys(raw_inputs)
    traces = raw_inputs.get("traces")
    if not isinstance(traces, list) or not traces:
        raise ValueError("Blind inputs require a nonempty traces list")
    trace_ids = [tr.get("trace_id") for tr in traces]
    if any(not isinstance(tid, str) or not tid for tid in trace_ids) or len(set(trace_ids)) != len(trace_ids):
        raise ValueError("Missing or duplicate trace_id")

    traces_actual_logs: Dict[str, Any] = {}
    for tr in raw_inputs["traces"]:
        sim = OriginalDomainCapeExecutor(tr)
        logs = sim.run_to_shift_end()
        traces_actual_logs[tr["trace_id"]] = logs

    # A separate synthetic completer diagnostic; no learned policy is evaluated.
    b6_inputs = [t for t in traces if t["trace_id"] == "B6_ORIG_HOVER_SINGLE_CAUSE_CONTROLS"]
    deterministic_completer_check = {"status": "NOT_APPLICABLE", "trained_policy": False}
    timing_profile = {"status": "NOT_APPLICABLE", "wall_decision_to_commit_ms": None}
    if b6_inputs:
        b6_input = b6_inputs[0]
        b6_step_spec = {
            "trace_id": "B6_SHARED_COMPLETER_STEP_TEST",
            "initial_state": b6_input["initial_state"],
            "exogenous_events": [],
        }
        sim_step = OriginalDomainCapeExecutor(b6_step_spec)
        sim_step._process_internal_at_now()
        sim_step._settle_to(240000)
        sim_step._process_internal_at_now()
        sim_step._settle_to(540000)
        sim_step._process_internal_at_now()
        sim_step._settle_to(600000)
        sim_step.tickets["e_ok"] = {
            "ticket_id": "e_ok",
            "node_id": "r_near",
            "sla_ms": 1200000,
            "srv_ms": 300000,
            "status": "UNCLAIMED",
        }

        def greedy_physical_selector(_snap, feasible_list):
            for act, comp in feasible_list:
                if act[0] == "STEP" and act[3][0] == "TICKET" and act[3][1] == "e_ok":
                    return act
            return feasible_list[0][0]

        timed_step = SharedWitnessCompleter.execute_timed_decision_step(
            sim_step, greedy_physical_selector, proposal_id="P_SHARED_COMPLETER_E_OK", proposal_seq=1
        )
        deterministic_completer_check = {
            "chosen_action": timed_step["chosen_action"],
            "feasible_action_count": timed_step["feasible_action_count"],
            "cas_verdict": timed_step["cas_verdict"],
            "status": "SYNTHETIC_ENGINEERING_DIAGNOSTIC",
            "trained_policy": False,
        }
        timing_profile = {
            "wall_decision_to_commit_ms": timed_step["wall_decision_to_commit_ms"],
            "within_500ms_fast_budget": timed_step["wall_decision_to_commit_ms"] <= 500.0,
            "decision_ready_ms": timed_step["decision_ready_ms"],
            "simulated_decision_elapsed_ms": timed_step["simulated_decision_elapsed_ms"],
            "latency_model": "ceil(measured snapshot-to-ready wall ms), physical events processed before CAS",
        }
    deterministic_bundle = {
        "evidence_scope": "SYNTHETIC_ENGINEERING_REPLAY_ONLY_NOT_RESEARCH_OR_RL_RESULTS",
        "fixture_proposal_timing": "EXOGENOUS_INPUT_TIMESTAMPS_NOT_MEASURED_POLICY_LATENCY",
        "frozen_mode_transition_table": list(FROZEN_MODE_TRANSITION_TABLE),
        "b3_full_trace_first": traces_actual_logs.get("B3_ORIG_DUAL_HOVER_ALARMS", []),
        "shared_completer_and_attribution_check": deterministic_completer_check,
        "traces_actual_logs": traces_actual_logs,
    }
    return deterministic_bundle, timing_profile


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track B blind executor & separate comparator")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=REPO_ROOT / "track_b" / "inputs" / "track_b_v5_1_inputs.json",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=REPO_ROOT / "track_b" / "expected" / "track_b_v5_1_expected.json",
    )
    parser.add_argument(
        "--b3-trace-out",
        type=Path,
        default=REPO_ROOT / "receipts" / "track_b_b3_full_trace.json",
    )
    parser.add_argument(
        "--actual-logs-out",
        type=Path,
        default=REPO_ROOT / "receipts" / "track_b_actual_logs.json",
    )
    parser.add_argument(
        "--timing-out",
        type=Path,
        default=REPO_ROOT / "receipts" / "track_b_completer_timing.json",
    )
    parser.add_argument(
        "--comparison-out",
        type=Path,
        default=REPO_ROOT / "receipts" / "track_b_comparison_report.json",
    )
    parser.add_argument("--blind-only", action="store_true", help="Emit actual logs without opening the expected file")
    args = parser.parse_args()

    outputs = [args.b3_trace_out, args.actual_logs_out, args.timing_out, args.comparison_out]
    resolved_outputs = [path.resolve() for path in outputs]
    protected = {args.inputs.resolve(), args.expected.resolve(),
                 (REPO_ROOT / "track_b" / "inputs" / "track_b_v5_1_inputs.json").resolve(),
                 (REPO_ROOT / "track_b" / "expected" / "track_b_v5_1_expected.json").resolve()}
    if len(set(resolved_outputs)) != len(resolved_outputs) or any(path in protected for path in resolved_outputs):
        parser.error("Output paths must be distinct and must not overwrite inputs or expected checkpoints")

    actual_bundle, timing_profile = run_blind_executor_all_traces(args.inputs)
    args.b3_trace_out.parent.mkdir(parents=True, exist_ok=True)
    args.b3_trace_out.write_text(
        json.dumps(actual_bundle["b3_full_trace_first"], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.actual_logs_out.parent.mkdir(parents=True, exist_ok=True)
    args.actual_logs_out.write_text(
        json.dumps(actual_bundle, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    args.timing_out.parent.mkdir(parents=True, exist_ok=True)
    args.timing_out.write_text(
        json.dumps(timing_profile, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.blind_only:
        print(json.dumps({"status": "BLIND_EXECUTION_ONLY", "trace_count": len(actual_bundle["traces_actual_logs"]),
                          "comparison_performed": False}, indent=2))
        return 0

    expected_bundle = json.loads(args.expected.read_text(encoding="utf-8"))
    cmp_report = compare_actual_logs_against_expected(actual_bundle, expected_bundle)
    args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
    args.comparison_out.write_text(json.dumps(cmp_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not cmp_report["all_passed"]:
        print(json.dumps(cmp_report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "status": "PASS",
                "blind_inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "b3_full_trace_sha256": hashlib.sha256(args.b3_trace_out.read_bytes()).hexdigest(),
                "actual_logs_sha256": hashlib.sha256(args.actual_logs_out.read_bytes()).hexdigest(),
                "expected_sha256": hashlib.sha256(args.expected.read_bytes()).hexdigest(),
                "comparison_report_sha256": hashlib.sha256(args.comparison_out.read_bytes()).hexdigest(),
                "total_verified_checkpoints": cmp_report["total_verified_checkpoints"],
                "completer_wall_ms": timing_profile["wall_decision_to_commit_ms"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
