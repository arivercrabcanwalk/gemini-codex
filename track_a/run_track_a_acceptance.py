"""Track A v5.1 Standalone Acceptance Runner & Independent Comparator."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from track_a.decoder_and_mask import (
    StationCoSortieMDP,
    canonicalize_station_solution_equivalence,
)
from track_a.frozen_adapter import (
    COST_TOL,
    ENERGY_TOL,
    FrozenEvaluatorBinding,
    frozen_solution_to_model_keyed,
    model_keyed_to_frozen_solution,
    wind_mps_to_m_per_min,
)
from track_a.lex_bopo_loss import compute_lex_bopo_batch_loss, verify_decoupled_training_vs_delivery


def _reject_nonfinite(value: str):
    raise ValueError("nonfinite JSON number: " + value)


def _check_blind_inputs(value):
    if isinstance(value, dict):
        if any(str(key).lower().startswith("expected") for key in value):
            raise ValueError("expected verdicts are forbidden in blind inputs")
        for child in value.values():
            _check_blind_inputs(child)
    elif isinstance(value, list):
        for child in value:
            _check_blind_inputs(child)


def execute_track_a_from_blind_inputs(
    inputs_path: Path,
    hn01_repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    raw_inputs = json.loads(inputs_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    _check_blind_inputs(raw_inputs)
    if raw_inputs.get("schema_version") != "v5.1-TrackA-BlindInputs":
        raise ValueError("unsupported Track A input schema")
    binding = FrozenEvaluatorBinding(hn01_repo_root=hn01_repo_root)

    # 1. Evaluate 6-tower DABS instance
    dabs_inst = raw_inputs["dabs_6tower_instance"]
    ref_routes = dabs_inst["synthetic_reference_solution"]["routes"]
    ref_eval = binding.evaluate_solution(dabs_inst, ref_routes)
    if not ref_eval["valid"]:
        raise ValueError("invalid synthetic reference: " + ref_eval["reason"])

    cand_results: Dict[str, Any] = {}
    for cid, c_obj in dabs_inst["candidate_solutions"].items():
        cand_results[cid] = binding.validate_candidate(
            dabs_inst,
            c_obj["routes"],
            ref_routes,
            wall_elapsed_s=c_obj["wall_elapsed_s"],
            wall_limit_s=c_obj["wall_limit_s"],
        )

    # Verify bidirectional adapter round-trip on DABS and SABS
    station_types_1 = {k: v["station_type"] for k, v in dabs_inst["stations"].items()}
    faster_routes = dabs_inst["candidate_solutions"]["COSORTIE_FASTER_ROUTE"]["routes"]
    perm_routes = dabs_inst["candidate_solutions"]["COSORTIE_SAME_SLOT_PERM_EQUIV"]["routes"]
    roundtrip_ok = (
        model_keyed_to_frozen_solution(
            station_types_1, frozen_solution_to_model_keyed(station_types_1, faster_routes)
        )
        == faster_routes
    )
    eq_faster = canonicalize_station_solution_equivalence(faster_routes["S1"])
    eq_perm = canonicalize_station_solution_equivalence(perm_routes["S1"])

    # 2. Evaluate unified policy action mask & separate trajectory witness harness
    mdp = StationCoSortieMDP(
        dabs_inst,
        station_id="S1",
        global_makespan_cap_min=float(dabs_inst["global_makespan_cap_min"]),
        binding=binding,
    )
    traj_results: Dict[str, Any] = {}
    for wid, w_obj in dabs_inst["trajectory_witnesses"].items():
        if type(w_obj["truncated"]) is not bool:
            raise ValueError("truncated must be a boolean")
        traj_results[wid] = mdp.evaluate_trajectory_witness(
            w_obj["actions"], truncated=w_obj["truncated"]
        )
        if traj_results[wid]["can_submit_to_validator"]:
            witness_routes = {"S1": mdp.actions_to_frozen_station_routes(w_obj["actions"])}
            witness_check = binding.validate_candidate(dabs_inst, witness_routes, ref_routes)
            if not witness_check["admissible"]:
                raise ValueError("complete fixture witness failed terminal certification")

    state_space_census = mdp.enumerate_canonical_state_space()

    # 3. Evaluate 2-station DABS + SABS cross-station compensation & SABS slot-0 empty rule
    inst_2s = raw_inputs["two_station_dabs_sabs_instance"]
    ref_2s = inst_2s["synthetic_reference_routes"]
    ref_2s_eval = binding.evaluate_solution(inst_2s, ref_2s)
    comp_2s_res = binding.validate_candidate(inst_2s, inst_2s["compensated_candidate_routes"], ref_2s)
    sabs_bad_res = binding.validate_candidate(inst_2s, inst_2s["sabs_slot0_nonempty_negative_routes"], ref_2s)

    # 4. Tolerance & unit probes
    probes = raw_inputs["tolerance_probes"]
    c_probe = probes["cost_cap_subtraction_probe"]
    e_probe = probes["energy_1e12_vs_1e7_probe"]
    tolerance_audit = {
        "wind_9mps_in_m_per_min": wind_mps_to_m_per_min(9.0),
        "cost_300_naive_add_check": (
            float(c_probe["candidate_cost_rmb"]) <= float(c_probe["budget_rmb"]) + COST_TOL
        ),
        "cost_300_frozen_sub_check": binding.cost_within_cap(
            float(c_probe["candidate_cost_rmb"]), float(c_probe["budget_rmb"]), COST_TOL
        ),
        "energy_5e10_over_wrong_1e7_check": (
            (float(e_probe["sortie_energy"]) - float(e_probe["sd_cap"])) <= COST_TOL
        ),
        "energy_5e10_over_frozen_1e12_check": (
            (float(e_probe["sortie_energy"]) - float(e_probe["sd_cap"])) <= ENERGY_TOL
        ),
    }

    # 5. Lex-BOPO preference loss & decoupled delivery audit
    decoupled = verify_decoupled_training_vs_delivery()
    loss_rows = [
        [
            {"id": "opt", "k1": 0, "k2": 0, "v3": 48.5, "v4": 21.5, "sum_logp": -5.4, "n_act": 9},
            {"id": "tie", "k1": 0, "k2": 0, "v3": 48.5, "v4": 22.5, "sum_logp": -4.5, "n_act": 9},
            {"id": "dead", "k1": 2, "k2": 6, "v3": 49.75, "v4": 27.5, "sum_logp": -1.2, "n_act": 4},
        ]
    ]
    loss_check = compute_lex_bopo_batch_loss(loss_rows, beta=1.0)

    return {
        "adapter_mode": binding.mode,
        "binding_evidence": {"source_sha256": binding.source_sha256,
                             "frozen_validation_calls": binding.frozen_validation_calls,
                             "scientific_validation": False,
                             "loss_implementation_scope": loss_check["implementation_scope"]},
        "dabs_6tower_reference": {
            "provenance_label": dabs_inst["synthetic_reference_solution"]["provenance_label"],
            "raw_cost_rmb": round(ref_eval["raw_cost_rmb"], 10),
            "makespan_min": round(ref_eval["makespan_min"], 10),
            "flight_min": round(ref_eval["flight_min"], 10),
            "service_min": round(ref_eval["service_min"], 10),
        },
        "dabs_6tower_candidates": {
            cid: {
                "admissible": res["admissible"],
                "reason": res["reason"],
                "raw_cost_rmb": round(res["raw_cost_rmb"], 10) if "raw_cost_rmb" in res else None,
                "makespan_min": round(res["makespan_min"], 10) if "makespan_min" in res else None,
                "cost_diff_rmb": round(res["cost_diff_rmb"], 10) if "cost_diff_rmb" in res else None,
            }
            for cid, res in cand_results.items()
        },
        "tie_break_winner_between_faster_and_costlier": (
            "COSORTIE_FASTER_ROUTE"
            if (
                cand_results["COSORTIE_FASTER_ROUTE"]["makespan_min"],
                cand_results["COSORTIE_FASTER_ROUTE"]["raw_cost_rmb"],
            )
            < (
                cand_results["TIE_MAKESPAN_COSTLIER_ROUTE"]["makespan_min"],
                cand_results["TIE_MAKESPAN_COSTLIER_ROUTE"]["raw_cost_rmb"],
            )
            else "TIE_MAKESPAN_COSTLIER_ROUTE"
        ),
        "slot_adapter_and_equivalence": {
            "bidirectional_roundtrip_exact": roundtrip_ok,
            "ordered_equal_between_perm_variants": eq_faster["ordered_key"] == eq_perm["ordered_key"],
            "intra_slot_multiset_equal_between_perm_variants": (
                eq_faster["intra_slot_sortie_multiset_key"] == eq_perm["intra_slot_sortie_multiset_key"]
            ),
        },
        "trajectory_witness_verdicts": traj_results,
        "state_space_no_hard_flight_prune": {
            "reachable_canonical_states": state_space_census["reachable_canonical_states"],
            "complete_terminal_states": state_space_census["complete_terminal_states"],
            "dead_end_states": state_space_census["dead_end_states"],
        },
        "two_station_cross_offset_and_sabs": {
            "reference_raw_cost_rmb": round(ref_2s_eval["raw_cost_rmb"], 10),
            "reference_makespan_min": round(ref_2s_eval["makespan_min"], 10),
            "compensated_admissible": comp_2s_res["admissible"],
            "compensated_raw_cost_rmb": round(comp_2s_res["raw_cost_rmb"], 10),
            "compensated_makespan_min": round(comp_2s_res["makespan_min"], 10),
            "compensated_cost_diff_rmb": round(comp_2s_res["cost_diff_rmb"], 10),
            "sabs_slot0_nonempty_admissible": sabs_bad_res["admissible"],
            "sabs_slot0_nonempty_reason": sabs_bad_res["reason"],
        },
        "tolerance_and_unit_audit": tolerance_audit,
        "decoupled_training_vs_delivery": decoupled,
        "lex_bopo_sample_batch_mean": round(loss_check["batch_mean"], 12),
    }


def compare_track_a_actual_vs_expected(actual: Dict[str, Any], expected: Dict[str, Any]) -> Dict[str, Any]:
    # Binding provenance is reported separately; it is never a scientific pass.
    ignored = {"adapter_mode", "binding_evidence"}
    act_cmp = {k: v for k, v in actual.items() if k not in ignored}
    exp_cmp = {k: v for k, v in expected.items() if k not in ignored}
    mismatches = []
    required = {"dabs_6tower_reference", "dabs_6tower_candidates", "tie_break_winner_between_faster_and_costlier",
                "slot_adapter_and_equivalence", "trajectory_witness_verdicts", "state_space_no_hard_flight_prune",
                "two_station_cross_offset_and_sabs", "tolerance_and_unit_audit", "decoupled_training_vs_delivery",
                "lex_bopo_sample_batch_mean"}
    if set(exp_cmp) != required:
        mismatches.append({"field": "expected_schema", "reason": "missing_or_unexpected_sections"})
    def strict_equal(left, right):
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return set(left) == set(right) and all(strict_equal(left[key], right[key]) for key in left)
        if isinstance(left, list):
            return len(left) == len(right) and all(strict_equal(a, b) for a, b in zip(left, right))
        if isinstance(left, float) and (not math.isfinite(left) or not math.isfinite(right)):
            return False
        return left == right
    for k in sorted(set(act_cmp.keys()) | set(exp_cmp.keys())):
        if k not in act_cmp or k not in exp_cmp or not strict_equal(act_cmp[k], exp_cmp[k]):
            mismatches.append({"field": k, "actual": act_cmp.get(k), "expected": exp_cmp.get(k)})
    return {
        "all_passed": len(mismatches) == 0,
        "checked_top_sections": len(exp_cmp),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Track A v5.1 blind acceptance & comparator")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=REPO_ROOT / "track_a" / "inputs" / "track_a_v5_1_instances.json",
    )
    parser.add_argument(
        "--expected",
        type=Path,
        default=REPO_ROOT / "track_a" / "expected" / "track_a_v5_1_expected.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "receipts" / "track_a_actual_output.json",
    )
    parser.add_argument("--hn01-repo-root", type=str, default=None)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    protected = [args.inputs.expanduser().resolve(), args.expected.expanduser().resolve()]
    protected_dirs = [REPO_ROOT / "track_a" / "inputs", REPO_ROOT / "track_a" / "expected"]
    if (any(output == path or (output.exists() and path.exists() and output.samefile(path)) for path in protected)
            or any(output.is_relative_to(path.resolve()) for path in protected_dirs)):
        parser.error("output must not alias an input, expected file or protected fixture directory")
    if output.exists():
        parser.error("output must be a new file; existing files are never overwritten")

    actual = execute_track_a_from_blind_inputs(args.inputs, hn01_repo_root=args.hn01_repo_root)
    expected = json.loads(args.expected.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    cmp_report = compare_track_a_actual_vs_expected(actual, expected)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents a concurrent replacement from becoming
    # an overwrite between the path checks and the completed comparison.
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(actual, indent=2, ensure_ascii=False, allow_nan=False))
    if not cmp_report["all_passed"]:
        print(json.dumps(cmp_report, indent=2, ensure_ascii=False), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "PASS_SYNTHETIC_ENGINEERING_FIXTURES",
                "adapter_mode": actual["adapter_mode"],
                "binding_evidence": actual["binding_evidence"],
                "inputs_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                "actual_output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                "expected_sha256": hashlib.sha256(args.expected.read_bytes()).hexdigest(),
                "checked_top_sections": cmp_report["checked_top_sections"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
