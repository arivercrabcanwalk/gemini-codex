"""Comprehensive unit test suite covering every Review §2.A, §2.B, and §3 requirement."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stats.paired_inference import (
    betting_martingale_pvalue,
    empirical_bernstein_lower_bound,
    evaluate_line_a_g0_gate,
    min_n_for_eb_zero_variance,
    min_n_for_gaffke_zero_variance,
)
from track_a.decoder_and_mask import StationCoSortieMDP
from track_a.frozen_adapter import COST_TOL, ENERGY_TOL, FrozenEvaluatorBinding
from track_a.run_track_a_acceptance import execute_track_a_from_blind_inputs
from track_b.cape_executor import OriginalDomainCapeExecutor, SharedWitnessCompleter, quantize_flight_ms
from track_b.independent_comparator import compare_actual_logs_against_expected


class TestAcceptancePackV51(unittest.TestCase):
    def test_track_a_slot_order_and_exact_costs(self) -> None:
        inputs_path = REPO_ROOT / "track_a" / "inputs" / "track_a_v5_1_instances.json"
        res = execute_track_a_from_blind_inputs(inputs_path)

        # Review §2.A.2: exact Liu §5.1 + wind_scorer.py costs: 91.40 / 90.15 / 90.65 RMB
        self.assertAlmostEqual(res["dabs_6tower_reference"]["raw_cost_rmb"], 91.40, places=9)
        self.assertAlmostEqual(
            res["dabs_6tower_candidates"]["COSORTIE_FASTER_ROUTE"]["raw_cost_rmb"], 90.15, places=9
        )
        self.assertAlmostEqual(
            res["dabs_6tower_candidates"]["TIE_MAKESPAN_COSTLIER_ROUTE"]["raw_cost_rmb"], 90.65, places=9
        )
        self.assertEqual(res["tie_break_winner_between_faster_and_costlier"], "COSORTIE_FASTER_ROUTE")

        # Review §2.A.1: wrong-slot inversion (0.882 into slot 1 SD=0.81) & SABS non-empty slot 0 rejected
        wrong_slot = res["dabs_6tower_candidates"]["A_NEG_WRONG_SLOT_INVERSION"]
        self.assertFalse(wrong_slot["admissible"])
        self.assertIn("BATTERY_EXCEEDED:S1:slot1(SD)", wrong_slot["reason"])

        sabs_check = res["two_station_cross_offset_and_sabs"]
        self.assertTrue(sabs_check["compensated_admissible"])
        self.assertAlmostEqual(sabs_check["compensated_cost_diff_rmb"], -1.25, places=9)
        self.assertFalse(sabs_check["sabs_slot0_nonempty_admissible"])
        self.assertEqual(sabs_check["sabs_slot0_nonempty_reason"], "SABS_SLOT0_NONEMPTY:S2")

    def test_untrusted_frozen_root_is_not_a_server_receipt(self) -> None:
        # Presence of a directory or a made-up scorer is not evaluator provenance.
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((FileNotFoundError, ImportError, ValueError, RuntimeError)):
                FrozenEvaluatorBinding(hn01_repo_root=tmp)

    def test_track_a_tolerances_and_unified_mask(self) -> None:
        inputs_path = REPO_ROOT / "track_a" / "inputs" / "track_a_v5_1_instances.json"
        raw = json.loads(inputs_path.read_text(encoding="utf-8"))
        binding = FrozenEvaluatorBinding()
        mdp = StationCoSortieMDP(raw["dabs_6tower_instance"], "S1", 50.0, binding=binding)

        st0 = mdp.initial_state()
        st1, rsn1 = mdp.step(st0, (1, 1))
        self.assertEqual(rsn1, "OK")
        assert st1 is not None
        valids_step2, _ = mdp.probe_candidates_deterministic(st1)
        self.assertIn((1, 0), valids_step2)

        self.assertEqual(ENERGY_TOL, 1e-12)
        self.assertEqual(COST_TOL, 1e-7)
        self.assertFalse((0.8100000005 - 0.81) <= ENERGY_TOL)
        self.assertFalse(binding.cost_within_cap(300.0000001, 300.0, COST_TOL))

        with self.assertRaises(ValueError):
            mdp.actions_to_frozen_station_routes(
                raw["dabs_6tower_instance"]["trajectory_witnesses"]["TRAJ_TRUNCATED_UNCLOSED"]["actions"]
            )

    def test_track_b_epoch_table_b3_and_b6_exact_arithmetic(self) -> None:
        inputs_path = REPO_ROOT / "track_b" / "inputs" / "track_b_v5_1_inputs.json"
        expected_path = REPO_ROOT / "track_b" / "expected" / "track_b_v5_1_expected.json"
        raw_inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        expected_bundle = json.loads(expected_path.read_text(encoding="utf-8"))

        polluted = copy.deepcopy(raw_inputs["traces"][2])
        polluted["expected_checkpoints"] = []
        with self.assertRaises(ValueError):
            OriginalDomainCapeExecutor(polluted)

        b3_stale = copy.deepcopy(raw_inputs["traces"][2])
        b3_stale["exogenous_events"][1]["snapshot_epoch"] = 3
        sim_stale = OriginalDomainCapeExecutor(b3_stale)
        logs_stale = sim_stale.run_to_shift_end()
        cas_stale = [lg for lg in logs_stale if lg["event_type"] == "CAS_DECISION"][0]
        self.assertEqual(cas_stale["verdict"], "REJECT_STALE_EPOCH_OR_WITNESS")
        self.assertEqual(cas_stale["actual_epoch"], 4)

        b3_ok_logs = OriginalDomainCapeExecutor(raw_inputs["traces"][2]).run_to_shift_end()
        b3_types = [(lg["t_ms"], lg["event_type"]) for lg in b3_ok_logs]
        self.assertIn((2160000, "CAS_DECISION"), b3_types)
        self.assertIn((3006000, "HOLD_ALARM_FIRED"), b3_types)
        self.assertIn((3114000, "GLOBAL_HOVER_BUDGET_ALARM_FIRED"), b3_types)

        b6_sim = OriginalDomainCapeExecutor(raw_inputs["traces"][5])
        b6_logs = b6_sim.run_to_shift_end()
        b6_cas = {lg["proposal_id"]: lg for lg in b6_logs if lg["event_type"] == "CAS_DECISION"}

        sla_diag = b6_cas["P_B6_SLA_LATE"]["rollout_diag"]["sortie_metrics"][0]
        self.assertAlmostEqual(sla_diag["energy_float"], 0.368, places=6)
        self.assertEqual(sla_diag["last_service_end_ms"], 972000)
        self.assertEqual(sla_diag["return_station_ms"], 1152000)

        batt_diag = b6_cas["P_B6_BATT_OVER"]["rollout_diag"]["sortie_metrics"][0]
        self.assertAlmostEqual(batt_diag["energy_float"], 1.015937, places=6)

        shift_diag = b6_cas["P_B6_SHIFT_OVER"]["rollout_diag"]["sortie_metrics"]
        self.assertAlmostEqual(shift_diag[0]["energy_float"], 0.649537, places=6)
        self.assertAlmostEqual(shift_diag[1]["energy_float"], 0.444, places=6)
        self.assertEqual(shift_diag[1]["return_station_ms"], 6180000)

        self.assertEqual(quantize_flight_ms((0, 0), (200, 0)), 13334)

        actual_logs_map = {
            tr["trace_id"]: OriginalDomainCapeExecutor(tr).run_to_shift_end()
            for tr in raw_inputs["traces"]
        }
        cmp_res = compare_actual_logs_against_expected(
            {"traces_actual_logs": actual_logs_map}, expected_bundle
        )
        self.assertTrue(cmp_res["all_passed"])

    def test_shared_completer_uniqueness_and_no_feasible_witness(self) -> None:
        inputs_path = REPO_ROOT / "track_b" / "inputs" / "track_b_v5_1_inputs.json"
        raw_inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
        b6_spec = {
            "trace_id": "B6_COMPLETER_UNIT",
            "initial_state": raw_inputs["traces"][5]["initial_state"],
            "exogenous_events": [],
        }
        sim = OriginalDomainCapeExecutor(b6_spec)
        sim._process_internal_at_now()
        sim._settle_to(240000)
        sim._process_internal_at_now()
        sim._settle_to(540000)
        sim._process_internal_at_now()
        sim._settle_to(612000)
        sim.tickets["e_sla"] = {
            "ticket_id": "e_sla",
            "node_id": "r_near",
            "sla_ms": 666000,
            "srv_ms": 300000,
            "status": "UNCLAIMED",
        }
        comp_bad, status_bad = SharedWitnessCompleter.complete_after_explicit_next_step(
            sim, "S1", 1, ("TICKET", "e_sla", "r_near")
        )
        self.assertIsNone(comp_bad)
        self.assertEqual(status_bad, "NO_FEASIBLE_WITNESS")

    def test_stats_power_and_line_a_g0_gate(self) -> None:
        lb_95 = empirical_bernstein_lower_bound([0.02] * 60, -1.0, 1.0, 0.05)
        lb_holm = empirical_bernstein_lower_bound([0.02] * 60, -1.0, 1.0, 0.0125)
        self.assertAlmostEqual(lb_95, -0.271776, places=5)
        self.assertAlmostEqual(lb_holm, -0.381426, places=5)
        self.assertEqual(min_n_for_eb_zero_variance(0.02, 2.0, 0.0125), 1186)
        self.assertEqual(min_n_for_gaffke_zero_variance(0.02, 0.0125), 222)

        # Exact finite-sample Betting Martingale rejects +2pp (SD=0.03) on N=300 at Holm 0.0125
        sample = [0.05 if i % 2 == 0 else -0.01 for i in range(300)]
        self.assertLess(betting_martingale_pvalue(sample), 0.0125)

        # Line A 5% registered gate on N=80 base maps (+6% passes; +4% fails 5% threshold)
        anc = [100.0] * 160
        non_rl = [96.5] * 160
        cand_pass = [93.5 if i % 2 == 0 else 94.5 for i in range(160)]  # mean +6.0%, (1.06)^80 >> 20
        cand_fail = [95.5 if i % 2 == 0 else 96.5 for i in range(160)]  # mean +4.0% < 5.0%
        g_pass = evaluate_line_a_g0_gate(anc, cand_pass, non_rl)
        g_fail = evaluate_line_a_g0_gate(anc, cand_fail, non_rl)
        self.assertTrue(g_pass["vs_author_anchor"]["meets_registered_5pct_target"])
        self.assertTrue(g_pass["vs_strongest_non_rl"]["beats_strongest_non_rl"])
        self.assertFalse(g_fail["vs_author_anchor"]["meets_registered_5pct_target"])


if __name__ == "__main__":
    unittest.main()
