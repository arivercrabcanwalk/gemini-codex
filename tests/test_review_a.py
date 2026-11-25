"""Synthetic engineering regression checks; no research data or server runs.

Set TRACK_A_TEST_FROZEN_ROOT explicitly to exercise an available, locked local
frozen source tree. That compatibility check still uses only these fixtures.
"""
from __future__ import annotations

from collections import deque
import copy
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track_a.decoder_and_mask import StationCoSortieMDP, canonical_state_key
from track_a.frozen_adapter import FrozenEvaluatorBinding, frozen_solution_to_model_keyed, model_keyed_to_frozen_solution
from track_a.lex_bopo_loss import compute_lex_bopo_batch_loss, lex_rank_key
from track_a.run_track_a_acceptance import execute_track_a_from_blind_inputs, compare_track_a_actual_vs_expected


def tiny_instance(kind="SABS", service=1.0):
    return {"wind_xy_mps": [0.0, 0.0],
            "stations": {"S": {"station_type": kind, "xy_m": [0., 0.], "assigned_towers": [1]}},
            "towers": {"1": {"xy_m": [0., 0.], "t_srv_min": service, "energy": 0.01}}}


def tiny_routes(kind="SABS"):
    return {"S": [[], [[1]]]} if kind == "SABS" else {"S": [[[1]], []]}


class TestReviewAFailClosed(unittest.TestCase):
    def setUp(self):
        self.binding = FrozenEvaluatorBinding()

    def test_binding_requires_explicit_root(self):
        with patch.dict(os.environ, {"HN01_REPO_ROOT": "untrusted-implicit-root"}):
            self.assertEqual(FrozenEvaluatorBinding().mode, "STANDALONE_SYNTHETIC_ENGINEERING_SHIM")

    def test_fake_source_is_rejected_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            frozen = base / "production" / "frozen"
            for rel in ("env/__init__.py", "env/constant_audit.py", "baselines/__init__.py",
                        "baselines/problem.py", "baselines/wind_scorer.py"):
                source = frozen / rel
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("raise AssertionError('fake source executed')\n")
            before = list(sys.path)
            with self.assertRaises(ValueError):
                FrozenEvaluatorBinding(str(base))
            self.assertEqual(sys.path, before)

    def test_invalid_wind_is_rejected_even_for_zero_distance(self):
        for wind in ([15., 0.], [math.nan, 0.], [0., math.inf], [True, 0.]):
            with self.subTest(wind=wind), self.assertRaises(ValueError):
                self.binding.leg_minutes_xy([0., 0.], [0., 0.], wind)

    def test_vector_wind_units_and_heading(self):
        self.assertEqual(self.binding.leg_minutes_xy((0., 0.), (-1440., 0.), (9., 0.)), 4.)
        self.assertEqual(self.binding.leg_minutes_xy((0., 0.), (1440., 0.), (9., 0.)), 1.)

    def test_cap_rejects_nonfinite_negative_and_bool(self):
        for value in (math.nan, math.inf, -1., True, "1", 10 ** 1000):
            self.assertFalse(self.binding.cost_within_cap(value, 1.))
            self.assertFalse(self.binding.cost_within_cap(1., value))
            self.assertFalse(self.binding.cost_within_cap(1., 1., value))
        self.assertFalse(self.binding.cost_within_cap(300.0000001, 300.))

    def test_routes_do_not_coerce_ids(self):
        for value in (True, 1.0, 1.9, "1", None):
            routes = {"S": [[], [[value]]]}
            self.assertFalse(self.binding.evaluate_solution(tiny_instance(), routes)["valid"])
            with self.assertRaises(ValueError):
                frozen_solution_to_model_keyed({"S": "SABS"}, routes)
        with self.assertRaises(ValueError):
            model_keyed_to_frozen_solution({"S": "SABS"}, {"S": {"SD": [[1]], "TYPO": []}})

    def test_invalid_resource_values_cannot_validate(self):
        for field in ("energy", "t_srv_min"):
            for value in (math.nan, math.inf, -0.1, True, "0.1"):
                inst = tiny_instance()
                inst["towers"]["1"][field] = value
                with self.subTest(field=field, value=value):
                    self.assertFalse(self.binding.evaluate_solution(inst, tiny_routes())["valid"])

    def test_ownership_is_complete_and_disjoint(self):
        inst = tiny_instance()
        inst["stations"]["T"] = copy.deepcopy(inst["stations"]["S"])
        self.assertFalse(self.binding.evaluate_solution(inst, {"S": [[], [[1]]], "T": [[], []]})["valid"])
        inst = tiny_instance()
        inst["stations"]["S"]["assigned_towers"] = []
        self.assertFalse(self.binding.evaluate_solution(inst, tiny_routes())["valid"])

    def test_empty_trips_follow_frozen_ignored_semantics(self):
        base = self.binding.evaluate_solution(tiny_instance(), tiny_routes())
        empty = self.binding.evaluate_solution(tiny_instance(), {"S": [[[]], [[], [1], []]]})
        self.assertTrue(empty["valid"])
        self.assertEqual(base["raw_cost_rmb"], empty["raw_cost_rmb"])
        for routes in ({"S": [[], [None]]}, {"S": [[], None]}, {"S": [[], [[1]], []]}):
            self.assertFalse(self.binding.evaluate_solution(tiny_instance(), routes)["valid"])

    def test_cumulative_duty_is_not_reset_between_trips(self):
        inst = tiny_instance(service=250.)
        inst["towers"]["2"] = {"xy_m": [0., 0.], "t_srv_min": 250., "energy": .01}
        inst["stations"]["S"]["assigned_towers"] = [1, 2]
        result = self.binding.evaluate_solution(inst, {"S": [[], [[1], [2]]]})
        self.assertFalse(result["valid"])
        self.assertIn("CUMULATIVE_DUTY", result["reason"])
        mdp = StationCoSortieMDP(inst, "S", 1000.)
        st, _ = mdp.step(mdp.initial_state(), (1, 1))
        st, _ = mdp.step(st, (1, 0))
        self.assertEqual(mdp.step(st, (1, 2))[1], "CUMULATIVE_DUTY")

    def test_duty_and_energy_tolerances_match_frozen_comparators(self):
        inst = tiny_instance(service=480. + 5e-8)
        self.assertTrue(self.binding.evaluate_solution(inst, tiny_routes())["valid"])
        inst["towers"]["1"]["t_srv_min"] = 480. + 2e-7
        self.assertFalse(self.binding.evaluate_solution(inst, tiny_routes())["valid"])
        inst = tiny_instance()
        inst["towers"]["1"]["energy"] = .8100000005
        self.assertFalse(self.binding.evaluate_solution(inst, tiny_routes())["valid"])

    def test_wall_values_and_validator_finish_are_checked(self):
        for value in (math.nan, math.inf, -1., True, "0"):
            result = self.binding.validate_candidate(tiny_instance(), tiny_routes(), tiny_routes(), wall_elapsed_s=value)
            self.assertFalse(result["admissible"])
        with patch("track_a.frozen_adapter.perf_counter", side_effect=[10., 12.]):
            result = self.binding.validate_candidate(tiny_instance(), tiny_routes(), tiny_routes(), wall_elapsed_s=5., wall_limit_s=6.)
        self.assertEqual(result["reason"], "WALL_CLOCK_EXCEEDED_AT_VALIDATOR_FINISH")

    def test_closed_station_is_inactive(self):
        inst = tiny_instance()
        inst["stations"]["C"] = {"station_type": "CLOSED", "xy_m": [0., 0.], "assigned_towers": []}
        routes = {**tiny_routes(), "C": [[], []]}
        self.assertTrue(self.binding.evaluate_solution(inst, routes)["valid"])
        mdp = StationCoSortieMDP(inst, "C", 1000.)
        self.assertEqual(mdp.active_slots(), ())
        self.assertEqual(mdp.step(mdp.initial_state(), (0, 1))[1], "INACTIVE_CLOSED_STATION")

    def test_invalid_actions_and_states_are_rejected(self):
        mdp = StationCoSortieMDP(tiny_instance(), "S", 1000.)
        for action in ((-1, 1), (2, 1), (True, 1), (1, True), (1, 1.0), (1,), None):
            self.assertEqual(mdp.step(mdp.initial_state(), action), (None, "INVALID_ACTION"))
        with self.assertRaises(ValueError):
            mdp.actions_to_frozen_station_routes([(True, 1)])
        st = list(mdp.initial_state())
        st[4] = math.nan
        self.assertEqual(mdp.step(tuple(st), (1, 1)), (None, "INVALID_STATE"))

    def test_coverage_and_parameters_are_used_by_mask(self):
        inst = tiny_instance()
        inst["towers"]["1"]["xy_m"] = [4000. + 1e-5, 0.]
        mdp = StationCoSortieMDP(inst, "S", 1000.)
        self.assertEqual(mdp.step(mdp.initial_state(), (1, 1))[1], "COVERAGE_EXCEEDED")
        inst = tiny_instance()
        inst["parameters"] = {"small_battery": .005}
        mdp = StationCoSortieMDP(inst, "S", 1000.)
        self.assertEqual(mdp.step(mdp.initial_state(), (1, 1))[1], "BATT_CAP")

    def test_return_sentinel_collision_is_explicit(self):
        inst = tiny_instance()
        inst["towers"] = {"0": inst["towers"]["1"]}
        inst["stations"]["S"]["assigned_towers"] = [0]
        with self.assertRaises(ValueError):
            StationCoSortieMDP(inst, "S", 1000.)

    def test_resource_key_keeps_distinct_binary64_values(self):
        left = ((1,), (0, 0), (1e-13, 0.), (0., 0.), 0.)
        right = ((1,), (0, 0), (2e-13, 0.), (0., 0.), 0.)
        self.assertNotEqual(canonical_state_key(left), canonical_state_key(right))

    def test_exact_census_matches_independent_unrounded_tuple_bfs(self):
        raw = json.loads((ROOT / "track_a/inputs/track_a_v5_1_instances.json").read_text())
        mdp = StationCoSortieMDP(raw["dabs_6tower_instance"], "S1", 50.)
        queue = deque([mdp.initial_state()])
        seen, complete, dead = {mdp.initial_state()}, 0, 0
        while queue:
            st = queue.popleft()
            if not st[0] and st[1] == (0, 0):
                complete += 1
                continue
            children = [mdp.step(st, (slot, target))[0] for slot in (0, 1) for target in (0,) + st[0]]
            children = [child for child in children if child is not None]
            if not children:
                dead += 1
            for child in children:
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        self.assertEqual((len(seen), complete, dead), (7930, 30, 912))
        result = mdp.enumerate_canonical_state_space()
        self.assertEqual((result["reachable_canonical_states"], result["complete_terminal_states"], result["dead_end_states"]), (len(seen), complete, dead))


class TestReviewARunnerAndScalarSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs = ROOT / "track_a/inputs/track_a_v5_1_instances.json"
        cls.actual = execute_track_a_from_blind_inputs(cls.inputs)
        cls.expected = json.loads((ROOT / "track_a/expected/track_a_v5_1_expected.json").read_text())

    def test_original_fixture_meaning_is_preserved(self):
        self.assertTrue(compare_track_a_actual_vs_expected(self.actual, self.expected)["all_passed"])
        self.assertEqual(self.actual["dabs_6tower_reference"]["raw_cost_rmb"], 91.4)
        self.assertEqual(self.actual["decoupled_training_vs_delivery"]["delivered_id"], "A")
        self.assertEqual(self.actual["binding_evidence"]["frozen_validation_calls"], 0)
        self.assertFalse(self.actual["binding_evidence"]["scientific_validation"])

    def test_comparator_does_not_pass_empty_or_bool_numeric_equivalence(self):
        self.assertFalse(compare_track_a_actual_vs_expected({}, {})["all_passed"])
        bad = copy.deepcopy(self.actual)
        bad["dabs_6tower_candidates"]["COSORTIE_FASTER_ROUTE"]["admissible"] = 1
        self.assertFalse(compare_track_a_actual_vs_expected(bad, self.expected)["all_passed"])
        bad = copy.deepcopy(self.actual)
        del bad["dabs_6tower_candidates"]["A_NEG_WRONG_SLOT_INVERSION"]["raw_cost_rmb"]
        self.assertFalse(compare_track_a_actual_vs_expected(bad, self.expected)["all_passed"])

    def test_blind_input_pollution_is_rejected(self):
        raw = json.loads(self.inputs.read_text())
        raw["dabs_6tower_instance"]["expected_verdict"] = True
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "inputs.json"
            path.write_text(json.dumps(raw))
            with self.assertRaises(ValueError):
                execute_track_a_from_blind_inputs(path)

    def test_cli_cannot_overwrite_expected(self):
        result = subprocess.run([sys.executable, str(ROOT / "track_a/run_track_a_acceptance.py"), "--write-expected"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unrecognized arguments", result.stderr)

    def test_cli_output_cannot_alias_goldens_or_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            expected = base / "expected.json"
            original = json.dumps(self.expected).encode()
            expected.write_bytes(original)
            aliases = [expected, base / "hardlink.json", base / "symlink.json"]
            os.link(expected, aliases[1])
            aliases[2].symlink_to(expected)
            for output in aliases:
                result = subprocess.run([sys.executable, str(ROOT / "track_a/run_track_a_acceptance.py"),
                                         "--inputs", str(self.inputs), "--expected", str(expected), "--output", str(output)],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn("must not alias", result.stderr)
                self.assertEqual(expected.read_bytes(), original)
            result = subprocess.run([sys.executable, str(ROOT / "track_a/run_track_a_acceptance.py"),
                                     "--output", str(self.inputs)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("must not alias", result.stderr)

    def test_loss_is_explicitly_scalar_and_checks_degenerate_counts(self):
        base = {"id": "a", "k1": 0, "k2": 0, "v3": 1., "v4": 0., "sum_logp": -.5, "n_act": 1}
        other = {**base, "id": "b", "v3": 2., "sum_logp": -.8}
        result = compute_lex_bopo_batch_loss([[base, other], [base, base]])
        self.assertEqual(result["implementation_scope"], "SCALAR_PAIRWISE_SMOKE_NO_REFERENCE_POLICY")
        self.assertEqual(result["informative_groups"], 1)
        for change in ({"n_act": 0}, {"n_act": True}, {"sum_logp": math.nan}, {"sum_logp": .1}):
            with self.assertRaises(ValueError):
                compute_lex_bopo_batch_loss([[{**base, **change}, other]])
        for beta in (0., -1., math.nan, math.inf, True):
            with self.assertRaises(ValueError):
                compute_lex_bopo_batch_loss([[base, other]], beta)
        for change in ({"k1": 3}, {"k1": True}, {"k2": 1}, {"v3": math.nan}):
            with self.assertRaises(ValueError):
                lex_rank_key({**base, **change})


@unittest.skipUnless(os.environ.get("TRACK_A_TEST_FROZEN_ROOT"), "explicit locked local frozen root not supplied")
class TestReviewALocalFrozenCompatibility(unittest.TestCase):
    def setUp(self):
        self.binding = FrozenEvaluatorBinding(os.environ["TRACK_A_TEST_FROZEN_ROOT"])

    def test_real_api_schema_anchor_and_units_are_exercised(self):
        raw = json.loads((ROOT / "track_a/inputs/track_a_v5_1_instances.json").read_text())
        inst = raw["dabs_6tower_instance"]
        anchor = inst["synthetic_reference_solution"]["routes"]
        candidate = inst["candidate_solutions"]["COSORTIE_FASTER_ROUTE"]["routes"]
        with patch.object(self.binding, "_frozen_validate_candidate", wraps=self.binding._frozen_validate_candidate) as validator:
            result = self.binding.validate_candidate(inst, candidate, anchor)
        self.assertTrue(result["admissible"])
        native, solution, native_anchor = validator.call_args.args
        self.assertEqual(native.params.speed, 900.)
        self.assertEqual(native_anchor.wind, (540., 0.))
        self.assertEqual(solution["station_types"], [2])
        self.assertEqual(len(solution["routes"][0]), 2)
        self.assertEqual(solution["instance_sha256"], native.sha256)
        self.assertEqual(result["frozen_validation"]["anchor_solution_sha256"], native_anchor.solution_sha256)
        native_anchor.verify(native)
        detached = native_anchor.solution()
        detached["routes"][0] = [[], []]
        native_anchor.verify(native)
        with self.assertRaises(ValueError):
            replace(native_anchor, cost=native_anchor.cost + 1.).verify(native)
        self.assertEqual(self.binding.frozen_validation_calls, 1)
        self.assertEqual(len(self.binding.source_sha256), 3)

    def test_imports_are_isolated_from_existing_env_modules(self):
        existing_env, existing_audit = ModuleType("env"), ModuleType("env.constant_audit")
        existing_audit.heading_leg = lambda *args: -123
        before = list(sys.path)
        with patch.dict(sys.modules, {"env": existing_env, "env.constant_audit": existing_audit}):
            binding = FrozenEvaluatorBinding(os.environ["TRACK_A_TEST_FROZEN_ROOT"])
            self.assertEqual(binding.leg_minutes_xy((0., 0.), (-1440., 0.), (9., 0.)), 4.)
            self.assertIs(sys.modules["env"], existing_env)
            self.assertIs(sys.modules["env.constant_audit"], existing_audit)
        self.assertEqual(sys.path, before)

    def test_direct_and_shim_match_raw_unrounded_fixture_metrics(self):
        raw = json.loads((ROOT / "track_a/inputs/track_a_v5_1_instances.json").read_text())
        inst = raw["two_station_dabs_sabs_instance"]
        inst["wind_xy_mps"] = [3.1, -4.2]
        for routes in (inst["synthetic_reference_routes"], inst["compensated_candidate_routes"]):
            direct = self.binding.evaluate_solution(inst, routes)
            shim = FrozenEvaluatorBinding().evaluate_solution(inst, routes)
            self.assertTrue(direct["valid"])
            for field in ("raw_cost_rmb", "makespan_min", "total_duty_min"):
                self.assertEqual(direct[field], shim[field])

    def test_identity_preserves_station_and_target_order(self):
        inst = tiny_instance()
        inst["stations"]["T"] = {"station_type": "CLOSED", "xy_m": [1., 2.], "assigned_towers": []}
        other = copy.deepcopy(inst)
        other["stations"] = dict(reversed(list(other["stations"].items())))
        self.assertNotEqual(self.binding.to_native_instance(inst).sha256, self.binding.to_native_instance(other).sha256)


if __name__ == "__main__":
    unittest.main()
