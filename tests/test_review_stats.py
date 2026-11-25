"""Deterministic engineering checks, not experiments or research admission."""
from __future__ import annotations

import itertools
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stats.paired_inference import (
    betting_martingale_lower_bound,
    betting_martingale_pvalue,
    empirical_bernstein_lower_bound,
    evaluate_line_a_g0_gate,
    min_n_for_eb_zero_variance,
    min_n_for_gaffke_zero_variance,
    run_statistical_power_and_calibration_audit,
    studentized_paired_t_permutation_pvalue,
)


class TestReviewedStats(unittest.TestCase):
    def test_bounded_inference_rejects_invalid_inputs_without_clipping(self):
        for values in ([], [math.nan], [math.inf], [-math.inf], [1.0000000000001], [-1.0000000000001]):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    betting_martingale_pvalue(values)
                with self.assertRaises(ValueError):
                    betting_martingale_lower_bound(values)
        for clip in (-0.01, 1.0, 2.0, math.nan):
            with self.assertRaises(ValueError):
                betting_martingale_pvalue([0.5], c_clip=clip)
        self.assertEqual(betting_martingale_pvalue([1.0], c_clip=0.0), 1.0)
        self.assertAlmostEqual(betting_martingale_pvalue([1.0], c_clip=0.1), 1 / 1.1)

    def test_alpha_tolerance_and_eb_sample_validation(self):
        for alpha in (0.0, 1.0, -0.1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                betting_martingale_lower_bound([0.02] * 4, alpha=alpha)
            with self.assertRaises(ValueError):
                empirical_bernstein_lower_bound([0.02] * 4, alpha=alpha)
        for tol in (0.0, -0.1, 1.0, math.nan):
            with self.assertRaises(ValueError):
                betting_martingale_lower_bound([0.02] * 4, tol=tol)
        for values in ([], [0.0], [math.nan, 0.0], [-2.0, 0.0]):
            with self.assertRaises(ValueError):
                empirical_bernstein_lower_bound(values)
        with self.assertRaises(ValueError):
            empirical_bernstein_lower_bound([0.0, 0.0], a=1.0, b=-1.0)
        self.assertEqual(empirical_bernstein_lower_bound([-1.0, -1.0]), -1.0)

    def test_small_sample_barriers_remain_necessary_only(self):
        self.assertEqual(min_n_for_eb_zero_variance(), 1186)
        self.assertEqual(min_n_for_gaffke_zero_variance(alpha=0.05), 152)
        self.assertEqual(min_n_for_gaffke_zero_variance(alpha=0.0125), 222)
        # The least-favorable necessary N does not ensure this test rejects.
        self.assertGreater(betting_martingale_pvalue([0.02] * 152), 0.05)
        self.assertGreater(betting_martingale_pvalue([0.02] * 222), 0.0125)
        self.assertLess(betting_martingale_pvalue([0.02] * 300), 0.0125)
        for delta in (0.0, -0.02, math.nan, math.inf):
            with self.assertRaises(ValueError):
                min_n_for_eb_zero_variance(delta=delta)
            with self.assertRaises(ValueError):
                min_n_for_gaffke_zero_variance(delta=delta)
        with self.assertRaises(ValueError):
            min_n_for_gaffke_zero_variance(delta=1.1)

    def test_zero_variance_lcb_matches_independent_closed_form(self):
        delta, n, alpha, tol = 0.06, 160, 0.05, 1e-8
        bound = betting_martingale_lower_bound([delta] * n, alpha=alpha, tol=tol)

        def constant_path_log_wealth(mean):
            shifted = (delta - mean) / (1 + mean)
            return math.log1p(0.5 * shifted) + (n - 1) * math.log1p(0.98 * shifted)

        self.assertGreater(bound, 0.0)
        self.assertGreaterEqual(constant_path_log_wealth(bound), -math.log(alpha))
        self.assertLess(constant_path_log_wealth(bound + 2 * tol), -math.log(alpha))
        self.assertEqual(betting_martingale_lower_bound([-1.0] * 4), -1.0)
        self.assertTrue(math.isfinite(betting_martingale_lower_bound([0.02] * 4, tol=1e-30)))

    def test_finite_null_tree_checks_type1_and_coverage(self):
        # Enumerate all 64 fair independent +/-1 paths: no Monte Carlo or maps.
        paths = list(itertools.product((-1.0, 1.0), repeat=6))
        rejected = sum(betting_martingale_pvalue(path) <= 0.05 for path in paths)
        uncovered = sum(betting_martingale_lower_bound(path) > 0.0 for path in paths)
        self.assertGreater(rejected, 0)
        self.assertLessEqual(rejected / len(paths), 0.05)
        self.assertLessEqual(uncovered / len(paths), 0.05)

    def test_signflip_symmetry_is_not_a_mean_null(self):
        # Mean-zero law: .99 mass at +.01 and .01 at -.99. Its all-positive
        # path alone causes sign-flip rejection with probability .99**6 > .94.
        path = [0.01] * 6
        signflip = studentized_paired_t_permutation_pvalue(path, n_perm=64)
        self.assertEqual(signflip['p_value'], 1 / 64)
        self.assertGreater(0.99**6, 0.94)
        self.assertGreater(betting_martingale_pvalue(path), 0.05)
        self.assertFalse(signflip['arbitrary_mean_zero_null_valid'])

    def test_signflip_degenerate_cases_and_exact_denominator(self):
        self.assertEqual(studentized_paired_t_permutation_pvalue([0.02] * 4, n_perm=16)['p_value'], 1 / 16)
        self.assertEqual(studentized_paired_t_permutation_pvalue([-0.02] * 4, n_perm=16)['p_value'], 1.0)
        self.assertEqual(studentized_paired_t_permutation_pvalue([0.0] * 4, n_perm=16)['p_value'], 1.0)
        mixed = studentized_paired_t_permutation_pvalue([1.0, 1.0, 2.0], n_perm=8)
        self.assertEqual(mixed['p_value'], 1 / 8)
        self.assertEqual(mixed['sign_patterns_evaluated'], 8)
        self.assertEqual(mixed['method'], 'EXHAUSTIVE_SIGN_FLIPS')
        self.assertEqual(studentized_paired_t_permutation_pvalue([2.0], n_perm=2)['p_value'], 0.5)

    def test_signflip_monte_carlo_plus_one_and_validation(self):
        # Fixed bit stream tests the MC denominator without any simulated law.
        with patch('stats.paired_inference.random.Random') as rng_type:
            rng_type.return_value.getrandbits.return_value = 0
            result = studentized_paired_t_permutation_pvalue([1.0] * 4, n_perm=3)
        self.assertEqual(result['p_value'], 1 / 4)
        self.assertEqual(result['method'], 'MONTE_CARLO_SIGN_FLIPS_PLUS_ONE')
        for values in ([], [math.nan], [math.inf]):
            with self.assertRaises(ValueError):
                studentized_paired_t_permutation_pvalue(values)
        for draws in (0, -1, 2.5, True):
            with self.assertRaises(ValueError):
                studentized_paired_t_permutation_pvalue([1.0], n_perm=draws)

    def test_gate_rejects_misalignment_invalid_times_and_unbounded_losses(self):
        cases = (
            ([], [], []), ([100.0, 100.0], [94.0], [96.5, 96.5]),
            ([0.0], [1.0], [100.0]), ([100.0], [-1.0], [100.0]),
            ([100.0], [math.nan], [100.0]), ([100.0], [94.0], [math.inf]),
            ([100.0], [300.0], [100.0]), ([100.0], [94.0], [0.0]),
        )
        for args in cases:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    evaluate_line_a_g0_gate(*args)

    def test_gate_preserves_5pct_target_and_distinguishes_admission(self):
        refs, nonrl = [100.0] * 160, [96.5] * 160
        passed = evaluate_line_a_g0_gate(refs, [94.0] * 160, nonrl)
        failed = evaluate_line_a_g0_gate(refs, [96.0] * 160, nonrl)
        self.assertTrue(passed['vs_author_anchor']['meets_registered_5pct_target'])
        self.assertTrue(passed['vs_strongest_non_rl']['beats_strongest_non_rl'])
        self.assertFalse(failed['vs_author_anchor']['meets_registered_5pct_target'])
        self.assertFalse(passed['registration_verified'])
        self.assertFalse(passed['both_budgets_verified'])
        self.assertFalse(passed['bounded_population_support_verified'])
        self.assertEqual(passed['research_admission'], 'NOT_ASSESSED')
        small = evaluate_line_a_g0_gate([100.0] * 4, [94.0] * 4, [96.5] * 4)
        self.assertFalse(small['vs_author_anchor']['meets_registered_5pct_target'])
        precise = evaluate_line_a_g0_gate(refs, [100.0 * (1.0 - 0.0499999)] * 160, nonrl)
        self.assertLess(precise['vs_author_anchor']['mean_rel_improvement'], 0.05)
        self.assertFalse(precise['vs_author_anchor']['meets_registered_5pct_target'])

    def test_default_audit_is_analytic_and_does_not_register_300_maps(self):
        with patch('stats.paired_inference.random.Random', side_effect=AssertionError('unexpected simulation')):
            result = run_statistical_power_and_calibration_audit()
        self.assertEqual(result['monte_carlo']['status'], 'NOT_RUN')
        self.assertIsNone(result['pre_registered_active_base_maps_N'])
        self.assertFalse(result['registration_verified'])
        self.assertEqual(result['research_admission'], 'NOT_ASSESSED')
        diagnosis = result['bounded_range_power_diagnostic']
        self.assertTrue(diagnosis['bounded_min_N_is_necessary_not_sufficient'])
        self.assertTrue(diagnosis['ideal_constant_sample_is_not_a_power_estimate'])
        self.assertEqual(diagnosis['eb_closed_form_min_N_for_plus2pp_holm00125'], 1186)


if __name__ == '__main__':
    unittest.main()
