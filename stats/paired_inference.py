"""Bounded paired-mean diagnostics; engineering checks do not grant research admission.

Each observation must represent one independent base map in a frozen order.
Seeds, winds, stations and repeated runs are not additional independent maps.
Betting validity requires E[X_i | past] <= the tested mean under the null;
independent identically distributed bounded maps suffice. Dependence or sorting
maps by observed improvement do not generally satisfy that contract.

Historical function names are retained. They do not verify registration or
admit G0/G3, training, new maps or Formal. The default audit is analytic only.
"""
from __future__ import annotations

import itertools
import math
import random
from typing import Any, Dict, Sequence


def _finite(value: float, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _alpha(alpha: float) -> float:
    value = _finite(alpha, "alpha")
    if not 0.0 < value < 1.0:
        raise ValueError("alpha must lie strictly between 0 and 1")
    return value


def _samples(samples: Sequence[float], a: float, b: float, min_n: int = 1) -> list[float]:
    values = [_finite(x, "sample") for x in samples]
    if len(values) < min_n:
        raise ValueError(f"at least {min_n} observation(s) required")
    if any(x < a or x > b for x in values):
        raise ValueError(f"samples must lie in [{a}, {b}]; clipping is not permitted")
    return values


def empirical_bernstein_lower_bound(
    samples: Sequence[float], a: float = -1.0, b: float = 1.0, alpha: float = 0.05
) -> float:
    """Maurer--Pontil closed-form lower bound for independent bounded observations."""
    a, b, alpha = _finite(a, "a"), _finite(b, "b"), _alpha(alpha)
    if not a < b or not math.isfinite(b - a):
        raise ValueError("bounds must have positive finite width")
    values = _samples(samples, a, b, min_n=2)
    n = len(values)
    mean = math.fsum(values) / n
    var = math.fsum((x - mean) ** 2 for x in values) / (n - 1)
    log_term = math.log(2.0) - math.log(alpha)
    radius = math.sqrt(2.0 * var * log_term / n) + 7.0 * (b - a) * log_term / (3.0 * (n - 1))
    return max(a, mean - radius)


def min_n_for_eb_zero_variance(delta: float = 0.02, rng: float = 2.0, alpha: float = 0.0125) -> int:
    """Smallest N for positive constant-sample EB bound, not a power guarantee."""
    delta, rng, alpha = _finite(delta, "delta"), _finite(rng, "rng"), _alpha(alpha)
    if delta <= 0.0 or rng <= 0.0:
        raise ValueError("delta and rng must be positive")
    threshold = 7.0 * rng * (math.log(2.0) - math.log(alpha)) / (3.0 * delta)
    if not math.isfinite(threshold):
        raise ValueError("requested sample-size threshold exceeds numeric range")
    return math.floor(threshold) + 2  # N - 1 must exceed threshold strictly.


def min_n_for_gaffke_zero_variance(delta: float = 0.02, alpha: float = 0.0125) -> int:
    """Necessary sample size for the constant +delta path on [-1, 1].

    A mean-zero law puts mass 1/(1+delta) at +delta and the rest at -1.
    This path has probability (1+delta)^(-N). The condition is neither a
    sufficient sample size nor the power of the implemented betting test.
    """
    delta, alpha = _finite(delta, "delta"), _alpha(alpha)
    if not 0.0 < delta <= 1.0:
        raise ValueError("delta must lie in (0, 1] for the [-1, 1] construction")
    threshold = -math.log(alpha) / math.log1p(delta)
    if not math.isfinite(threshold):
        raise ValueError("requested sample-size threshold exceeds numeric range")
    return max(1, math.ceil(threshold))


def _predictable_bets(values: Sequence[float], c_clip: float) -> list[float]:
    """Bets depend on original past observations, never on candidate null means."""
    bets = []
    total = total_sq = 0.0
    for i, x in enumerate(values):
        if i == 0:
            bet = min(0.5, c_clip)
        else:
            mean = total / i
            var = max(1e-6, total_sq / i - mean * mean)
            bet = max(0.0, min(c_clip, mean / (var + mean * mean + 1e-4 / (i + 1))))
        bets.append(bet)
        total += x
        total_sq += x * x
    return bets


def _betting_pvalue_at_mean(values: Sequence[float], bets: Sequence[float], mean: float) -> float:
    # Each positive factor decreases in mean on (-1, 1]. At a conditional-mean
    # null its conditional expectation is <= 1, giving a valid wealth process.
    log_wealth = math.fsum(
        math.log1p(bet * ((x - mean) / (1.0 + mean)))
        for x, bet in zip(values, bets)
    )
    return 1.0 if log_wealth <= 0.0 else math.exp(-log_wealth)


def betting_martingale_pvalue(samples: Sequence[float], c_clip: float = 0.98) -> float:
    """Finite-sample p-value for E[X_i | past] <= 0 with X_i in [-1, 1].

    The predictable nonnegative wealth has expectation <= 1 under that null;
    Markov's inequality gives P(p <= alpha) <= alpha. This is not a universal
    power claim or a verification of the sampling assumptions.
    """
    values = _samples(samples, -1.0, 1.0)
    c_clip = _finite(c_clip, "c_clip")
    if not 0.0 <= c_clip < 1.0:
        raise ValueError("c_clip must lie in [0, 1)")
    return _betting_pvalue_at_mean(values, _predictable_bets(values, c_clip), 0.0)


def betting_martingale_lower_bound(
    samples: Sequence[float], alpha: float = 0.05, tol: float = 1e-6
) -> float:
    """Invert a monotone valid mean-test family for a (1-alpha) lower bound.

    The same predictable bets apply at every m; factors
    1 + lambda_i*(X_i-m)/(1+m) decrease in m. Refitting bets separately at m
    need not be monotone. Return the rejected endpoint, rounding downward.
    """
    values, alpha, tol = _samples(samples, -1.0, 1.0), _alpha(alpha), _finite(tol, "tol")
    if not 0.0 < tol < 1.0:
        raise ValueError("tol must lie in (0, 1)")
    bets = _predictable_bets(values, 0.98)
    lo, hi = -1.0, 1.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if mid == lo or mid == hi:
            break
        if _betting_pvalue_at_mean(values, bets, mid) <= alpha:
            lo = mid
        else:
            hi = mid
    return lo


def studentized_paired_t_permutation_pvalue(
    diffs: Sequence[float], n_perm: int = 2000, seed: int = 20261125
) -> Dict[str, Any]:
    """One-sided sign flips: finite-sample validity requires sign symmetry.

    For arbitrary asymmetric mean-zero laws this is only an asymptotic
    diagnostic. Signed sums order Student t statistics identically, including
    infinite t at zero variance; no unrelated mean-test p-value replaces it.
    """
    values = [_finite(x, "difference") for x in diffs]
    if not values:
        raise ValueError("at least one paired difference required")
    if isinstance(n_perm, bool) or not isinstance(n_perm, int) or n_perm < 1:
        raise ValueError("n_perm must be a positive integer")
    n = len(values)
    # Scaling preserves the statistic while avoiding overflow on large inputs.
    scale = max(abs(x) for x in values) or 1.0
    normalized = [x / scale for x in values]
    observed = math.fsum(normalized)
    mean = observed / n
    variance = math.fsum((x - mean) ** 2 for x in normalized) / (n - 1) if n > 1 else 0.0
    sd = math.sqrt(variance)
    skew = math.fsum((x - mean) ** 3 for x in normalized) / n / sd**3 if sd > 0.0 else 0.0
    exhaustive = n <= n_perm.bit_length() - 1
    if exhaustive:
        sums = (
            math.fsum(sign * x for sign, x in zip(signs, normalized))
            for signs in itertools.product((-1.0, 1.0), repeat=n)
        )
        count = sum(value >= observed - 1e-12 for value in sums)
        draws = 1 << n
        pvalue = count / draws
    else:
        rng = random.Random(seed)
        count = 1
        for _ in range(n_perm):
            value = math.fsum(x if rng.getrandbits(1) else -x for x in normalized)
            count += value >= observed - 1e-12
        draws = n_perm
        pvalue = count / (n_perm + 1)
    return {
        "p_value": pvalue, "sd_obs": sd * scale, "skewness": skew,
        "method": "EXHAUSTIVE_SIGN_FLIPS" if exhaustive else "MONTE_CARLO_SIGN_FLIPS_PLUS_ONE",
        "sign_patterns_evaluated": draws,
        "finite_sample_null": "independent sign symmetry / within-pair exchangeability",
        "arbitrary_mean_zero_null_valid": False,
    }


def evaluate_line_a_g0_gate(
    base_map_ref_makespans: Sequence[float],
    base_map_cand_makespans: Sequence[float],
    base_map_strong_non_rl_makespans: Sequence[float],
) -> Dict[str, Any]:
    """Numerical component of one Line A comparison; not complete G0 admission.

    Check >=5% anchor mean and positive strongest-non-RL mean, with positive
    one-sided 95% LCBs. Callers separately verify both budgets, map clustering,
    frozen order, sampling, cost, controls, registration and all other gates.
    In particular, observed in-range values do not prove a population bound:
    the registered relative loss is unbounded below unless the delivery contract
    supplies a genuine bound. These p-values/LCBs are conditional on that bound.
    Requiring both comparisons is an intersection-union claim. Separate
    superiority claims or candidate selection need registered multiplicity rules.
    """
    ref = [_finite(x, "reference makespan") for x in base_map_ref_makespans]
    cand = [_finite(x, "candidate makespan") for x in base_map_cand_makespans]
    non_rl = [_finite(x, "non-RL makespan") for x in base_map_strong_non_rl_makespans]
    if not ref or not len(ref) == len(cand) == len(non_rl):
        raise ValueError("paired map-level inputs must have identical nonzero lengths")
    if any(x <= 0.0 for x in ref + non_rl) or any(x < 0.0 for x in cand):
        raise ValueError("references must be positive; candidate makespans must be nonnegative")
    rel_anchor = [(r - c) / r for r, c in zip(ref, cand)]
    rel_non_rl = [(r - c) / r for r, c in zip(non_rl, cand)]
    # Relative losses are not inherently >= -1. Clipping changes the estimand.
    _samples(rel_anchor, -1.0, 1.0)
    _samples(rel_non_rl, -1.0, 1.0)

    def comparison(values: list[float], target: float, flag: str) -> Dict[str, Any]:
        mean = math.fsum(values) / len(values)
        lcb = betting_martingale_lower_bound(values)
        pvalue = betting_martingale_pvalue(values)
        effect_met = mean >= target if target > 0.0 else mean > 0.0
        return {
            "mean_rel_improvement": mean, "exact_betting_95_lcb": lcb,
            "exact_betting_p_value": pvalue,
            flag: effect_met and lcb > 0.0 and pvalue <= 0.05,
            "interpretation": "CONDITIONAL_NUMERICAL_RESULT; DOES_NOT_VERIFY_REGISTERED_GATE",
        }

    return {
        "n_independent_base_maps": len(ref),
        "vs_author_anchor": comparison(rel_anchor, 0.05, "meets_registered_5pct_target"),
        "vs_strongest_non_rl": comparison(rel_non_rl, 0.0, "beats_strongest_non_rl"),
        "scope": "NUMERICAL_COMPONENT_ONLY",
        "input_unit": "one independent preaggregated base map; identities/order verified by caller",
        "registration_verified": False, "both_budgets_verified": False,
        "bounded_population_support_verified": False,
        "inference_status": "CONDITIONAL_ON_DECLARED_POPULATION_SUPPORT_AND_SAMPLING",
        "research_admission": "NOT_ASSESSED",
    }


def run_statistical_power_and_calibration_audit(
    *, include_simulation: bool = False, trials: int = 1000, seed: int = 20261125
) -> Dict[str, Any]:
    """Default closed forms only; explicit optional Monte Carlo is descriptive.

    Simulations require separate project authorization and execution on hn01.
    No simulated estimate is an exact type-I rate or universal power guarantee.
    """
    if not isinstance(include_simulation, bool):
        raise ValueError("include_simulation must be a boolean")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    result: Dict[str, Any] = {
        "scope": "ENGINEERING_STATISTICAL_DIAGNOSTIC_ONLY",
        "research_admission": "NOT_ASSESSED", "registration_verified": False,
        "bounded_range_power_diagnostic": {
            "eb_closed_form_lb_N60_plus2pp_zero_var_alpha005": empirical_bernstein_lower_bound([0.02] * 60),
            "eb_closed_form_lb_N60_plus2pp_zero_var_holm00125": empirical_bernstein_lower_bound([0.02] * 60, alpha=0.0125),
            "eb_closed_form_min_N_for_plus2pp_holm00125": min_n_for_eb_zero_variance(),
            "exact_bounded_min_N_for_plus2pp_alpha005": min_n_for_gaffke_zero_variance(alpha=0.05),
            "exact_bounded_min_N_for_plus2pp_holm00125": min_n_for_gaffke_zero_variance(),
            "bounded_min_N_is_necessary_not_sufficient": True,
            "ideal_constant_sample_is_not_a_power_estimate": True,
        },
        "monte_carlo": {"status": "NOT_RUN", "reason": "explicit opt-in required; default is analytic only"},
        "proposed_active_base_maps_N": 300, "pre_registered_active_base_maps_N": None,
        "seed_deployment_rule": (
            "Training seeds are not independent base maps. Report all frozen seed models; "
            "a deployed ensemble must fit the common end-to-end wall-clock budget. "
            "Median-seed selection needs separately frozen selection rules and data."
        ),
    }
    if not include_simulation:
        return result
    rng = random.Random(seed)
    counts = {"rare_catastrophe_all_positive_heuristic": 0, "rare_catastrophe_betting": 0, "skewed_betting": 0}
    for _ in range(trials):
        catastrophic = [0.01 if rng.random() < 0.99 else -0.99 for _ in range(60)]
        counts["rare_catastrophe_all_positive_heuristic"] += all(x > 0.0 for x in catastrophic)
        counts["rare_catastrophe_betting"] += betting_martingale_pvalue(catastrophic) <= 0.05
        skewed = [0.02 if rng.random() < 0.8 else -0.08 for _ in range(300)]
        counts["skewed_betting"] += betting_martingale_pvalue(skewed) <= 0.05

    def estimate(count: int) -> Dict[str, Any]:
        rate = count / trials
        # Hoeffding intervals retain uncertainty even at 0/trials and trials/trials.
        radius = math.sqrt(math.log(40.0) / (2.0 * trials))
        return {"reject_count": count, "trial_count": trials, "rate": rate,
                "pointwise_95_mc_interval": [max(0.0, rate - radius), min(1.0, rate + radius)]}

    power: Dict[str, Any] = {}
    for n_maps in (60, 160, 230, 300):
        at_005 = at_00125 = 0
        for _ in range(trials):
            # Clipping here defines this synthetic law only. Its moments differ
            # from the raw normal parameters. Never clip observed research outcomes.
            values = [max(-1.0, min(1.0, 0.02 + rng.gauss(0.0, 0.04))) for _ in range(n_maps)]
            pvalue = betting_martingale_pvalue(values)
            at_005 += pvalue <= 0.05
            at_00125 += pvalue <= 0.0125
        power[f"N={n_maps}"] = {"alpha_0_05": estimate(at_005), "alpha_0_0125": estimate(at_00125)}
    result["monte_carlo"] = {
        "status": "SIMULATED_ESTIMATES_ONLY", "seed": seed, "trials_per_cell": trials,
        "type1_estimates": {name: estimate(count) for name, count in counts.items()},
        "power_estimates_clipped_normal_location002_scale004": power,
        "intervals_are_pointwise_not_simultaneous": True,
        "alpha_0_0125_is_only_a_hypothetical_four_test_first_step": True,
        "analytic_all_positive_rare_catastrophe_probability": 0.99**60,
    }
    return result
