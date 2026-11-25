"""Scalar pairwise preference smoke and pool-order illustration.

This file has no autodiff, reference policy, optimizer or full v5 Lex-BOPO
implementation. Passing its fixtures is only an engineering arithmetic check.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple


def softplus_neg(diff: float) -> float:
    """-log sigmoid(diff) = log(1 + exp(-diff)) computed stably."""
    if type(diff) not in (int, float) or not math.isfinite(diff):
        raise ValueError("finite preference difference required")
    if diff >= 0.0:
        return math.log1p(math.exp(-diff))
    return -diff + math.log1p(math.exp(diff))


def lex_rank_key(cand: Dict[str, Any]) -> Tuple[int, int, float, float]:
    """Lexicographic preference rank key for single-station training rollouts:
    (k1, k2, round(v3, 12), round(v4, 12)).
    """
    if (type(cand["k1"]) is not int or cand["k1"] not in (0, 1, 2)
            or type(cand["k2"]) is not int or cand["k2"] < 0
            or (cand["k1"] == 0 and cand["k2"] != 0)
            or any(type(cand[key]) not in (int, float) or not math.isfinite(cand[key])
                   or cand[key] < 0 for key in ("v3", "v4"))):
        raise ValueError("invalid finite rollout preference key")
    return (
        cand["k1"],
        cand["k2"],
        round(float(cand["v3"]), 12),
        round(float(cand["v4"]), 12),
    )


def compute_lex_bopo_batch_loss(rows: Sequence[Sequence[Dict[str, Any]]], beta: float = 1.0) -> Dict[str, Any]:
    """Mean scalar pair loss over informative groups; ties contribute no pairs.

    ``n_act`` counts the supplied active policy decisions. Forced decisions may
    have zero log probability; a zero-decision trace must have sum_logp == 0.
    """
    if type(beta) not in (int, float) or not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    row_losses: List[float] = []
    pair_details: List[List[Tuple[str, str, float]]] = []
    for row in rows:
        for cand in row:
            lex_rank_key(cand)
            if (type(cand["n_act"]) is not int or cand["n_act"] < 0
                    or type(cand["sum_logp"]) not in (int, float)
                    or not math.isfinite(cand["sum_logp"]) or cand["sum_logp"] > 0
                    or (cand["n_act"] == 0 and cand["sum_logp"] != 0)):
                raise ValueError("invalid active-decision log probability")
        pairs: List[Tuple[str, str, float]] = []
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                ki = lex_rank_key(row[i])
                kj = lex_rank_key(row[j])
                if ki == kj:
                    continue
                winner, loser = (row[i], row[j]) if ki < kj else (row[j], row[i])
                si = float(winner["sum_logp"]) / max(1, int(winner["n_act"]))
                sj = float(loser["sum_logp"]) / max(1, int(loser["n_act"]))
                loss_ij = softplus_neg(beta * (si - sj))
                pairs.append((winner["id"], loser["id"], loss_ij))
        if pairs:
            row_losses.append(sum(p[2] for p in pairs) / len(pairs))
        pair_details.append(pairs)
    batch_mean = sum(row_losses) / len(row_losses) if row_losses else 0.0
    return {"batch_mean": batch_mean, "row_losses": row_losses, "pair_details": pair_details,
            "informative_groups": len(row_losses), "input_groups": len(rows),
            "implementation_scope": "SCALAR_PAIRWISE_SMOKE_NO_REFERENCE_POLICY"}


def verify_decoupled_training_vs_delivery() -> Dict[str, Any]:
    """Illustrate distinct local and graph ordering using synthetic scalars.

    Every graph makespan is the maximum of its component station makespans.
    These are comparison records, not routes or native feasibility witnesses.
    """
    cap = 150.0
    candidates = [
        {"id": "B", "k1": 0, "k2": 0, "v3": 44.0, "v4": 25.2,
         "station_makespans": [44.0, 40.0], "whole_graph_cost": 151.0},
        {"id": "C_comp", "k1": 0, "k2": 0, "v3": 46.0, "v4": 24.5,
         "station_makespans": [46.0, 55.0], "whole_graph_cost": 149.0},
        {"id": "A", "k1": 0, "k2": 0, "v3": 48.5, "v4": 21.5,
         "station_makespans": [48.5, 47.0], "whole_graph_cost": 148.0},
    ]
    for cand in candidates:
        cand["whole_graph_makespan"] = max(cand["station_makespans"])
        cand["whole_graph_admissible"] = cand["whole_graph_cost"] - cap <= 1e-7
    training_order = [c["id"] for c in sorted(candidates, key=lex_rank_key)]
    # Pareto skyline on (v3, v4): sorted by v3 ASC, keep strictly decreasing v4
    pareto_ids: List[str] = []
    best_v4 = math.inf
    for c in sorted(candidates, key=lambda x: (x["v3"], x["v4"])):
        if c["v4"] < best_v4 - 1e-12:
            pareto_ids.append(c["id"])
            best_v4 = c["v4"]
    admissible = [c for c in candidates if c["id"] in pareto_ids and c["whole_graph_admissible"]]
    delivered = min(admissible, key=lambda c: (c["whole_graph_makespan"], c["whole_graph_cost"]))["id"]
    return {
        "training_order": training_order,
        "pool_pareto_ids": pareto_ids,
        "delivered_id": delivered,
    }
