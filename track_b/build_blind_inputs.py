"""Deterministic builder for Track B v5.1 PURE BLIND INPUTS (`track_b_v5_1_inputs.json`).

Contains ZERO expected checkpoints or expected verdicts.
Explicitly separates `tower_catalog` from `mandatory_routine_tasks` and sets `slot_idx`
(0=LD, 1=SD; SABS uses slot_idx=1 only) and `snapshot_epoch` strictly per the frozen
8-row mode transition table.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List


def build_track_b_blind_inputs() -> Dict[str, Any]:
    base_stations = [
        {"station_id": "S1", "station_type": "DABS", "x_m": 0, "y_m": 0},
        {"station_id": "S2", "station_type": "SABS", "x_m": 12000, "y_m": 0},
    ]
    base_catalog = [
        {"node_id": "r1", "x_m": 3600, "y_m": 2700, "default_routine_srv_ms": 480000},
        {"node_id": "r2", "x_m": 3600, "y_m": -2700, "default_routine_srv_ms": 420000},
        {"node_id": "r3", "x_m": -2700, "y_m": 3600, "default_routine_srv_ms": 300000},
        {"node_id": "r4", "x_m": -2700, "y_m": -3600, "default_routine_srv_ms": 300000},
        {"node_id": "r5", "x_m": 0, "y_m": 4500, "default_routine_srv_ms": 300000},
        {"node_id": "r6", "x_m": 0, "y_m": -4500, "default_routine_srv_ms": 300000},
        {"node_id": "r7", "x_m": 12000, "y_m": 2700, "default_routine_srv_ms": 150000},
        {"node_id": "r8", "x_m": 15600, "y_m": 0, "default_routine_srv_ms": 150000},
        {"node_id": "r9", "x_m": 12000, "y_m": -3600, "default_routine_srv_ms": 600000},
    ]
    base_mandatory = [
        {"task_id": "M_S1_r1", "station_id": "S1", "node_id": "r1", "srv_ms": 480000},
        {"task_id": "M_S1_r2", "station_id": "S1", "node_id": "r2", "srv_ms": 420000},
        {"task_id": "M_S1_r3", "station_id": "S1", "node_id": "r3", "srv_ms": 300000},
        {"task_id": "M_S1_r4", "station_id": "S1", "node_id": "r4", "srv_ms": 300000},
        {"task_id": "M_S1_r5", "station_id": "S1", "node_id": "r5", "srv_ms": 300000},
        {"task_id": "M_S1_r6", "station_id": "S1", "node_id": "r6", "srv_ms": 300000},
        {"task_id": "M_S2_r7", "station_id": "S2", "node_id": "r7", "srv_ms": 150000},
        {"task_id": "M_S2_r8", "station_id": "S2", "node_id": "r8", "srv_ms": 150000},
        {"task_id": "M_S2_r9", "station_id": "S2", "node_id": "r9", "srv_ms": 600000},
    ]

    def make_b1_b2(is_fast: bool) -> Dict[str, Any]:
        commit_e2_ms = 4919715 if is_fast else 4920200
        return {
            "trace_id": "B1_ORIG_FAST_E2E" if is_fast else "B2_ORIG_SLOW_E2E",
            "initial_state": {
                "t_start_ms": 0,
                "shift_end_ms": 6000000,
                "budget_total_yuan": "108",
                "stations": copy.deepcopy(base_stations),
                "tower_catalog": copy.deepcopy(base_catalog),
                "mandatory_routine_tasks": copy.deepcopy(base_mandatory),
                "drones": [
                    {
                        "drone_id": "D1_LD",
                        "station_id": "S1",
                        "slot_idx": 0,
                        "model": "LD",
                        "initial_epoch": 0,
                        "initial_witness_ver": "w_s1_ld",
                        "initial_witness_sorties": [
                            {
                                "sortie_id": "LD1_1",
                                "t_arm_ms": 0,
                                "stops": [
                                    {"node_id": "r1", "srv_ms": 480000},
                                    {"node_id": "r2", "srv_ms": 420000},
                                    {"node_id": "S1", "srv_ms": 0},
                                ],
                            },
                            {
                                "sortie_id": "LD1_2",
                                "t_arm_ms": 1860000,
                                "stops": [
                                    {"node_id": "r5", "srv_ms": 300000},
                                    {"node_id": "S1", "srv_ms": 0},
                                ],
                            },
                        ],
                    },
                    {
                        "drone_id": "D1_SD",
                        "station_id": "S1",
                        "slot_idx": 1,
                        "model": "SD",
                        "initial_epoch": 0,
                        "initial_witness_ver": "w_s1_sd",
                        "initial_witness_sorties": [
                            {
                                "sortie_id": "SD1_1",
                                "t_arm_ms": 0,
                                "stops": [
                                    {"node_id": "r3", "srv_ms": 300000},
                                    {"node_id": "r4", "srv_ms": 300000},
                                    {"node_id": "S1", "srv_ms": 0},
                                ],
                            },
                            {
                                "sortie_id": "SD1_2",
                                "t_arm_ms": 1680000,
                                "stops": [
                                    {"node_id": "r6", "srv_ms": 300000},
                                    {"node_id": "S1", "srv_ms": 0},
                                ],
                            },
                        ],
                    },
                    {
                        "drone_id": "D2_SD",
                        "station_id": "S2",
                        "slot_idx": 1,
                        "model": "SD",
                        "initial_epoch": 0,
                        "initial_witness_ver": "w_s2_sd",
                        "initial_witness_sorties": [
                            {
                                "sortie_id": "SD2_1",
                                "t_arm_ms": 0,
                                "stops": [
                                    {"node_id": "r7", "srv_ms": 150000},
                                    {"node_id": "r8", "srv_ms": 150000},
                                    {"node_id": "S2", "srv_ms": 0},
                                ],
                            },
                            {
                                "sortie_id": "SD2_2_LAST",
                                "t_arm_ms": 4920000,
                                "stops": [
                                    {"node_id": "r9", "srv_ms": 600000, "hover_timeout_ms": 300000},
                                    {"node_id": "S2", "srv_ms": 0},
                                ],
                            },
                        ],
                    },
                ],
            },
            "exogenous_events": [
                {
                    "t_ms": 3900000,
                    "type": "PUBLISH_TICKET",
                    "ticket": {"ticket_id": "e1", "node_id": "r5", "sla_ms": 4320000, "srv_ms": 300000},
                },
                {
                    "t_ms": 3900012,
                    "type": "PROPOSE_WITNESS_CAS",
                    "proposal_id": "P_E1",
                    "proposal_seq": 1,
                    "station_id": "S1",
                    "drone_id": "D1_LD",
                    "snapshot_epoch": 10,
                    "snapshot_witness_ver": "w_s1_ld",
                    "new_witness_ver": "w_s1_ld_e1",
                    "candidate_sorties": [
                        {
                            "sortie_id": "LD1_E1",
                            "t_arm_ms": 3900012,
                            "stops": [
                                {"node_id": "r5", "srv_ms": 300000, "ticket_id": "e1"},
                                {"node_id": "S1", "srv_ms": 0},
                            ],
                        }
                    ],
                },
                {
                    "t_ms": 4919700,
                    "type": "PUBLISH_TICKET",
                    "ticket": {"ticket_id": "e2", "node_id": "r6", "sla_ms": 5219880, "srv_ms": 300000},
                },
                {
                    "t_ms": commit_e2_ms,
                    "type": "PROPOSE_WITNESS_CAS",
                    "proposal_id": "P_E2",
                    "proposal_seq": 2,
                    "station_id": "S1",
                    "drone_id": "D1_LD",
                    "snapshot_epoch": 14,
                    "snapshot_witness_ver": "w_s1_ld_e1",
                    "new_witness_ver": "w_s1_ld_e2",
                    "candidate_sorties": [
                        {
                            "sortie_id": "LD1_E2",
                            "t_arm_ms": commit_e2_ms,
                            "stops": [
                                {"node_id": "r6", "srv_ms": 300000, "ticket_id": "e2"},
                                {"node_id": "S1", "srv_ms": 0},
                            ],
                        }
                    ],
                },
            ],
        }

    # B3: Dual hover alarms (50.1m hold alarm + 51.9m global budget alarm) with exact snapshot_epoch=4 at 36.0m!
    b3_trace = {
        "trace_id": "B3_ORIG_DUAL_HOVER_ALARMS",
        "initial_state": {
            "t_start_ms": 0,
            "shift_end_ms": 6000000,
            "budget_total_yuan": "171/5",
            "stations": [
                {"station_id": "S1", "station_type": "SABS", "x_m": 0, "y_m": 0},
                {"station_id": "S2", "station_type": "SABS", "x_m": 12000, "y_m": 0},
            ],
            "tower_catalog": [
                {"node_id": "r_a", "x_m": -2160, "y_m": 2880, "default_routine_srv_ms": 300000},
                {"node_id": "r_b", "x_m": 2160, "y_m": 2880, "default_routine_srv_ms": 360000},
                {"node_id": "r_d", "x_m": 12000, "y_m": 3600, "default_routine_srv_ms": 360000},
            ],
            "mandatory_routine_tasks": [
                {"task_id": "M_B3_ra", "station_id": "S1", "node_id": "r_a", "srv_ms": 300000},
                {"task_id": "M_B3_rb", "station_id": "S1", "node_id": "r_b", "srv_ms": 360000},
                {"task_id": "M_B3_rd", "station_id": "S2", "node_id": "r_d", "srv_ms": 360000},
            ],
            "drones": [
                {
                    "drone_id": "D1",
                    "station_id": "S1",
                    "slot_idx": 1,
                    "model": "SD",
                    "initial_epoch": 0,
                    "initial_witness_ver": "w1",
                    "initial_witness_sorties": [
                        {
                            "sortie_id": "D1_1",
                            "t_arm_ms": 0,
                            "stops": [
                                {"node_id": "r_a", "srv_ms": 300000},
                                {"node_id": "S1", "srv_ms": 0},
                            ],
                        },
                        {
                            "sortie_id": "D1_2",
                            "t_arm_ms": 2400000,
                            "stops": [
                                {"node_id": "r_b", "srv_ms": 360000, "hover_timeout_ms": 120000},
                                {"node_id": "S1", "srv_ms": 0},
                            ],
                        },
                    ],
                },
                {
                    "drone_id": "D2",
                    "station_id": "S2",
                    "slot_idx": 1,
                    "model": "SD",
                    "initial_epoch": 0,
                    "initial_witness_ver": "w2",
                    "initial_witness_sorties": [
                        {
                            "sortie_id": "D2_1",
                            "t_arm_ms": 2400000,
                            "stops": [
                                {"node_id": "r_d", "srv_ms": 360000, "hover_timeout_ms": 120000},
                                {"node_id": "S2", "srv_ms": 0},
                            ],
                        }
                    ],
                },
            ],
        },
        "exogenous_events": [
            {
                "t_ms": 2100000,
                "type": "PUBLISH_TICKET",
                "ticket": {"ticket_id": "e_b3", "node_id": "r_a", "sla_ms": 3294000, "srv_ms": 300000},
            },
            {
                "t_ms": 2160000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B3",
                "proposal_seq": 1,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 4,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w1_urg",
                "candidate_sorties": [
                    {
                        "sortie_id": "D1_2_URG",
                        "t_arm_ms": 2400000,
                        "stops": [
                            {"node_id": "r_b", "srv_ms": 360000, "hover_timeout_ms": 120000},
                            {"node_id": "r_a", "srv_ms": 300000, "ticket_id": "e_b3"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
        ],
    }

    # B4: Non-diversion after default takeoff at 4,920,000 ms
    b4_trace = copy.deepcopy(make_b1_b2(False))
    b4_trace["trace_id"] = "B4_ORIG_NO_DIVERSION_AFTER_DEFAULT_TAKEOFF"
    b4_trace["exogenous_events"][2] = {
        "t_ms": 4919700,
        "type": "PUBLISH_TICKET",
        "ticket": {"ticket_id": "e2_s2", "node_id": "r7", "sla_ms": 5400000, "srv_ms": 60000},
    }
    b4_trace["exogenous_events"][3] = {
        "t_ms": 4920200,
        "type": "PROPOSE_WITNESS_CAS",
        "proposal_id": "P_B4_STALE_TAKEOFF",
        "proposal_seq": 2,
        "station_id": "S2",
        "drone_id": "D2_SD",
        "snapshot_epoch": 6,
        "snapshot_witness_ver": "w_s2_sd",
        "new_witness_ver": "w_s2_sd_new",
        "candidate_sorties": [
            {
                "sortie_id": "SD2_NEW",
                "t_arm_ms": 4920200,
                "stops": [
                    {"node_id": "r7", "srv_ms": 60000, "ticket_id": "e2_s2"},
                    {"node_id": "r9", "srv_ms": 600000},
                    {"node_id": "S2", "srv_ms": 0},
                ],
            }
        ],
    }

    # B5: Zero-balance exact exhaust / micro-overdraft / zero-net-cost retiming
    b5_trace = copy.deepcopy(make_b1_b2(True))
    b5_trace["trace_id"] = "B5_ORIG_ZERO_BALANCE_TRIPLE"
    b5_trace["initial_state"]["budget_total_yuan"] = "95"
    b5_trace["exogenous_events"] = [
        {
            "t_ms": 3900000,
            "type": "PUBLISH_TICKET",
            "ticket": {"ticket_id": "e1", "node_id": "r5", "sla_ms": 4320000, "srv_ms": 300000},
        },
        {
            "t_ms": 3900012,
            "type": "PROPOSE_WITNESS_CAS",
            "proposal_id": "P_B5_EXACT",
            "proposal_seq": 1,
            "station_id": "S1",
            "drone_id": "D1_LD",
            "snapshot_epoch": 10,
            "snapshot_witness_ver": "w_s1_ld",
            "new_witness_ver": "w_e1",
            "candidate_sorties": [
                {
                    "sortie_id": "LD1_E1",
                    "t_arm_ms": 3900012,
                    "stops": [
                        {"node_id": "r5", "srv_ms": 300000, "ticket_id": "e1"},
                        {"node_id": "S1", "srv_ms": 0},
                    ],
                }
            ],
        },
        {
            "t_ms": 3900500,
            "type": "PUBLISH_TICKET",
            "ticket": {"ticket_id": "e_tiny", "node_id": "r7", "sla_ms": 5400000, "srv_ms": 60000},
        },
        {
            "t_ms": 3900600,
            "type": "PROPOSE_WITNESS_CAS",
            "proposal_id": "P_B5_OVER",
            "proposal_seq": 2,
            "station_id": "S2",
            "drone_id": "D2_SD",
            "snapshot_epoch": 6,
            "snapshot_witness_ver": "w_s2_sd",
            "new_witness_ver": "w_fail",
            "candidate_sorties": [
                {
                    "sortie_id": "SD2_TINY",
                    "t_arm_ms": 4200000,
                    "stops": [
                        {"node_id": "r7", "srv_ms": 60000, "ticket_id": "e_tiny"},
                        {"node_id": "S2", "srv_ms": 0},
                    ],
                },
                {
                    "sortie_id": "SD2_2_LAST",
                    "t_arm_ms": 4920000,
                    "stops": [
                        {"node_id": "r9", "srv_ms": 600000, "hover_timeout_ms": 300000},
                        {"node_id": "S2", "srv_ms": 0},
                    ],
                },
            ],
        },
        {
            "t_ms": 3900600,
            "type": "PROPOSE_WITNESS_CAS",
            "proposal_id": "P_B5_ZERO_NET",
            "proposal_seq": 3,
            "station_id": "S2",
            "drone_id": "D2_SD",
            "snapshot_epoch": 6,
            "snapshot_witness_ver": "w_s2_sd",
            "new_witness_ver": "w_retime",
            "candidate_sorties": [
                {
                    "sortie_id": "SD2_2_RETIMED",
                    "t_arm_ms": 4890000,
                    "stops": [
                        {"node_id": "r9", "srv_ms": 600000, "hover_timeout_ms": 300000},
                        {"node_id": "S2", "srv_ms": 0},
                    ],
                }
            ],
        },
    ]

    # B6: Single-cause hover controls with explicit separation of tower_catalog (5 towers)
    #     vs mandatory_routine_tasks (only r_h on S1 and r_s2_only on S2!) and snapshot_epoch=3!
    b6_trace = {
        "trace_id": "B6_ORIG_HOVER_SINGLE_CAUSE_CONTROLS",
        "initial_state": {
            "t_start_ms": 0,
            "shift_end_ms": 6000000,
            "budget_total_yuan": "120",
            "stations": [
                {"station_id": "S1", "station_type": "SABS", "x_m": 0, "y_m": 0},
                {"station_id": "S2", "station_type": "SABS", "x_m": 12000, "y_m": 0},
            ],
            "tower_catalog": [
                {"node_id": "r_h", "x_m": 0, "y_m": 3600, "default_routine_srv_ms": 300000},
                {"node_id": "r_near", "x_m": 0, "y_m": 2700, "default_routine_srv_ms": 60000},
                {"node_id": "r_west", "x_m": -2160, "y_m": 2880, "default_routine_srv_ms": 60000},
                {"node_id": "r_east", "x_m": 2160, "y_m": 2880, "default_routine_srv_ms": 60000},
                {"node_id": "r_s2_only", "x_m": 12000, "y_m": 3600, "default_routine_srv_ms": 60000},
            ],
            "mandatory_routine_tasks": [
                {"task_id": "M_B6_rh", "station_id": "S1", "node_id": "r_h", "srv_ms": 300000},
                {"task_id": "M_B6_rs2", "station_id": "S2", "node_id": "r_s2_only", "srv_ms": 60000},
            ],
            "drones": [
                {
                    "drone_id": "D1",
                    "station_id": "S1",
                    "slot_idx": 1,
                    "model": "SD",
                    "initial_epoch": 0,
                    "initial_witness_ver": "w1",
                    "initial_witness_sorties": [
                        {
                            "sortie_id": "D1_BASE",
                            "t_arm_ms": 0,
                            "stops": [
                                {"node_id": "r_h", "srv_ms": 300000, "hover_timeout_ms": 300000},
                                {"node_id": "S1", "srv_ms": 0},
                            ],
                        }
                    ],
                },
                {
                    "drone_id": "D2",
                    "station_id": "S2",
                    "slot_idx": 1,
                    "model": "SD",
                    "initial_epoch": 0,
                    "initial_witness_ver": "w2",
                    "initial_witness_sorties": [
                        {
                            "sortie_id": "D2_BASE",
                            "t_arm_ms": 0,
                            "stops": [
                                {"node_id": "r_s2_only", "srv_ms": 60000},
                                {"node_id": "S2", "srv_ms": 0},
                            ],
                        }
                    ],
                },
            ],
        },
        "exogenous_events": [
            {"t_ms": 600000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_cov", "node_id": "r_s2_only", "sla_ms": 5000000, "srv_ms": 60000}},
            {"t_ms": 600000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_sla", "node_id": "r_near", "sla_ms": 666000, "srv_ms": 300000}},
            {"t_ms": 600000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_batt1", "node_id": "r_west", "sla_ms": 5000000, "srv_ms": 900000}},
            {"t_ms": 600000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_batt2", "node_id": "r_east", "sla_ms": 5800000, "srv_ms": 900000}},
            {"t_ms": 600000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_ok", "node_id": "r_near", "sla_ms": 1200000, "srv_ms": 300000}},
            {
                "t_ms": 605000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B6_TAMPERED_SRV",
                "proposal_seq": 1,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 3,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w_bad1",
                "candidate_sorties": [
                    {
                        "sortie_id": "S_BAD1",
                        "t_arm_ms": 605000,
                        "stops": [
                            {"node_id": "r_near", "srv_ms": 240000, "ticket_id": "e_ok"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {
                "t_ms": 606000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B6_COV_OUT",
                "proposal_seq": 2,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 3,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w_bad2",
                "candidate_sorties": [
                    {
                        "sortie_id": "S_BAD2",
                        "t_arm_ms": 606000,
                        "stops": [
                            {"node_id": "r_s2_only", "srv_ms": 60000, "ticket_id": "e_cov"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {
                "t_ms": 612000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B6_SLA_LATE",
                "proposal_seq": 3,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 3,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w_bad3",
                "candidate_sorties": [
                    {
                        "sortie_id": "S_BAD3",
                        "t_arm_ms": 612000,
                        "stops": [
                            {"node_id": "r_near", "srv_ms": 300000, "ticket_id": "e_sla"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {
                "t_ms": 660000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B6_BATT_OVER",
                "proposal_seq": 4,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 3,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w_bad4",
                "candidate_sorties": [
                    {
                        "sortie_id": "S_BAD4",
                        "t_arm_ms": 660000,
                        "stops": [
                            {"node_id": "r_west", "srv_ms": 900000, "ticket_id": "e_batt1"},
                            {"node_id": "r_east", "srv_ms": 900000, "ticket_id": "e_batt2"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {
                "t_ms": 720000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B6_SHIFT_OVER",
                "proposal_seq": 5,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 3,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w_bad5",
                "candidate_sorties": [
                    {
                        "sortie_id": "S_OK_PART1",
                        "t_arm_ms": 720000,
                        "stops": [
                            {"node_id": "r_west", "srv_ms": 900000, "ticket_id": "e_batt1"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    },
                    {
                        "sortie_id": "S_LATE_PART2",
                        "t_arm_ms": 4800000,
                        "stops": [
                            {"node_id": "r_east", "srv_ms": 900000, "ticket_id": "e_batt2"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    },
                ],
            },
            {
                "t_ms": 750000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_B6_VALID",
                "proposal_seq": 6,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 3,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w_ok",
                "candidate_sorties": [
                    {
                        "sortie_id": "S_VALID",
                        "t_arm_ms": 750000,
                        "stops": [
                            {"node_id": "r_near", "srv_ms": 300000, "ticket_id": "e_ok"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
        ],
    }

    # B7_B8: Same-ms contention + 200 m ceiling quantization (13,334 ms -> return 86,668 ms, epoch=4)
    #        with explicit separation of r_mid (catalog overlap tower) from mandatory_routine_tasks!
    b7_b8_trace = {
        "trace_id": "B7_B8_ORIG_CONTENTION_AND_200M_QUANTIZATION",
        "initial_state": {
            "t_start_ms": 0,
            "shift_end_ms": 6000000,
            "budget_total_yuan": "27",
            "stations": [
                {"station_id": "S1", "station_type": "SABS", "x_m": 0, "y_m": 0},
                {"station_id": "S2", "station_type": "SABS", "x_m": 7200, "y_m": 0},
            ],
            "tower_catalog": [
                {"node_id": "r_mid", "x_m": 3600, "y_m": 0, "default_routine_srv_ms": 60000},
                {"node_id": "r_200m", "x_m": 200, "y_m": 0, "default_routine_srv_ms": 60000},
                {"node_id": "r_s2_local", "x_m": 7200, "y_m": 3600, "default_routine_srv_ms": 60000},
            ],
            "mandatory_routine_tasks": [
                {"task_id": "M_B7_200m", "station_id": "S1", "node_id": "r_200m", "srv_ms": 60000},
                {"task_id": "M_B7_s2loc", "station_id": "S2", "node_id": "r_s2_local", "srv_ms": 60000},
            ],
            "drones": [
                {
                    "drone_id": "D1",
                    "station_id": "S1",
                    "slot_idx": 1,
                    "model": "SD",
                    "initial_epoch": 0,
                    "initial_witness_ver": "w1",
                    "initial_witness_sorties": [
                        {
                            "sortie_id": "D1_INIT_200M",
                            "t_arm_ms": 0,
                            "stops": [
                                {"node_id": "r_200m", "srv_ms": 60000},
                                {"node_id": "S1", "srv_ms": 0},
                            ],
                        }
                    ],
                },
                {
                    "drone_id": "D2",
                    "station_id": "S2",
                    "slot_idx": 1,
                    "model": "SD",
                    "initial_epoch": 0,
                    "initial_witness_ver": "w2",
                    "initial_witness_sorties": [
                        {
                            "sortie_id": "D2_INIT",
                            "t_arm_ms": 0,
                            "stops": [
                                {"node_id": "r_s2_local", "srv_ms": 60000},
                                {"node_id": "S2", "srv_ms": 0},
                            ],
                        }
                    ],
                },
            ],
        },
        "exogenous_events": [
            {"t_ms": 590000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_race", "node_id": "r_mid", "sla_ms": 1200000, "srv_ms": 300000}},
            {
                "t_ms": 600000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_RACE_S2_LOSER",
                "proposal_seq": 1,
                "station_id": "S2",
                "drone_id": "D2",
                "snapshot_epoch": 4,
                "snapshot_witness_ver": "w2",
                "new_witness_ver": "w2_r",
                "candidate_sorties": [
                    {
                        "sortie_id": "S2_R",
                        "t_arm_ms": 600000,
                        "stops": [
                            {"node_id": "r_mid", "srv_ms": 300000, "ticket_id": "e_race"},
                            {"node_id": "S2", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {
                "t_ms": 600000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_RACE_S1_WINNER",
                "proposal_seq": 2,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 4,
                "snapshot_witness_ver": "w1",
                "new_witness_ver": "w1_r",
                "candidate_sorties": [
                    {
                        "sortie_id": "S1_R",
                        "t_arm_ms": 600000,
                        "stops": [
                            {"node_id": "r_mid", "srv_ms": 300000, "ticket_id": "e_race"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {"t_ms": 1790000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_loose", "node_id": "r_mid", "sla_ms": 2600000, "srv_ms": 300000}},
            {"t_ms": 1790000, "type": "PUBLISH_TICKET", "ticket": {"ticket_id": "e_tight", "node_id": "r_s2_local", "sla_ms": 2400000, "srv_ms": 300000}},
            {
                "t_ms": 1800000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_W2_S1_LOOSE_LOSER",
                "proposal_seq": 3,
                "station_id": "S1",
                "drone_id": "D1",
                "snapshot_epoch": 8,
                "snapshot_witness_ver": "w1_r",
                "new_witness_ver": "w1_l",
                "candidate_sorties": [
                    {
                        "sortie_id": "S1_L",
                        "t_arm_ms": 1800000,
                        "stops": [
                            {"node_id": "r_mid", "srv_ms": 300000, "ticket_id": "e_loose"},
                            {"node_id": "S1", "srv_ms": 0},
                        ],
                    }
                ],
            },
            {
                "t_ms": 1800000,
                "type": "PROPOSE_WITNESS_CAS",
                "proposal_id": "P_W2_S2_TIGHT_WINNER",
                "proposal_seq": 4,
                "station_id": "S2",
                "drone_id": "D2",
                "snapshot_epoch": 4,
                "snapshot_witness_ver": "w2",
                "new_witness_ver": "w2_t",
                "candidate_sorties": [
                    {
                        "sortie_id": "S2_T",
                        "t_arm_ms": 1800000,
                        "stops": [
                            {"node_id": "r_s2_local", "srv_ms": 300000, "ticket_id": "e_tight"},
                            {"node_id": "S2", "srv_ms": 0},
                        ],
                    }
                ],
            },
        ],
    }

    traces: List[Dict[str, Any]] = [
        make_b1_b2(True),
        make_b1_b2(False),
        b3_trace,
        b4_trace,
        b5_trace,
        b6_trace,
        b7_b8_trace,
    ]
    return {
        "schema_version": "v5.1-TrackB-PureBlindInputs",
        "note": "Strictly blind inputs: ZERO expected checkpoints or verdicts exist in this file.",
        "traces": traces,
    }


if __name__ == "__main__":
    out_path = Path(__file__).resolve().parent / "inputs" / "track_b_v5_1_inputs.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_track_b_blind_inputs()
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote blind inputs to {out_path}")
