"""Replay synthetic engineering fixtures. Never grant a research or host-attestation gate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parent
SUMMARY = "v5_1_acceptance_receipt.json"
PENDING = "UNVERIFIED_PENDING_HN01_RECEIPT"


def dump(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf8")
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory():
    files = [ROOT / x for x in ("run_acceptance_pack.py", "README.md", "SPEC_V5_1_ACCEPTANCE.md",
                                 "push_to_github.sh", "frozen_source_lock.json")]
    for name in ("track_a", "track_b", "stats", "tests"):
        files.extend(p for p in (ROOT / name).rglob("*") if p.is_file() and p.suffix in {".py", ".json", ".md"})
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


def verify_source_lock(path):
    lock = json.loads((ROOT / "frozen_source_lock.json").read_text())
    if not lock or any(not (path / k).is_file() or digest(path / k) != v for k, v in lock.items()):
        raise ValueError("frozen scorer sources missing or inconsistent with frozen_source_lock.json")
    return lock


def new_output(requested):
    if requested is None:
        base = ROOT / "runs"
        base.mkdir(exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="engineering-", dir=base))
    out = requested.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hn01-repo-root", "--frozen-root", dest="frozen_root", type=Path,
                    help="parent of production/frozen; binding source code does not identify the host")
    ap.add_argument("--output-dir", type=Path, help="new directory; never overwrites a prior run")
    args = ap.parse_args()
    try:
        out = new_output(args.output_dir)
    except FileExistsError:
        ap.error("output directory already exists; choose a fresh path")
    start = time.monotonic()
    receipt = {"schema": "engineering-acceptance/v5.1-reviewed-1", "status": "RUNNING",
               "scope": "SYNTHETIC_ENGINEERING_ONLY", "hn01_g1_status": PENDING,
               "training_authorized": False, "formal_authorized": False,
               "started_utc": datetime.now(timezone.utc).isoformat(), "platform": platform.system(),
               "python_version": platform.python_version(), "source_sha256": inventory(), "steps": []}
    dump(out / SUMMARY, receipt)
    env = os.environ.copy()
    for key in ("HN01_REPO_ROOT", "TRACK_A_TEST_FROZEN_ROOT", "PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONUTF8="1")

    def step(name, command):
        t = time.monotonic()
        row = {"step": name, "argv": ["python3", *list(map(str, command))]}
        receipt["steps"].append(row)
        with (out / (name + ".stdout")).open("wb") as stdout, (out / (name + ".stderr")).open("wb") as stderr:
            try:
                p = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                   timeout=240, check=False)
                row["exit_code"] = p.returncode
            except subprocess.TimeoutExpired:
                row.update(exit_code=124, timeout_seconds=240)
        row["elapsed_seconds"] = round(time.monotonic() - t, 4)
        if row["exit_code"]:
            raise ValueError(f"{name}: exit {row['exit_code']} (see {name}.stderr)")
    try:
        frozen = args.frozen_root.expanduser().resolve() if args.frozen_root else None
        receipt["frozen_source_sha256"] = verify_source_lock(frozen) if frozen else {}
        if frozen:
            env["TRACK_A_TEST_FROZEN_ROOT"] = str(frozen)
        a_path = out / "track_a_actual_output.json"
        step("track_a", ["track_a/run_track_a_acceptance.py", "--output", a_path] +
             (["--hn01-repo-root", frozen] if frozen else []))
        a = json.loads(a_path.read_text())
        receipt["adapter_mode"] = a["adapter_mode"]
        if frozen:
            evidence = a["binding_evidence"]
            loaded = {"production/frozen/" + k: v for k, v in evidence["source_sha256"].items()}
            if (a["adapter_mode"] != "DIRECT_FROZEN_IMPORT" or evidence["frozen_validation_calls"] < 1
                    or len(loaded) != 3 or any(receipt["frozen_source_sha256"].get(k) != v for k, v in loaded.items())):
                raise ValueError("frozen validator invocation or executed source digests not witnessed")
            receipt["frozen_validation_calls"] = evidence["frozen_validation_calls"]
        logs = out / "track_b_actual_logs.json"
        cmp_path = out / "track_b_comparison_report.json"
        step("track_b_blind", ["track_b/run_track_b_acceptance.py", "--blind-only", "--actual-logs-out", logs,
                              "--b3-trace-out", out / "track_b_b3_full_trace.json",
                              "--timing-out", out / "track_b_completer_timing.json"])
        step("track_b_compare", ["track_b/independent_comparator.py", "--actual-logs", logs, "--expected",
                                 "track_b/expected/track_b_v5_1_expected.json", "--report-out", cmp_path])
        b = json.loads(cmp_path.read_text())
        if b.get("all_passed") is not True or (b.get("trace_count"), b.get("total_verified_checkpoints")) != (7, 37):
            raise ValueError("expected all seven registered fixtures and 37 checkpoints")
        from stats.paired_inference import run_statistical_power_and_calibration_audit
        dump(out / "stats_power_audit.json", run_statistical_power_and_calibration_audit())
        step("unit_tests", ["-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py", "-v"])
        if inventory() != receipt["source_sha256"]:
            raise ValueError("source, expected or tests changed during acceptance")
        receipt["key_verdicts"] = {
            "fixture_costs_rmb": [a["dabs_6tower_reference"]["raw_cost_rmb"],
                                  *[a["dabs_6tower_candidates"][k]["raw_cost_rmb"] for k in
                                    ("COSORTIE_FASTER_ROUTE", "TIE_MAKESPAN_COSTLIER_ROUTE")]],
            "fixture_count": b["trace_count"], "matched_checkpoints": b["total_verified_checkpoints"]}
        receipt["status"] = "ENGINEERING_PASS"
    except Exception as e:
        receipt.update(status="FAILED", error=f"{type(e).__name__}: {e}")
    receipt["completed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt["elapsed_seconds"] = round(time.monotonic() - start, 4)
    receipt["output_sha256"] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name != SUMMARY}
    dump(out / SUMMARY, receipt)
    for line in (receipt["status"], receipt.get("error"), str(out / SUMMARY)):
        if line: print(line)
    return int(receipt["status"] != "ENGINEERING_PASS")


if __name__ == "__main__":
    raise SystemExit(main())
