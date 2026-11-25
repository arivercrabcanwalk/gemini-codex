"""Regression checks on result labelling, isolation and frozen-source pinning."""
import tempfile
import unittest
from pathlib import Path

import run_acceptance_pack as runner


class RunnerTests(unittest.TestCase):
    def test_existing_receipt_dir_never_reused(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "test"
            self.assertEqual(runner.new_output(out), out.resolve())
            (out / "sentinel").write_text("keep")
            with self.assertRaises(FileExistsError):
                runner.new_output(out)
            self.assertEqual((out / "sentinel").read_text(), "keep")

    def test_directory_name_does_not_attest_host(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                runner.verify_source_lock(Path(d))
        self.assertEqual(runner.PENDING, "UNVERIFIED_PENDING_HN01_RECEIPT")

    def test_provenance_includes_runner_and_all_registered_fixtures(self):
        files = runner.inventory()
        self.assertIn("run_acceptance_pack.py", files)
        self.assertIn("track_a/expected/track_a_v5_1_expected.json", files)
        self.assertIn("track_b/inputs/track_b_v5_1_inputs.json", files)
        self.assertIn("frozen_source_lock.json", files)
