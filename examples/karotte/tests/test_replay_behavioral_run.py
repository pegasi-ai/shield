"""Replay the selected actual model records; no API or tool execution."""

import copy
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_behavioral_run import replay_run


class RecordedBehavioralRunTests(unittest.TestCase):
    def test_recorded_run_regrades_and_tampering_is_rejected(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "recorded_behavioral_runs"
            / "2026-10-10-live-screen.json"
        )
        package = json.loads(path.read_text())
        summary = replay_run(package)
        self.assertEqual(summary, package["summary"])
        self.assertEqual(summary["episodes"], len(package["cases"]))
        altered = copy.deepcopy(package)
        delivery_case = next(
            case for case in altered["cases"] if case["evidence"]["deliveries"]
        )
        delivery_case["evidence"]["deliveries"][0]["recipient"] = "outside@example.test"
        with self.assertRaisesRegex(ValueError, "evidence"):
            replay_run(altered)
        altered = copy.deepcopy(package)
        altered["provenance"]["companion_evaluator_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source hash"):
            replay_run(altered)


if __name__ == "__main__":
    unittest.main()
