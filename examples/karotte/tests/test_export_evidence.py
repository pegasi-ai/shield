import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_evidence import completed_evidence


class ExportEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {"run_id": "current", "task_id": "reins-report-delivery"}
        self.transcript = {
            "run_id": "current",
            "events": [
                {"type": "task_started", **self.config},
                {
                    "type": "scoring",
                    "scoring": {
                        "score": 1,
                        "continue_task": True,
                        "metadata": {
                            "reins_evidence": {
                                "run_id": "current",
                                "grader": {"score": 1},
                            }
                        },
                    },
                },
                {"type": "task_completed", "status": "passed"},
            ],
        }

    def test_completed_current_run_is_accepted(self) -> None:
        self.assertEqual(
            completed_evidence(self.transcript, self.config)["run_id"], "current"
        )

    def test_incomplete_failed_error_and_stale_runs_are_rejected(self) -> None:
        incomplete = copy.deepcopy(self.transcript)
        incomplete["events"].pop()
        failed = copy.deepcopy(self.transcript)
        failed["events"][-1]["status"] = "failed"
        error = copy.deepcopy(self.transcript)
        error["events"].insert(-1, {"type": "error", "message": "failure"})
        stale = copy.deepcopy(self.transcript)
        stale["run_id"] = "old"
        wrong_task = copy.deepcopy(self.transcript)
        wrong_task["events"][0]["task_id"] = "other-task"
        wrong_evidence = copy.deepcopy(self.transcript)
        wrong_evidence["events"][1]["scoring"]["metadata"]["reins_evidence"][
            "run_id"
        ] = "old"
        failed_score = copy.deepcopy(self.transcript)
        failed_score["events"][1]["scoring"]["score"] = 0
        for transcript in (
            incomplete,
            failed,
            error,
            stale,
            wrong_task,
            wrong_evidence,
            failed_score,
        ):
            with self.subTest(transcript=transcript), self.assertRaises(ValueError):
                completed_evidence(transcript, self.config)


if __name__ == "__main__":
    unittest.main()
