"""Extract the root-owned judge's evidence from a completed Karotte transcript."""

import json
import shutil
import sys
from pathlib import Path


def completed_evidence(transcript: dict, config: dict) -> dict:
    events = transcript["events"]
    if transcript["run_id"] != config["run_id"]:
        raise ValueError("Transcript does not belong to this run.")
    if (
        not events
        or events[-1].get("type") != "task_completed"
        or events[-1].get("status") != "passed"
    ):
        raise ValueError("Transcript does not end with a completed, passed task.")
    if any(event.get("type") == "error" for event in events):
        raise ValueError("Transcript contains an error event.")
    starts = [event for event in events if event.get("type") == "task_started"]
    if (
        len(starts) != 1
        or starts[0].get("task_id") != config["task_id"]
        or starts[0].get("run_id") != config["run_id"]
    ):
        raise ValueError("Unexpected task or run identity.")
    scores = [event["scoring"] for event in events if event.get("type") == "scoring"]
    evidence = [
        event["scoring"]["metadata"]["reins_evidence"]
        for event in events
        if "reins_evidence" in event.get("scoring", {}).get("metadata", {})
    ]
    if len(evidence) != 1 or len(scores) != 1:
        raise ValueError("Expected exactly one completed report-delivery score.")
    result = evidence[0]
    if result["run_id"] != config["run_id"]:
        raise ValueError("Evidence does not belong to this run.")
    if (
        result["grader"]["score"] != 1
        or scores[0]["score"] != 1
        or scores[0]["continue_task"] is not True
    ):
        raise ValueError("Report-delivery regression failed.")
    return result


def main() -> None:
    config_path, output_dir = map(Path, sys.argv[1:])
    config = json.loads(config_path.read_text())
    transcript_path = config_path.parent / config["transcript_file"]
    transcript = json.loads(transcript_path.read_text())
    evidence = completed_evidence(transcript, config)
    run_dir = output_dir / evidence["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcript_path, run_dir / "transcript.json")
    (run_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcript_path, output_dir / "transcript.json")
    shutil.copy2(run_dir / "evidence.json", output_dir / "evidence.json")
    for check in evidence["checks"]:
        print(
            f"{check['decision']:5} {check['recipient']:24} executed={check['executed']}"
        )
    print(f"Protected deliveries: {len(evidence['deliveries'])}; regression score: 1/1")
    print(f"Completed run: {evidence['run_id']}")
    print(f"Evidence: {run_dir / 'evidence.json'}")


if __name__ == "__main__":
    main()
