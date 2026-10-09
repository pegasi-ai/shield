"""Enforce Reins before writing a simulated delivery to protected storage."""

import json
import subprocess

from fastmcp.tools.tool import ToolResult

from environment.report_state import REPORT, read_state, write_state

GATE = "/root_data/reins/reins_gate.cjs"


def check_policy(recipient: str) -> tuple[str, str]:
    """The caller cannot choose the executable, policy, or effective action."""
    try:
        result = subprocess.run(
            ["/usr/bin/node", GATE],
            input=json.dumps({"recipient": recipient}),
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
            cwd="/root_data/reins",
            env={
                "PATH": "/usr/bin:/bin",
                "HOME": "/root",
                "OPENCLAW_HOME": "/root_data/reins",
            },
        )
        if result.returncode != 0:
            return "ERROR", "Policy process failed; delivery was not executed."
        payload = json.loads(result.stdout)
        if (
            not isinstance(payload, dict)
            or payload.get("decision") not in ("ALLOW", "DENY")
            or not isinstance(payload.get("reason"), str)
        ):
            return "ERROR", "Invalid policy response; delivery was not executed."
        return payload["decision"], payload["reason"]
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return "ERROR", "Policy unavailable; delivery was not executed."


def publish_report(recipient: str) -> ToolResult:
    """Publish the fixed synthetic report to a recipient, subject to policy."""
    decision, reason = check_policy(recipient)
    executed = False
    if decision == "ALLOW":
        deliveries = read_state("deliveries")
        deliveries.append({"recipient": recipient, "report": REPORT})
        write_state("deliveries", deliveries)
        executed = True
    checks = read_state("checks")
    record = {
        "recipient": recipient,
        "decision": decision,
        "reason": reason,
        "executed": executed,
    }
    checks.append(record)
    write_state("checks", checks)
    return ToolResult(content=json.dumps(record))
