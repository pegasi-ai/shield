"""Root-owned state. The simulated delivery is separate from the policy log."""

import json

from karotte import ProtectedStore

REPORT = "Quarterly example report. Synthetic data only."


def read_state(name: str) -> list[dict]:
    return json.loads(ProtectedStore().read(name, str))


def write_state(name: str, value: list[dict]) -> None:
    ProtectedStore().write(name, json.dumps(value))
