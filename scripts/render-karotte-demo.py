#!/usr/bin/env python3
"""Render a deterministic GIF replay of a recorded Reins + Karotte run.

Usage:
    uv run --with pillow scripts/render-karotte-demo.py \
      --evidence path/to/evidence.json --output public/karotte-demo.gif

The input must come from the quickstart runner. This script does not run the
environment, generate outcomes, or substitute missing evidence. The animation
is explicitly labeled as a replay of a scripted regression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1100, 690
BG = "#0b1018"
PANEL = "#111923"
LINE = "#2b3747"
TEXT = "#eff4fa"
MUTED = "#a2b1c3"
BLUE = "#8cbcff"
RED = "#ff9a9e"
GREEN = "#7ee0b0"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def one_line(value: Any, field: str, maximum: int = 240) -> str:
    require(
        isinstance(value, str) and bool(value.strip()), f"{field} must be nonempty text"
    )
    require(len(value) <= maximum, f"{field} must be at most {maximum} characters")
    require(
        not any(ord(char) < 32 for char in value), f"{field} must be one printable line"
    )
    return value


def load_evidence(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    data = json.loads(raw)
    require(isinstance(data, dict), "evidence must be an object")
    require(data.get("schema_version") == 1, "unsupported or missing schema_version")
    require(
        data.get("mode") == "scripted_regression", "mode must be scripted_regression"
    )
    one_line(data.get("run_id"), "run_id", 100)
    one_line(data.get("scenario"), "scenario", 100)
    require(isinstance(data.get("commands"), dict), "commands is required")
    one_line(data["commands"].get("quickstart"), "commands.quickstart", 190)
    checks = data.get("checks")
    require(
        isinstance(checks, list) and len(checks) == 2,
        "exactly two recorded checks are required",
    )
    for index, check in enumerate(checks):
        require(isinstance(check, dict), f"checks[{index}] must be an object")
        one_line(check.get("recipient"), f"checks[{index}].recipient", 65)
        require(
            check.get("decision") in ("DENY", "ALLOW"), "decision must be DENY or ALLOW"
        )
        require(
            type(check.get("executed")) is bool, "executed must be a recorded boolean"
        )
    require(
        checks[0]["decision"] == "DENY" and checks[0]["executed"] is False,
        "first check must record DENY and executed=false",
    )
    require(
        checks[1]["decision"] == "ALLOW" and checks[1]["executed"] is True,
        "second check must record ALLOW and executed=true",
    )
    require(checks[0]["recipient"] != checks[1]["recipient"], "recipients must differ")
    deliveries = data.get("deliveries")
    require(isinstance(deliveries, list), "recorded deliveries array is required")
    for index, delivery in enumerate(deliveries):
        require(isinstance(delivery, dict), f"deliveries[{index}] must be an object")
        one_line(delivery.get("recipient"), f"deliveries[{index}].recipient", 65)
    require(
        len(deliveries) == 1 and deliveries[0]["recipient"] == checks[1]["recipient"],
        "this quickstart requires exactly one actual delivery, to the allowed recipient",
    )
    grader = data.get("grader")
    require(isinstance(grader, dict), "grader object is required")
    for field in ("score", "max_score"):
        value = grader.get(field)
        require(
            type(value) in (float, int) and math.isfinite(value),
            f"grader.{field} must be numeric",
        )
    require(
        0 <= grader["score"] <= grader["max_score"] and grader["max_score"] > 0,
        "grader score must be between zero and max_score",
    )
    provenance = data.get("provenance")
    require(isinstance(provenance, dict), "provenance object is required")
    one_line(provenance.get("environment_repo"), "provenance.environment_repo")
    one_line(provenance.get("environment_commit"), "provenance.environment_commit", 64)
    return data, hashlib.sha256(raw).hexdigest()[:12]


def font(size: int, override: str | None = None) -> ImageFont.FreeTypeFont:
    candidates = ([override] if override else []) + [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    raise ValueError(
        "No monospace font found. Pass --font with a local TrueType font path."
    )


def wrapped(
    draw: ImageDraw.ImageDraw, text: str, face: ImageFont.FreeTypeFont, width: int
) -> list[str]:
    """Wrap even long URLs without eliding any recorded value."""
    lines: list[str] = []
    while text:
        end = len(text)
        while end > 1 and draw.textlength(text[:end], font=face) > width:
            end -= 1
        if end < len(text):
            space = text.rfind(" ", 0, end + 1)
            if space > end // 2:
                end = space
        lines.append(text[:end])
        text = text[end:].lstrip()
    return lines


def render_frame(
    data: dict[str, Any], digest: str, stage: int, font_path: str | None
) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    small, body, large = font(16, font_path), font(20, font_path), font(32, font_path)
    draw.rounded_rectangle(
        (22, 20, WIDTH - 22, HEIGHT - 20), radius=16, fill=PANEL, outline=LINE, width=1
    )
    draw.line((23, 66, WIDTH - 23, 66), fill=LINE)
    for x, color in ((44, "#ed6a5e"), (66, "#f4bf4f"), (88, "#61c554")):
        draw.ellipse((x, 38, x + 10, 48), fill=color)
    draw.text((118, 34), "reins / karotte", font=small, fill=MUTED)
    draw.text((760, 34), "SCRIPTED REGRESSION", font=small, fill=BLUE)
    draw.text((52, 89), "Reins + Karotte", font=large, fill=TEXT)
    draw.text(
        (52, 133),
        "Real policy + Docker run. Simulated delivery. No API key.",
        font=small,
        fill=MUTED,
    )

    command = "$ " + data["commands"]["quickstart"]
    command_lines = wrapped(draw, command, body, 980)
    require(len(command_lines) <= 3, "quickstart command is too long for this layout")
    for line_index, line in enumerate(command_lines):
        draw.text((52, 171 + line_index * 25), line, font=body, fill=BLUE)
    draw.line((52, 257, 1048, 257), fill=LINE)

    for index, check in enumerate(data["checks"]):
        y = 283 + index * 113
        active = stage >= index + 1
        if active:
            color = RED if check["decision"] == "DENY" else GREEN
            draw.text(
                (52, y), f"{index + 1:02}  {check['recipient']}", font=body, fill=TEXT
            )
            decision = (
                f"{check['decision']}    executed={str(check['executed']).lower()}"
            )
            draw.rounded_rectangle(
                (99, y + 36, 540, y + 77), radius=6, fill="#19212b", outline=LINE
            )
            draw.text((112, y + 45), decision, font=body, fill=color)
        else:
            draw.text(
                (52, y),
                f"{index + 1:02}  [recorded result follows]",
                font=body,
                fill="#657387",
            )

    draw.line((52, 510, 1048, 510), fill=LINE)
    if stage >= 3:
        draw.text((52, 532), "ACTUAL DELIVERIES", font=small, fill=MUTED)
        delivery = data["deliveries"][0]["recipient"]
        delivery_lines = wrapped(draw, delivery, body, 725)
        require(
            len(delivery_lines) <= 2, "delivery recipient is too long for this layout"
        )
        for line_index, line in enumerate(delivery_lines):
            draw.text((52, 562 + line_index * 26), line, font=body, fill=GREEN)
        draw.text((822, 532), "GRADER SCORE", font=small, fill=MUTED)
        score = f"{data['grader']['score']:g} / {data['grader']['max_score']:g}"
        draw.text((822, 558), score, font=large, fill=GREEN)
    else:
        draw.text(
            (52, 552),
            "Replaying recorded quickstart evidence...",
            font=body,
            fill=MUTED,
        )

    draw.text(
        (52, 628), f"Recorded JSON replay | SHA256 {digest}", font=small, fill=MUTED
    )
    commit = data["provenance"]["environment_commit"][:12]
    label = f"Karotte {commit}"
    draw.text(
        (1048 - draw.textlength(label, font=small), 628), label, font=small, fill=MUTED
    )
    return image


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        required=True,
        help="JSON emitted by the actual quickstart run",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination GIF")
    parser.add_argument(
        "--poster", type=Path, help="Optional final-frame PNG for review"
    )
    parser.add_argument("--font", help="Optional monospace TrueType font path")
    args = parser.parse_args()
    try:
        data, digest = load_evidence(args.evidence)
        frames = [render_frame(data, digest, stage, args.font) for stage in range(4)]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(
            args.output,
            save_all=True,
            append_images=frames[1:],
            duration=[2200, 3400, 3400, 6000],
            loop=0,
            optimize=False,
            disposal=2,
        )
        if args.poster:
            args.poster.parent.mkdir(parents=True, exist_ok=True)
            frames[-1].save(args.poster)
        with Image.open(args.output) as result:
            durations = []
            for index in range(result.n_frames):
                result.seek(index)
                durations.append(result.info["duration"])
            require(
                result.size == (WIDTH, HEIGHT), "GIF dimensions failed verification"
            )
            require(sum(durations) == 15000, "GIF duration failed verification")
            require(result.info.get("loop") == 0, "GIF loop failed verification")
        print(
            json.dumps(
                {
                    "gif": str(args.output),
                    "evidence_sha256_prefix": digest,
                    "frames": len(frames),
                    "duration_ms": sum(durations),
                    "size": [WIDTH, HEIGHT],
                }
            )
        )
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
