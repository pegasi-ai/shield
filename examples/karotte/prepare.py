# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "karotte @ git+https://github.com/preferencemodel/karotte.git@31c693eb925f5ab768e1c3294e331187b4f1b6c9",
# ]
# ///
"""Generate Karotte's pinned scaffold and add the report-delivery task."""

import hashlib
import json
import shutil
import subprocess
import tarfile
import uuid
from pathlib import Path

from karotte.create_env import create_env

EXAMPLE = Path(__file__).resolve().parent
REPO = EXAMPLE.parent.parent
GENERATED = EXAMPLE / ".generated"
NODE_IMAGE = "node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392"


def main() -> None:
    if not GENERATED.exists():
        create_env(GENERATED, ["default"], vendor_karotte=True, no_lock=True)
        # This example has one task and no student data or model dependency.
        shutil.rmtree(GENERATED / "src/environment/tasks")
        (GENERATED / "src/environment/tasks").mkdir()
        (GENERATED / "src/environment/tasks/__init__.py").touch()
    containerfile = GENERATED / "Containerfile"
    base_containerfile = containerfile.read_text().split("\n# Node's official binary")[
        0
    ]
    containerfile.write_text(
        base_containerfile
        + "\n# Node's official binary works with Amazon Linux 2023.\n"
        + f"COPY --from={NODE_IMAGE} /usr/local/bin/node /usr/bin/node\n"
        + "RUN dnf install -y libstdc++ && dnf clean all\n"
    )

    shutil.copytree(EXAMPLE / "src", GENERATED / "src", dirs_exist_ok=True)
    bundle = GENERATED / "root_data/reins"
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXAMPLE / "reins_gate.cjs", bundle / "reins_gate.cjs")
    packed = subprocess.run(
        [
            "npm",
            "pack",
            "--ignore-scripts",
            "--json",
            "--pack-destination",
            str(GENERATED),
        ],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    )
    tarball = GENERATED / json.loads(packed.stdout)[0]["filename"]
    for filename in ("package.json", "package-lock.json"):
        shutil.copy2(REPO / filename, bundle / filename)
    subprocess.run(
        [
            "npm",
            "ci",
            "--ignore-scripts",
            "--omit=dev",
            "--no-audit",
            "--no-fund",
            "--prefix",
            str(bundle),
        ],
        check=True,
    )
    package_dir = bundle / "node_modules/@pegasi-ai/reins"
    package_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball) as archive:
        for member in archive.getmembers():
            relative = Path(member.name).relative_to("package")
            if member.isfile():
                destination = package_dir / relative
                if not destination.resolve().is_relative_to(package_dir.resolve()):
                    raise ValueError("Packed package contains an unsafe path")
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source:
                    destination.write_bytes(source.read())
    files = [
        REPO / "package.json",
        REPO / "package-lock.json",
        EXAMPLE / "reins_gate.cjs",
    ]
    files += sorted((REPO / "src").rglob("*.ts"))
    files += sorted((REPO / "dist").rglob("*.js"))
    files += sorted((EXAMPLE / "src").rglob("*.py"))
    hashes = {
        str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    manifest = {
        "reins_base_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "source_sha256": hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        "files_sha256": hashes,
        "node_image": NODE_IMAGE,
    }
    (bundle / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    config = json.loads((EXAMPLE / "run_config.json").read_text())
    config["run_id"] = "reins-report-delivery-" + uuid.uuid4().hex[:12]
    config["transcript_file"] = f"out/{config['run_id']}.json"
    (GENERATED / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")
    subprocess.run(["uv", "sync", "--python", "3.12"], cwd=GENERATED, check=True)


if __name__ == "__main__":
    main()
