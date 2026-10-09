#!/usr/bin/env bash
set -euo pipefail

example_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd "$example_dir/../.." && pwd)"

cd "$repo_dir"
npm ci --ignore-scripts --no-audit --no-fund
npm run build
uv run "$example_dir/prepare.py"
cd "$example_dir/.generated"
uv run karotte run --no-ui --runtime docker --config run_config.json
uv run "$example_dir/export_evidence.py" run_config.json "$example_dir/out"
