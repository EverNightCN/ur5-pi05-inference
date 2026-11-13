#!/usr/bin/env bash
# API planning uses a separate CPU environment and the pinned RPent API loop.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
ASSETS="${PI05_ASSETS_DIR:-$ROOT/.assets}"
python3 "$ROOT/tools/fetch_assets.py" --group harness --directory "$ASSETS"
"$UV" python install 3.11.14
"$UV" venv --allow-existing "$ROOT/.venv-harness" --python 3.11.14
"$UV" pip sync --python "$ROOT/.venv-harness/bin/python" "$ROOT/requirements/harness.lock"
"$ROOT/.venv-harness/bin/python" "$ROOT/tools/prepare_harness.py" --assets "$ASSETS"
if [[ ! -f "$ROOT/config/ur5.local.toml" ]]; then
    cp "$ROOT/config/ur5.example.toml" "$ROOT/config/ur5.local.toml"
    echo "Created config/ur5.local.toml. Set local device addresses before running."
fi
"$UV" pip check --python "$ROOT/.venv-harness/bin/python"
bash "$ROOT/run_harness.sh" check
