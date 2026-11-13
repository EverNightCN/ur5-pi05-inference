#!/usr/bin/env bash
# Rebuild the pinned JetPack 6.2 inference environment in this checkout.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV="${UV:-$HOME/.local/bin/uv}"
ASSETS="${PI05_ASSETS_DIR:-$ROOT/.assets}"
test "$(uname -m)" = aarch64
grep -q 'R36 .*REVISION: 4.3' /etc/nv_tegra_release
test -f /usr/local/cuda-12.6/targets/aarch64-linux/include/cuda.h
test -f /usr/include/python3.10/Python.h
command -v gcc >/dev/null
test -x "$UV"
python3 "$ROOT/tools/fetch_assets.py" --group inference --directory "$ASSETS"
"$UV" venv --allow-existing "$ROOT/.venv" --python /usr/bin/python3.10
"$UV" pip install --python "$ROOT/.venv/bin/python" -r "$ROOT/requirements/inference.lock"
"$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps     "$ASSETS/torch-2.11.0-cp310-cp310-linux_aarch64.whl"     "$ASSETS/triton-3.6.0-cp310-cp310-linux_aarch64.whl"
"$ROOT/.venv/bin/python" "$ROOT/tools/prepare_runtime.py" --assets "$ASSETS"
mkdir -p "$ROOT/.runtime/logs"
if [[ ! -f "$ROOT/config/ur5.local.toml" ]]; then
    cp "$ROOT/config/ur5.example.toml" "$ROOT/config/ur5.local.toml"
    echo "Created config/ur5.local.toml. Set this site's robot IP, camera and gripper devices."
fi
"$UV" pip check --python "$ROOT/.venv/bin/python"
bash "$ROOT/run.sh" "$ROOT/tools/check_environment.py"
