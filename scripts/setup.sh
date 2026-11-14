#!/bin/bash
# One-time setup: pinned mlx-speech + a venv with the harness deps.
# Usage: scripts/setup.sh   (run from the repo root; needs git and uv)
set -euo pipefail
cd "$(dirname "$0")/.."
MLX_SPEECH_REV=2ecabcd   # the mlx-speech commit every number in README.md was measured on
[ -d vendor/mlx-speech ] || git clone https://github.com/appautomaton/mlx-speech vendor/mlx-speech
git -C vendor/mlx-speech checkout -q "$MLX_SPEECH_REV"
uv venv -q --python 3.13 .venv
uv pip install -q --python .venv/bin/python -e vendor/mlx-speech mlx-whisper zhconv
# harness/src-base is the package copy that model-proposed patches are applied to
rm -rf harness/src-base && mkdir -p harness/src-base
cp -R vendor/mlx-speech/src/mlx_speech harness/src-base/mlx_speech
echo "ok: .venv ready, mlx-speech @ $MLX_SPEECH_REV"
