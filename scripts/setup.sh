#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -d vendor/llama.cpp/.git ]; then
  git clone https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
  git -C vendor/llama.cpp checkout "$(cat llama.cpp.rev)"
fi
if [ "$(git -C vendor/llama.cpp rev-parse HEAD)" != "$(cat llama.cpp.rev)" ]; then
  echo 'llama.cpp differs from the tested revision; keeping your checkout.' >&2
fi
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python -r requirements.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target qwentuner-runner -j 8
