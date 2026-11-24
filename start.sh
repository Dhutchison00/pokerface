#!/bin/sh
# Create the virtual environment on first run, then start the server.
set -e

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Creating virtual environment..."
  if ! command -v python3.11 >/dev/null 2>&1; then
    echo "Python 3.11 not found. Install it and ensure python3.11 is on your PATH." >&2
    exit 1
  fi
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --python "$(command -v python3.11)" .venv
  else
    python3.11 -m venv .venv
  fi
fi

if [ ! -f ".venv/.hud-dependencies-installed" ]; then
  echo "Installing dependencies..."
  if command -v uv >/dev/null 2>&1; then
    uv pip install --python .venv/bin/python -r requirements.txt
  else
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
  fi
  touch .venv/.hud-dependencies-installed
fi

exec .venv/bin/python -m preflop_hud "$@"
