#!/usr/bin/env bash

set -eufo pipefail

cd "$(git rev-parse --show-toplevel)"

pre-commit install --install-hooks

if command -v brew > /dev/null; then
  brew bundle
fi

uv sync
