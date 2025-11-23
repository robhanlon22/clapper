#!/usr/bin/env bash

set -eufo pipefail

cd "$(git rev-parse --show-toplevel)"

uv sync

pre-commit install --install-hooks
