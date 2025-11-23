#!/usr/bin/env bash

set -eufo pipefail

cd "$(git rev-parse --show-toplevel)"

NONINTERACTIVE=1 bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

echo >>/root/.bashrc
echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >>/root/.bashrc
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"

scripts/codex-maintenance.sh
