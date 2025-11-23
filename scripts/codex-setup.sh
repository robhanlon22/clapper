#!/usr/bin/env bash

set -eufo pipefail

cd "$(git rev-parse --show-toplevel)"

apt-get update
apt-get install -y libportaudio2

scripts/codex-maintenance.sh
