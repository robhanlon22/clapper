#!/usr/bin/env bash

set -eufo pipefail

cd "$(git rev-parse --show-toplevel)"

mise trust
mise install
