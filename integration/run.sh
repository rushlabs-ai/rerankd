#!/usr/bin/env sh
# Build rerankd for linux/amd64 and run the integration test on Modal (needs `modal setup`).
set -eu
cd "$(dirname "$0")/.."
mkdir -p integration/build
CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -trimpath -o integration/build/rerankd .
modal run integration/modal_integration.py "$@"   # add --gpu for the L4 run
