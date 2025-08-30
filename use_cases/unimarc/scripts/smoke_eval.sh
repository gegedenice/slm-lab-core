#!/usr/bin/env bash
set -euo pipefail

# This script runs a full prep -> train -> eval pipeline for the current use-case.
# It is intended to be run from the project root via `make smoke`.
# The `USE_CASE` variable is expected to be set by the Makefile.

echo " smoketest"
echo "================================="
echo " USE CASE: ${USE_CASE}"
echo "================================="

echo "\n[1/3] Preparing data..."
make prep USE_CASE=${USE_CASE}

echo "\n[2/3] Running a short training..."
# We can override make variables for the smoke test.
# For example, run for fewer steps.
# This requires the training script to accept these overrides.
# For now, we run the default config.
make train USE_CASE=${USE_CASE}

echo "\n[3/3] Running evaluation..."
make eval USE_CASE=${USE_CASE}

echo "\n smoketest complete for ${USE_CASE}!"
