#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p test-results
python -m pytest . -vv \
  --junitxml=test-results/junit.xml | tee test-results/pytest.log