#!/usr/bin/env bash
set -e

dir="$(dirname "$0")"
parentdir="$(dirname "$dir")"
cd $parentdir

jupyter nbconvert \
  --to markdown \
  --output-dir docs/tutorials \
  "tutorials/*.ipynb"