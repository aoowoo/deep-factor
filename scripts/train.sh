#!/bin/bash
set -e
# 使用默认超参训练
echo "--- Running Training ---"
cd "$(dirname "$0")/.."
python src/train.py --config scripts/hyper.json "$@"
