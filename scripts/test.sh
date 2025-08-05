#!/bin/bash
set -e
# 根据训练好的模型进行测试，并且计算相应指标和画图
echo "--- Running Testing ---"
cd "$(dirname "$0")/.."
python src/test.py --config scripts/hyper.json "$@"
