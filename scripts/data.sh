set -e
# 抓取数据并且进行特征工程处理

echo "--- Running Data Preparation ---"
cd "$(dirname "$0")/.."

echo "Fetching data..."
python data/get_data.py

echo "Extracting indicators..."
python data/indicator_extractor.py

echo "--- Data preparation complete. ---"
