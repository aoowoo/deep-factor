## Getting Started

### 1. Installation

To install the required packages:

```bash
conda create -n deepindicator python=3.10
conda activate deepindicator
pip install -r requirements.txt
```

### 2. Feature Engineering

This step runs the data fetching and indicator extraction scripts.

```bash
bash scripts/data.sh
```

### 3. Train

This step runs the training script with the default hyperparameters defined in `scripts/hyper.json`.

```bash
bash scripts/train.sh
```

### 4. Test

This step runs the test script to evaluate the trained model.

```bash
bash scripts/test.sh patch/crypto_model.zip
```
