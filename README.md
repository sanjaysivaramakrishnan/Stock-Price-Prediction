# Stock Price Prediction with PyTorch RNN 

A minimal PyTorch RNN implementation to predict stock closing prices from historical data using sliding window sequences. This README mirrors the workflow in `main.ipynb`.

## Contents
- `main.ipynb`: End-to-end notebook (data prep → model → train → evaluate)
- `trainset.csv`, `testset.csv`: CSVs with a `Close` column used for training/testing

## Overview
The notebook:
1. Loads `trainset.csv` and `testset.csv`
2. Uses the `Close` price only
3. Scales data with `MinMaxScaler`
4. Creates rolling sequences (default sequence length = 60)
5. Trains a multi-layer RNN and predicts the next step
6. Plots training loss and compares predicted vs actual prices

## Requirements
- Python 3.10+ (notebook shows 3.11)
- PyTorch
- NumPy
- Pandas
- scikit-learn
- Matplotlib
- torchinfo (for model summary)

Install:
```bash
pip install torch numpy pandas scikit-learn matplotlib torchinfo
```

Optional: create and activate a virtual environment first.
```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -U pip
```

## Data format
Both `trainset.csv` and `testset.csv` must include a `Close` column. Only that column is used.

Example (minimum columns):
```csv
Date,Open,High,Low,Close,Volume
2024-01-01,100,105,98,102,123456
...
```

## Notebook structure (high-level)
- Imports and setup
- Load CSVs:
  - `df_train = pd.read_csv(<path>/trainset.csv)`
  - `df_test  = pd.read_csv(<path>/testset.csv)`
- Extract and scale `Close`
- Create sequences with window length 60
- Build `TensorDataset` and `DataLoader`
- Define model:
  - RNN(input_size=1, hidden_size=64, num_layers=2) → Linear(64 → 1)
- Train:
  - Optimizer: Adam(lr=1e-3), Loss: MSELoss, Epochs: 50, Batch size: 64
- Plot training loss
- Predict on test split, inverse-transform, and plot predicted vs actual

## Model summary (from torchinfo)
Input `(batch, seq_len, features) = (64, 60, 1)`:
```
RNN -> [64, 60, 64]
Linear -> [64, 1]
Total params: ~12.7K
```

## How to run
1. Place `trainset.csv` and `testset.csv` in the project folder (or update paths in the notebook).
2. Open `main.ipynb` in Jupyter or VS Code and run all cells.
3. Review:
   - Training loss curve
   - Final predicted vs actual plot
   - Printed last predicted and actual prices

## Key hyperparameters (edit in notebook)
- `seq_length = 60`
- `hidden_size = 64`
- `num_layers = 2`
- `batch_size = 64`
- `num_epochs = 50`
- `lr = 1e-3`

## Tips and troubleshooting
- CUDA: The notebook automatically uses GPU if available (`cuda`) else `cpu`.
- Paths: If on Windows, you can keep absolute paths or switch to relative paths for portability.
- Data leakage: Only fit `MinMaxScaler` on training data; transform test data with the same scaler (already done).
- Sequence coverage: With short datasets and `seq_length=60`, the resulting number of samples will be small; adjust if needed.



