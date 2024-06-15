## Setup

1. Clone the repository.
2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
3. Place the data files in the `data/` directory.
4. Run the training script:
    ```bash
    python src/main.py
    ```

## Model

The model architecture includes:
- Convolutional layers (Conv1D)
- LSTM layers
- Self-attention mechanism
- Fully connected layers

## Data

The data includes mRNA expression, copy number alterations (CNA), and mutation data for various cancer cell lines, which are combined to predict protein levels.

## Results

The model's performance is evaluated using Mean Absolute Error (MAE)