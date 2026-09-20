# Tema 2 - trabajo practico

This directory contains the explainable artificial intelligence practical work.

## Structure

- `code/`: notebook, helper script, and Python dependencies.
- `data/`: training data, test data, data dictionary, and generated predictions.
- `report/`: LaTeX report, bibliography, assignment statement, and report figures.

## Reproduce the notebook

Open `code/phase-1.ipynb` and run the cells from top to bottom with the working directory set to `code/`. The notebook reads `../data/train.csv` and `../data/test.csv`, and writes predictions to `../data/test_predictions.csv`.

Install the dependencies with:

```text
pip install -r code/requirements.txt
```

The notebook uses a stratified validation split, one-hot encoding for categorical predictors, an `interpret` classification tree, validation metrics, and global model explanations.
