# Tema 2 - trabajo practico

Explainable classification of `payment_difficulty_next_month` with a decision tree, logistic regression and an
Explainable Boosting Machine (Phase II, final submission).

## Deliverables

| Deliverable | File |
| --- | --- |
| Report (PDF, at most 15 pages) | `report/main.pdf` (source: `report/main.tex`) |
| Notebook with the complete analysis | `code/payment_difficulty_xai.ipynb` |
| Completed test set (`predicted_class` appended) | `data/test.csv` |
| Slide deck for the oral defense (PDF, 4 slides) | `presentation/presentation.pdf` (source: `presentation/presentation.tex`) |

## Structure

- `code/payment_difficulty_xai.ipynb`: final, self-contained notebook. It fits, compares and interprets the three
  model families, writes every report figure to `report/figures/` and appends the predictions to `data/test.csv`.
- `code/phase_*.ipynb` and `code/common.py`: earlier per-model notebooks from Phases I and II, kept for reference.
  The final notebook supersedes them.
- `data/`: training data, test data and data dictionary.
- `report/`: LaTeX report, bibliography, assignment statement and figures.
- `presentation/`: Beamer slide deck (the `.pptx` file is an unused template).

## Reproduce the analysis

Install the dependencies and run the notebook from top to bottom with `code/` as the working directory:

```text
pip install -r code/requirements.txt
```

The notebook reads `../data/train.csv` and `../data/test.csv`. In Google Colab it installs InterpretML and asks
you to upload both files. A full run takes about 10-15 minutes, most of it the EBM grid search. All random
seeds are fixed (411).

Rebuild the PDFs with `latexmk -pdf main.tex` in `report/` and `latexmk -pdf presentation.tex` in `presentation/`.
