# Tema 2 - trabajo practico

Explainable classification of `payment_difficulty_next_month` with a decision tree, logistic regression and an
Explainable Boosting Machine (Phase II, final submission).

## Deliverables

| Deliverable | File |
| --- | --- |
| Report (PDF, at most 15 pages) | `report/main.pdf` (source: `report/main.tex`) |
| Notebook with the complete analysis | `code/payment_difficulty_xai.ipynb` |
| Completed test set (`predicted_class` appended) | `data/test.csv` |
| Slide deck for the oral defense (PDF, 4 slides) | `presentation/defense_slides.pdf` (source: `presentation/defense_slides.pptx`) |

## Structure

- `code/payment_difficulty_xai.ipynb`: final, self-contained notebook. It fits, compares and interprets the three
  model families, writes every report figure to `report/figures/` and appends the predictions to `data/test.csv`.
- `code/phase_*.ipynb` and `code/common.py`: earlier per-model notebooks from Phases I and II, kept for reference.
  The final notebook supersedes them.
- `data/`: training data, test data and data dictionary.
- `report/`: LaTeX report, bibliography, assignment statement and figures.
- `presentation/`: the defense deck, built from the UPM template `presentation.pptx` by `build_deck.py`, with figures
  drawn by `make_slide_figures.py`.

## Reproduce the analysis

Install the dependencies and run the notebook from top to bottom with `code/` as the working directory:

```text
pip install -r code/requirements.txt
```

The notebook reads `../data/train.csv` and `../data/test.csv`. In Google Colab, upload both files to the
session's working directory (Files panel) before running; the first cell installs InterpretML. A full run takes about 10-15 minutes, most of it the EBM grid search. All random
seeds are fixed (411).

Rebuild the report with `latexmk -pdf main.tex` in `report/`. Rebuild the deck by running
`python make_slide_figures.py` and `python build_deck.py` in `presentation/` (requires `python-pptx`), then
export `defense_slides.pptx` to PDF from PowerPoint.
