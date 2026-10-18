"""Draw the figures for the slide deck at the size of the template's picture slots.

Rebuilds the notebook's training/validation split and refits the selected logistic regression and EBM
(same settings and seed as in code/payment_difficulty_xai.ipynb, so the models are identical).
Run from this folder: python make_slide_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

seed = 411
target = "payment_difficulty_next_month"
categorical = ["sex", "education", "marital_status", "repayment_assistance_plan"]
money = ["credit_limit", "average_bill_6m", "bill_change_sep_to_apr", "total_payment_6m"]
lr_color, ebm_color = "#eb6834", "#1baf7a"
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

plt.rcParams.update({
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df",
    "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "savefig.dpi": 300,
})

train = pd.read_csv("../data/train.csv")
train[categorical] = train[categorical].astype(str)
X = train.drop(columns=["row_id", target])
y = train[target].astype(int)
numeric = [c for c in X.columns if c not in categorical]
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)


def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))


def pretty(name):
    return name.replace("_6m", "").replace("_", " ")


# Logistic regression: selected settings (C = 0.01, signed log of the monetary variables).
encoder = ColumnTransformer(
    [
        ("money", Pipeline([("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
                            ("scale", StandardScaler())]), money),
        ("num", StandardScaler(), [c for c in numeric if c not in money]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical),
    ],
    verbose_feature_names_out=False,
)
logit = Pipeline([("prep", encoder), ("model", LogisticRegression(C=0.01, max_iter=5000))]).fit(X_train, y_train)
features = list(logit.named_steps["prep"].get_feature_names_out())
coefficients = pd.Series(logit.named_steps["model"].coef_[0], index=features)

# 1. The eight largest coefficients.
top = coefficients.reindex(coefficients.abs().sort_values().index[-8:])
fig, ax = plt.subplots(figsize=(4.25, 2.42))
ax.barh(range(len(top)), top, color=np.where(top > 0, lr_color, "#2a78d6"), height=0.65)
ax.set_yticks(range(len(top)), [pretty(n) for n in top.index])
ax.axvline(0, color="#0b0b0b", lw=0.8)
ax.set_xlabel("Log-odds per SD")
ax.set_title("Eight largest coefficients")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(out_dir / "lr_coefficients.png")
plt.close(fig)

# 2. Local explanation of case D (row CR26-125976): contribution = coefficient x standardized value.
case_d = X_validation.loc[train.loc[X_validation.index, "row_id"] == "CR26-125976"]
contributions = pd.Series(logit.named_steps["prep"].transform(case_d)[0] * coefficients.to_numpy(), index=features)
contributions = contributions.groupby(lambda n: next((c for c in categorical if n.startswith(c + "_")), n)).sum()
contributions = contributions.reindex(contributions.abs().sort_values().index[-6:])
probability = logit.predict_proba(case_d)[0, 1]
fig, ax = plt.subplots(figsize=(4.25, 2.42))
ax.barh(range(len(contributions)), contributions, color=lr_color, height=0.65)
ax.set_yticks(range(len(contributions)), [pretty(n) for n in contributions.index])
ax.axvline(0, color="#0b0b0b", lw=0.8)
ax.set_xlabel("Contribution (log-odds)")
ax.set_title(f"Case D: LR p = {probability:.2f} (false alarm)")
ax.grid(axis="y", visible=False)
fig.tight_layout()
fig.savefig(out_dir / "lr_case_d.png")
plt.close(fig)

# EBM: selected settings (256 bins, min_samples_leaf 50, no interactions).
ebm = ExplainableBoostingClassifier(interactions=0, max_bins=256, min_samples_leaf=50, random_state=seed, n_jobs=2)
ebm.fit(X_train, y_train)
explanation = ebm.explain_global()

# 3. One shape function per picture slot.
for term, filename in [("payment_to_bill_ratio_6m", "ebm_ratio"), ("zero_payment_months_6m", "ebm_zero_payment"),
                       ("age", "ebm_age")]:
    data = explanation.data(ebm.term_names_.index(term))
    edges = np.asarray(data["names"], dtype=float)
    scores, lower, upper = (np.asarray(data[k]) for k in ("scores", "lower_bounds", "upper_bounds"))
    low, high = X_train[term].quantile([0.005, 0.995])
    fig, ax = plt.subplots(figsize=(2.15, 2.05))
    ax.fill_between(edges, np.r_[lower, lower[-1]], np.r_[upper, upper[-1]], step="post", color=ebm_color,
                    alpha=0.25, lw=0)
    ax.step(edges, np.r_[scores, scores[-1]], where="post", color=ebm_color, lw=2)
    ax.axhline(0, color="#0b0b0b", lw=0.7)
    ax.set_xlim(low, high)
    visible = (edges[1:] > low) & (edges[:-1] < high)
    span = upper[visible].max() - lower[visible].min()
    ax.set_ylim(lower[visible].min() - 0.08 * span, upper[visible].max() + 0.08 * span)
    ax.set_title(pretty(term), fontsize=9)
    ax.set_ylabel("Log-odds", fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_dir / f"{filename}.png")
    plt.close(fig)

# The refitted models must match the notebook (validation AUC: LR 0.766, EBM 0.775).
lr_auc = roc_auc_score(y_validation, logit.predict_proba(X_validation)[:, 1])
ebm_auc = roc_auc_score(y_validation, ebm.predict_proba(X_validation)[:, 1])
print(f"Validation AUC: LR {lr_auc:.3f}, EBM {ebm_auc:.3f}")
assert round(lr_auc, 3) == 0.766 and round(ebm_auc, 3) == 0.775, "models differ from the notebook"
print("Figures written to", out_dir.resolve())
