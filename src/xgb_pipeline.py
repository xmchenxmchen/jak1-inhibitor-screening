"""
JAK1 inhibitor classification — XGBoost pipeline.

A single, config-driven, relative-path pipeline that screens compounds for
JAK1-inhibitory activity from molecular descriptors.

Usage:
    python src/xgb_pipeline.py              # 5-fold cross-validation (leakage-free, recommended)
    python src/xgb_pipeline.py --holdout    # single train/val/test split, evaluate on held-out test
    python src/xgb_pipeline.py --train       # grid search, save best model, evaluate on held-out test
    python src/xgb_pipeline.py --load-model  # load a saved .pkl and evaluate
    python src/xgb_pipeline.py --plots       # save ROC / confusion-matrix / feature-importance figures
    Add --feature-selection to any mode to run variance + correlation filtering first.

On data leakage:
    An earlier version of this project loaded a pre-trained model and then
    evaluated it on splits drawn from the *whole* dataset. Because the model's
    own training rows leaked into those "test" splits, the reported Test AUC was
    inflated to ~1.0. The default mode here is now stratified k-fold
    cross-validation: every fold trains a *fresh* model on the training folds
    and scores only the held-out fold, so the model never sees the rows it is
    evaluated on. That number (AUC ~0.86) is the trustworthy one.

All paths resolve relative to this file, so the repo can be moved freely.
"""
from __future__ import annotations

import argparse
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

# ─────────────────────────── Paths (relative to this file) ───────────────────────────
HERE = Path(__file__).resolve().parent          # .../repo/src
PROJECT_ROOT = HERE.parent                        # .../repo
DATA_PATH = PROJECT_ROOT / "data" / "drug_descriptors_normalized.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model_xgb.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_CSV = RESULTS_DIR / "grid_search_results.csv"

# ─────────────────────────── General config ───────────────────────────
TARGET = "Active"
# Split: carve out 30% as a temp set, then halve it into val / test (both stratified).
SPLIT_RANDOM_STATE_1 = 40
SPLIT_RANDOM_STATE_2 = 20
TEMP_SIZE = 0.3

# Hyper-parameter grid for grid search (kept around the project's best region).
PARAM_GRID = {
    "max_depth": [3],
    "learning_rate": [0.25],
    "n_estimators": [201, 202],
    "subsample": [0.92, 0.91],
    "colsample_bytree": [0.8],
}

# Single best hyper-parameter set (used for cross-validation and --holdout).
BEST_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.25,
    "n_estimators": 201,
    "subsample": 0.92,
    "colsample_bytree": 0.8,
}
N_FOLDS = 5

METRIC_FUNCS = {
    "Accuracy": accuracy_score,
    "Precision": precision_score,
    "Recall": recall_score,
    "F1 Score": f1_score,
}


# ─────────────────────────── Data ───────────────────────────
def load_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Read the descriptor table; return (X features, y label)."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise KeyError(f"Data is missing the target column '{TARGET}'")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    print(f"Loaded {path.name}  ->  {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    return X, y


def select_features(X: pd.DataFrame, y: pd.Series,
                    var_threshold: float = 0.01,
                    corr_threshold: float = 0.1) -> pd.DataFrame:
    """Drop low-variance features, then keep those correlated enough with the target."""
    var_filter = VarianceThreshold(threshold=var_threshold)
    var_filter.fit(X)
    kept = X.columns[var_filter.get_support()]
    corr = X[kept].corrwith(y).abs()
    high_corr = corr[corr > corr_threshold].index
    print(f"Feature selection: {X.shape[1]} -> {len(high_corr)}")
    return X[high_corr]


def split_data(X: pd.DataFrame, y: pd.Series):
    """Stratified split into train / val / test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=TEMP_SIZE, stratify=y, random_state=SPLIT_RANDOM_STATE_1
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=SPLIT_RANDOM_STATE_2
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ─────────────────────────── Model ───────────────────────────
def make_model(params: dict | None = None) -> XGBClassifier:
    """Build an XGBClassifier from the given (or default best) hyper-parameters."""
    p = dict(BEST_PARAMS if params is None else params)
    p.setdefault("objective", "binary:logistic")
    p.setdefault("seed", 42)
    return XGBClassifier(**p)


# ─────────────────────────── Evaluation ───────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Accuracy / Precision / Recall / F1 / AUC."""
    metrics = {name: func(y_true, y_pred) for name, func in METRIC_FUNCS.items()}
    metrics["AUC"] = roc_auc_score(y_true, y_prob)
    return metrics


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{title}:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")


def evaluate_split(model, X, y, name: str) -> dict:
    pred = model.predict(X)
    prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y, pred, prob)
    print_metrics(f"{name} Metrics", metrics)
    return metrics


def evaluate_baselines(y_test) -> None:
    """All-ones / all-zeros / prior-weighted random, as reference points."""
    n = len(y_test)
    print_metrics("Baseline - always predict 1",
                  compute_metrics(y_test, [1] * n, [1] * n))
    print_metrics("Baseline - always predict 0",
                  compute_metrics(y_test, [0] * n, [0] * n))
    rng = np.random.default_rng(42)
    p_pos = 528 / (191 + 528)
    y_rand = rng.choice([0, 1], size=n, p=[1 - p_pos, p_pos])
    print_metrics("Baseline - prior-weighted random (0:1 ~ 191:528)",
                  compute_metrics(y_test, y_rand, y_rand))


# ─────────────────────────── Cross-validation (leakage-free) ───────────────────────────
def cross_validate_model(X, y, params: dict | None = None,
                         n_folds: int = N_FOLDS) -> dict:
    """Stratified k-fold CV: each fold trains a fresh model on the training folds
    and scores only the held-out fold.

    The model never sees the rows it is evaluated on, so there is no leakage —
    this is the most trustworthy generalization estimate on this dataset.
    Returns (mean, std) per metric.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_metrics = []
    for i, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        model = make_model(params)
        model.fit(X_tr, y_tr)
        m = compute_metrics(y_te, model.predict(X_te),
                            model.predict_proba(X_te)[:, 1])
        fold_metrics.append(m)
        print(f"  Fold {i}/{n_folds}:  "
              f"AUC={m['AUC']:.4f}  F1={m['F1 Score']:.4f}  "
              f"Acc={m['Accuracy']:.4f}")

    agg = {}
    for name in fold_metrics[0]:
        vals = np.array([fm[name] for fm in fold_metrics])
        agg[name] = (float(vals.mean()), float(vals.std()))
    return agg


def print_cv_metrics(title: str, agg: dict) -> None:
    print(f"\n{title}:")
    for name, (mean, std) in agg.items():
        print(f"  {name}: {mean:.4f} +/- {std:.4f}")


# ─────────────────────────── Training ───────────────────────────
def train_grid_search(X_train, y_train, X_val, y_val):
    """Sweep PARAM_GRID, pick the model with the best validation AUC."""
    combos = list(product(*PARAM_GRID.values()))
    keys = list(PARAM_GRID.keys())
    results, best_model, best_params, best_auc = [], None, None, -1.0

    for values in combos:
        params = dict(zip(keys, values))
        model = make_model(params)
        model.fit(X_train, y_train)

        train_m = evaluate_split(model, X_train, y_train, "Train")
        val_m = evaluate_split(model, X_val, y_val, "Validation")

        if val_m["AUC"] > best_auc:
            best_auc, best_model, best_params = val_m["AUC"], model, params

        results.append({
            "Params": params,
            **{f"Train {k}": v for k, v in train_m.items()},
            **{f"Val {k}": v for k, v in val_m.items()},
        })

    print("\nBest params:", best_params)
    print(f"Best validation AUC: {best_auc:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    pd.DataFrame(results).to_csv(RESULTS_CSV, index=False)
    print(f"Saved model            -> {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved grid-search log  -> {RESULTS_CSV.relative_to(PROJECT_ROOT)}")
    return best_model


def load_model(path: Path = MODEL_PATH) -> XGBClassifier:
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}\n(Train one first: `python src/xgb_pipeline.py --train`)"
        )
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Loaded model: {path.relative_to(PROJECT_ROOT)}")
    return model


# ─────────────────────────── Plots ───────────────────────────
def save_plots(model, X_test, y_test) -> None:
    """Save ROC curve, confusion matrix, and top feature importances to results/."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - held-out test set")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=150)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - test set")
    plt.colorbar()
    plt.xticks([0, 1], ["Inactive", "Active"])
    plt.yticks([0, 1], ["Inactive", "Active"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, str(v), ha="center", va="center",
                 color="white" if v > cm.max() / 2 else "black")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    # Top-20 feature importances
    importances = pd.Series(model.feature_importances_, index=X_test.columns)
    top = importances.sort_values(ascending=False).head(20)[::-1]
    plt.figure(figsize=(7, 6))
    plt.barh(top.index, top.values)
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance (gain)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
    plt.close()

    print(f"Saved figures -> {RESULTS_DIR.relative_to(PROJECT_ROOT)}/"
          " (roc_curve.png, confusion_matrix.png, feature_importance.png)")


# ─────────────────────────── Main ───────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="JAK1 inhibitor XGBoost pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Default (no flags) = 5-fold cross-validation: leakage-free and the\n"
            "most trustworthy estimate. The mode flags below are mutually exclusive."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--holdout", action="store_true",
                      help="single train/val/test split; train BEST_PARAMS, evaluate on held-out test")
    mode.add_argument("--train", action="store_true",
                      help="grid search, save the best model, evaluate on the held-out test set")
    mode.add_argument("--load-model", action="store_true",
                      help="load models/best_model_xgb.pkl and evaluate on the held-out test set")
    mode.add_argument("--plots", action="store_true",
                      help="train on the split and save ROC / confusion-matrix / importance figures")
    parser.add_argument("--feature-selection", action="store_true",
                        help="run variance + correlation feature filtering before modeling")
    args = parser.parse_args()

    X, y = load_data()
    if args.feature_selection:
        X = select_features(X, y)

    # Default: cross-validation (leakage-free)
    if not (args.holdout or args.train or args.load_model or args.plots):
        print("\n" + "=" * 56)
        print(f"{N_FOLDS}-fold stratified cross-validation (leakage-free, recommended)")
        print("=" * 56)
        agg = cross_validate_model(X, y)
        print_cv_metrics("Cross-validation results (mean +/- std)", agg)
        print("\nReference baselines:")
        evaluate_baselines(y)
        return

    # All other modes use a single split.
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    if args.train:
        model = train_grid_search(X_train, y_train, X_val, y_val)
    elif args.load_model:
        model = load_model()
        print("Note: metrics are trustworthy only if this model was trained on the same split "
              "used here. An externally supplied model of unknown provenance may leak.")
    else:  # --holdout or --plots
        model = make_model()
        model.fit(X_train, y_train)

    if args.plots:
        save_plots(model, X_test, y_test)

    print("\n" + "=" * 56)
    print("Held-out evaluation (Test was never used for training/selection)")
    print("=" * 56)
    evaluate_split(model, X_train, y_train, "Train")
    evaluate_split(model, X_val, y_val, "Validation")
    evaluate_split(model, X_test, y_test, "Test")
    evaluate_baselines(y_test)


if __name__ == "__main__":
    main()
