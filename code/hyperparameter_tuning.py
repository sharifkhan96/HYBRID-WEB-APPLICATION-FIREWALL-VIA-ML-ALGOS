"""
Hyperparameter tuning for the binary (normal vs. anomalous) task.

Addresses the "hyperparameters were fixed rather than tuned" limitation:
runs a randomized search (3-fold CV, macro-F1) over a real parameter
distribution for each of the six models, then evaluates the tuned
estimator on both the in-distribution test set and the OOD lexical test
set (zero retraining, same protocol as run_experiments.py) so we can see
whether tuning changes the ID accuracy, the OOD accuracy, or the
central generalization-gap finding.

Usage: python3 hyperparameter_tuning.py
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from features import WAFFeatureExtractor

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "archive"
RESULTS = ROOT / "results"
METRICDIR = RESULTS / "metrics"
MODELDIR = RESULTS / "models"
FIGDIR = RESULTS / "figures"
LOGDIR = ROOT / "logs"
N_ITER = 20
CV_FOLDS = 3


def load_split(name):
    df = pd.read_csv(DATA_DIR / name)
    df = df.dropna(subset=["payload", "label"]).reset_index(drop=True)
    df["payload"] = df["payload"].astype(str)
    return df


PARAM_DISTRIBUTIONS = {
    "LogisticRegression": {
        "estimator": lambda: LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "params": {
            "C": uniform(0.01, 20),
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
    },
    "MultinomialNB": {
        "estimator": lambda: MultinomialNB(),
        "params": {
            "alpha": uniform(0.001, 2.0),
            "fit_prior": [True, False],
        },
    },
    "LinearSVM": {
        "estimator": lambda: CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, max_iter=5000), cv=3),
        "params": {
            "estimator__C": uniform(0.001, 10),
        },
    },
    "RandomForest": {
        "estimator": lambda: RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
        "params": {
            "n_estimators": randint(100, 500),
            "max_depth": [None, 10, 20, 40, 60],
            "min_samples_leaf": randint(1, 5),
            "max_features": ["sqrt", "log2", None],
        },
    },
    "XGBoost": {
        "estimator": lambda: XGBClassifier(
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
        "params": {
            "n_estimators": randint(100, 500),
            "max_depth": randint(3, 12),
            "learning_rate": uniform(0.01, 0.3),
            "subsample": uniform(0.6, 0.4),
            "colsample_bytree": uniform(0.6, 0.4),
        },
    },
    "MLP": {
        "estimator": lambda: MLPClassifier(random_state=RANDOM_STATE, early_stopping=True),
        "params": {
            "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32)],
            "alpha": uniform(1e-5, 1e-2),
            "learning_rate_init": uniform(1e-4, 1e-2),
            "max_iter": [150, 250],
        },
    },
}


def main():
    log_path = LOGDIR / "hyperparameter_tuning.log"
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    t_start = time.perf_counter()
    train = load_split("payload_train.csv")
    test = load_split("payload_test.csv")
    lexical = load_split("payload_test_lexical.csv")

    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    y_test = le.transform(test["label"])
    y_lex = le.transform(lexical["label"])

    extractor = WAFFeatureExtractor(max_features=3000, mode="combined")
    X_train = extractor.fit_transform(train["payload"])
    X_test = extractor.transform(test["payload"])
    X_lex = extractor.transform(lexical["payload"])
    log(f"Feature matrix: {X_train.shape[1]} dims")

    # untuned baseline numbers, recomputed here for a like-for-like comparison
    untuned = pd.read_csv(METRICDIR / "results_binary.csv").set_index("model")

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    best_params_all = {}

    for name, spec in PARAM_DISTRIBUTIONS.items():
        log(f"\n{'='*70}\nTuning {name} ({N_ITER} candidates x {CV_FOLDS}-fold CV)\n{'='*70}")
        t0 = time.perf_counter()
        search = RandomizedSearchCV(
            spec["estimator"](), spec["params"], n_iter=N_ITER, cv=skf,
            scoring="f1_macro", random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        search_time = time.perf_counter() - t0
        best_model = search.best_estimator_
        best_params_all[name] = search.best_params_
        log(f"  Best CV macro-F1: {search.best_score_:.4f}  ({search_time:.1f}s)")
        log(f"  Best params: {search.best_params_}")

        # in-distribution test
        y_pred_id = best_model.predict(X_test)
        acc_id = accuracy_score(y_test, y_pred_id)
        _, _, f1_id, _ = precision_recall_fscore_support(y_test, y_pred_id, average="macro", zero_division=0)

        # OOD lexical test, zero retraining
        y_pred_ood = best_model.predict(X_lex)
        acc_ood = accuracy_score(y_lex, y_pred_ood)
        _, _, f1_ood, _ = precision_recall_fscore_support(y_lex, y_pred_ood, average="macro", zero_division=0)

        row = dict(
            model=name,
            untuned_id_acc=untuned.loc[name, "accuracy"],
            untuned_id_f1=untuned.loc[name, "f1_macro"],
            tuned_id_acc=acc_id,
            tuned_id_f1=f1_id,
            untuned_ood_acc=np.nan,
            tuned_ood_acc=acc_ood,
            tuned_ood_f1=f1_ood,
            cv_best_f1=search.best_score_,
            search_time_s=search_time,
        )
        rows.append(row)
        log(f"  ID:  untuned acc={untuned.loc[name,'accuracy']:.4f} f1={untuned.loc[name,'f1_macro']:.4f}"
            f"  ->  tuned acc={acc_id:.4f} f1={f1_id:.4f}")
        log(f"  OOD: tuned acc={acc_ood:.4f} f1={f1_ood:.4f}")

        joblib.dump(best_model, MODELDIR / f"tuned_binary_{name}.joblib")

    # fill in untuned OOD numbers from the existing OOD results file
    ood_untuned = pd.read_csv(METRICDIR / "results_binary_lexical_ood.csv").set_index("model")
    df = pd.DataFrame(rows).set_index("model")
    df["untuned_ood_acc"] = ood_untuned["accuracy"]
    df["untuned_ood_f1"] = ood_untuned["f1_macro"]
    df = df.reset_index()

    df.to_csv(METRICDIR / "hyperparameter_tuning.csv", index=False)
    with open(METRICDIR / "hyperparameter_tuning_best_params.json", "w") as f:
        json.dump(best_params_all, f, indent=2, default=str)

    joblib.dump(extractor, MODELDIR / "tuned_binary_feature_extractor.joblib")
    joblib.dump(le, MODELDIR / "tuned_binary_label_encoder.joblib")

    log(f"\nTotal elapsed: {(time.perf_counter()-t_start)/60:.1f} min")
    log("\n=== Summary (sorted by tuned OOD accuracy) ===")
    log(df.sort_values("tuned_ood_acc", ascending=False)[
        ["model", "untuned_id_acc", "tuned_id_acc", "untuned_ood_acc", "tuned_ood_acc"]
    ].to_string(index=False))

    with open(log_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
