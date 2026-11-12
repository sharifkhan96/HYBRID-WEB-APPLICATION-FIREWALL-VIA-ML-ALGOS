"""
End-to-end experiment runner for the ML-based WAF payload classification study.

Reproduces every number and figure reported in the paper (paper/paper.tex):
  1. Dataset statistics
  2. Binary classification (norm vs. anomalous) model comparison
  3. Multi-class attack-type classification model comparison
  4. Out-of-distribution (lexical) generalization test
  5. Feature ablation (TF-IDF only / engineered only / combined)
  6. 5-fold cross-validation of the best model
  7. Inference latency benchmark
  8. All figures used in the paper

Usage:
    python3 run_experiments.py

Outputs are written under ../results/ (metrics, models, figures) and a full
console log is tee'd to ../logs/run_experiments.log.
"""
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

from features import WAFFeatureExtractor

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "archive"
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
MODELDIR = RESULTS / "models"
METRICDIR = RESULTS / "metrics"
LOGDIR = ROOT / "logs"

for d in (FIGDIR, MODELDIR, METRICDIR, LOGDIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- plotting --
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PALETTE = ["#2a78d6", "#008300", "#e87ba4", "#eda100",
           "#1baf7a", "#eb6834", "#4a3aa7", "#e34948"]
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "font.family": "sans-serif",
})


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def log(msg=""):
    print(msg)


# --------------------------------------------------------------- data load --
def load_data():
    train = pd.read_csv(DATA_DIR / "payload_train.csv")
    test = pd.read_csv(DATA_DIR / "payload_test.csv")
    lexical = pd.read_csv(DATA_DIR / "payload_test_lexical.csv")
    for df in (train, test, lexical):
        df.dropna(subset=["payload", "label"], inplace=True)
        df["payload"] = df["payload"].astype(str)
    return train, test, lexical


def dataset_stats(train, test, lexical):
    rows = []
    for name, df in [("train", train), ("test", test), ("lexical_ood_test", lexical)]:
        row = {"split": name, "n_samples": len(df)}
        for cls, cnt in df["label"].value_counts().items():
            row[f"label_{cls}"] = cnt
        rows.append(row)
    stats = pd.DataFrame(rows)
    stats.to_csv(METRICDIR / "dataset_stats.csv", index=False)
    log(f"[dataset_stats]\n{stats.to_string(index=False)}\n")

    # attack-type distribution (train, in-distribution)
    at = train["attack_type"].value_counts().rename_axis("attack_type").reset_index(name="count")
    at.to_csv(METRICDIR / "attack_type_distribution_train.csv", index=False)

    # figure: class distribution
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, df) in zip(axes, [("Train", train), ("Test", test)]):
        vc = df["attack_type"].value_counts()
        ax.bar(vc.index, vc.values, color=PALETTE[: len(vc)])
        ax.set_title(f"{name} set attack-type distribution")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGDIR / "dataset_distribution.png", bbox_inches="tight")
    plt.close(fig)
    return stats


# ------------------------------------------------------------- model zoo ---
def build_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "MultinomialNB": MultinomialNB(),
        "LinearSVM": CalibratedClassifierCV(
            LinearSVC(random_state=RANDOM_STATE, max_iter=5000), cv=3
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(64,), max_iter=150, random_state=RANDOM_STATE,
            early_stopping=True,
        ),
    }


def latency_ms_per_sample(model, X, n_repeats=3):
    n = X.shape[0]
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model.predict(X)
        times.append((time.perf_counter() - t0) / n * 1000.0)
    return float(np.mean(times))


# ---------------------------------------------------------- core routine ---
def run_task(task_name, train, test, label_col, extractor_mode="combined",
             save_models=True, fig_prefix=None, extra_test=None, extra_name=None):
    log(f"\n{'='*70}\n TASK: {task_name}  (label={label_col}, features={extractor_mode})\n{'='*70}")
    le = LabelEncoder()
    y_train = le.fit_transform(train[label_col])
    y_test = le.transform(test[label_col])
    classes = le.classes_
    n_classes = len(classes)

    extractor = WAFFeatureExtractor(max_features=3000, mode=extractor_mode)
    t0 = time.perf_counter()
    X_train = extractor.fit_transform(train["payload"])
    X_test = extractor.transform(test["payload"])
    log(f"Feature extraction: {X_train.shape[1]} dims "
        f"(fit in {time.perf_counter()-t0:.2f}s)")

    results = []
    proba_store = {}
    for name, model in build_models().items():
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

        acc = accuracy_score(y_test, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )
        prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        roc_auc = np.nan
        if y_proba is not None:
            try:
                if n_classes == 2:
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])
                else:
                    yb = label_binarize(y_test, classes=range(n_classes))
                    roc_auc = roc_auc_score(yb, y_proba, average="macro", multi_class="ovr")
            except Exception:
                roc_auc = np.nan

        lat = latency_ms_per_sample(model, X_test)
        buf = io.BytesIO()
        joblib.dump(model, buf)
        model_size_kb = buf.tell() / 1024.0

        row = dict(
            task=task_name, model=name, accuracy=acc,
            precision_macro=prec_m, recall_macro=rec_m, f1_macro=f1_m,
            precision_weighted=prec_w, recall_weighted=rec_w, f1_weighted=f1_w,
            roc_auc=roc_auc, fit_time_s=fit_time,
            inference_latency_ms_per_sample=lat, model_size_kb=model_size_kb,
        )
        results.append(row)
        proba_store[name] = (y_pred, y_proba)
        log(f"  {name:18s} acc={acc:.4f}  f1_macro={f1_m:.4f}  "
            f"roc_auc={roc_auc if np.isnan(roc_auc) else round(roc_auc,4)}  "
            f"fit={fit_time:.2f}s  latency={lat:.4f}ms/sample")

        if save_models:
            joblib.dump(model, MODELDIR / f"{task_name}_{name}.joblib")

        # full classification report -> json
        rep = classification_report(y_test, y_pred, target_names=[str(c) for c in classes],
                                     output_dict=True, zero_division=0)
        with open(METRICDIR / f"classification_report_{task_name}_{name}.json", "w") as f:
            json.dump(rep, f, indent=2)

    if save_models:
        joblib.dump(extractor, MODELDIR / f"{task_name}_feature_extractor.joblib")
        joblib.dump(le, MODELDIR / f"{task_name}_label_encoder.joblib")

    res_df = pd.DataFrame(results).sort_values("f1_macro", ascending=False)
    res_df.to_csv(METRICDIR / f"results_{task_name}.csv", index=False)

    best_name = res_df.iloc[0]["model"]
    y_pred_best, y_proba_best = proba_store[best_name]

    if fig_prefix:
        _plot_model_comparison(res_df, f"{fig_prefix}_model_comparison.png",
                                title=f"{task_name}: model comparison")
        _plot_confusion(y_test, y_pred_best, classes,
                         f"{fig_prefix}_confusion_{best_name}.png",
                         title=f"{task_name}: confusion matrix ({best_name})")
        if n_classes == 2:
            _plot_roc_curves(y_test, proba_store, f"{fig_prefix}_roc_curves.png",
                              title=f"{task_name}: ROC curves")
        _plot_latency(res_df, f"{fig_prefix}_latency.png",
                       title=f"{task_name}: inference latency")

    # optional OOD evaluation using the trained models + fitted extractor
    ood_df = None
    if extra_test is not None:
        ood_df = _evaluate_ood(task_name, extra_name, extra_test, label_col,
                                extractor, le, build_models_dict_names=res_df["model"].tolist())

    return res_df, extractor, le, best_name


def _evaluate_ood(task_name, extra_name, extra_df, label_col, extractor, le, build_models_dict_names):
    log(f"\n--- OOD evaluation on '{extra_name}' (models trained for {task_name}) ---")
    y_true = le.transform(extra_df[label_col])
    X = extractor.transform(extra_df["payload"])
    rows = []
    for name in build_models_dict_names:
        model = joblib.load(MODELDIR / f"{task_name}_{name}.joblib")
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        rows.append(dict(task=task_name, eval_set=extra_name, model=name,
                          accuracy=acc, precision_macro=prec_m,
                          recall_macro=rec_m, f1_macro=f1_m))
        log(f"  {name:18s} OOD acc={acc:.4f}  f1_macro={f1_m:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(METRICDIR / f"results_{task_name}_{extra_name}.csv", index=False)
    return df


# ------------------------------------------------------------------ plots --
def _plot_model_comparison(res_df, fname, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    order = res_df.sort_values("f1_macro", ascending=True)
    ax.barh(order["model"], order["f1_macro"], color=PALETTE[0])
    for i, v in enumerate(order["f1_macro"]):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9, color="#0b0b0b")
    ax.set_xlabel("Macro F1-score (test set)")
    ax.set_xlim(0, 1.05)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(y_true, y_pred, classes, fname, title):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cmap = sns.light_palette(SEQ_BLUES[3], as_cmap=True)
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap=cmap, cbar=True,
                xticklabels=classes, yticklabels=classes, ax=ax,
                linewidths=0.5, linecolor="white", annot_kws={"fontsize": 10})
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.setp(ax.get_yticklabels(), rotation=0, va="center")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_roc_curves(y_test, proba_store, fname, title):
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    for i, (name, (_, proba)) in enumerate(proba_store.items()):
        if proba is None:
            continue
        fpr, tpr, _ = roc_curve(y_test, proba[:, 1])
        auc = roc_auc_score(y_test, proba[:, 1])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                color=PALETTE[i % len(PALETTE)], linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="#898781", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches="tight")
    plt.close(fig)


def _plot_latency(res_df, fname, title):
    order = res_df.sort_values("inference_latency_ms_per_sample", ascending=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(order["model"], order["inference_latency_ms_per_sample"], color=PALETTE[5])
    ax.set_xlabel("Inference latency (ms / sample, CPU, batch predict)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches="tight")
    plt.close(fig)


def plot_ood_gap(binary_test_df, ood_df, fname):
    m = binary_test_df.set_index("model")["accuracy"].rename("in_distribution_test")
    o = ood_df.set_index("model")["accuracy"].rename("ood_lexical_test")
    merged = pd.concat([m, o], axis=1).dropna().sort_values("in_distribution_test", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(merged))
    w = 0.35
    ax.bar(x - w/2, merged["in_distribution_test"], width=w, label="In-distribution test", color=PALETTE[0])
    ax.bar(x + w/2, merged["ood_lexical_test"], width=w, label="OOD lexical test", color=PALETTE[5])
    ax.set_xticks(x)
    ax.set_xticklabels(merged.index, rotation=20)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generalization gap: in-distribution vs. out-of-distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / fname, bbox_inches="tight")
    plt.close(fig)
    merged.to_csv(METRICDIR / "ood_generalization_gap.csv")
    return merged


# --------------------------------------------------------------- ablation --
def run_ablation(train, test):
    log(f"\n{'='*70}\n FEATURE ABLATION (binary task)\n{'='*70}")
    le = LabelEncoder()
    y_train = le.fit_transform(train["label"])
    y_test = le.transform(test["label"])

    rows = []
    for mode in ["tfidf", "engineered", "combined"]:
        extractor = WAFFeatureExtractor(max_features=3000, mode=mode)
        X_train = extractor.fit_transform(train["payload"])
        X_test = extractor.transform(test["payload"])
        for model_name, model in [
            ("RandomForest", RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE)),
            ("XGBoost", XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                       eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)),
        ]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            _, _, f1_m, _ = precision_recall_fscore_support(y_test, y_pred, average="macro", zero_division=0)
            rows.append(dict(feature_mode=mode, model=model_name, n_features=X_train.shape[1],
                              accuracy=acc, f1_macro=f1_m))
            log(f"  mode={mode:10s} model={model_name:14s} dims={X_train.shape[1]:5d} "
                f"acc={acc:.4f} f1_macro={f1_m:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(METRICDIR / "feature_ablation.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    modes = ["tfidf", "engineered", "combined"]
    x = np.arange(len(modes))
    w = 0.35
    all_vals = []
    for i, model_name in enumerate(["RandomForest", "XGBoost"]):
        vals = [df[(df.feature_mode == m) & (df.model == model_name)]["f1_macro"].values[0] for m in modes]
        all_vals.extend(vals)
        bars = ax.bar(x + (i - 0.5) * w, vals, width=w, label=model_name, color=PALETTE[i])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.0003, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["TF-IDF only", "Engineered only", "Combined"])
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(min(all_vals) - 0.002, max(all_vals) + 0.003)
    ax.set_title("Feature ablation: binary classification")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGDIR / "ablation_features.png", bbox_inches="tight")
    plt.close(fig)
    return df


# -------------------------------------------------------- cross-validation --
def run_cross_validation(train, model_name="XGBoost"):
    log(f"\n{'='*70}\n 5-FOLD CROSS-VALIDATION on training set ({model_name}, binary task)\n{'='*70}")
    le = LabelEncoder()
    y = le.fit_transform(train["label"])
    extractor = WAFFeatureExtractor(max_features=3000, mode="combined")
    X = extractor.fit_transform(train["payload"])
    model = build_models()[model_name]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=skf, scoring="f1_macro", n_jobs=1)
    log(f"  Fold F1-macro scores: {np.round(scores, 4).tolist()}")
    log(f"  Mean: {scores.mean():.4f}  Std: {scores.std():.4f}")
    pd.DataFrame({"fold": range(1, 6), "f1_macro": scores}).to_csv(
        METRICDIR / "cross_validation.csv", index=False)
    return scores


# ---------------------------------------------------- feature importance ---
def plot_feature_importance(task_name, model_name, extractor, top_n=20):
    model = joblib.load(MODELDIR / f"{task_name}_{model_name}.joblib")
    names = np.array(extractor.feature_names())
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(names[idx], importances[idx], color=PALETTE[4])
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} features — {model_name} ({task_name})")
    fig.tight_layout()
    fig.savefig(FIGDIR / f"{task_name}_feature_importance_{model_name}.png", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- main --
def main():
    log_path = LOGDIR / "run_experiments.log"
    with open(log_path, "w") as logfile:
        orig_stdout = sys.stdout
        sys.stdout = Tee(orig_stdout, logfile)
        try:
            t_start = time.perf_counter()
            train, test, lexical = load_data()
            dataset_stats(train, test, lexical)

            # --- Task A: binary classification + OOD generalization ---
            bin_res, bin_extractor, bin_le, bin_best = run_task(
                task_name="binary", train=train, test=test, label_col="label",
                extractor_mode="combined", fig_prefix="binary",
                extra_test=lexical, extra_name="lexical_ood",
            )
            ood_df = pd.read_csv(METRICDIR / "results_binary_lexical_ood.csv")
            plot_ood_gap(bin_res, ood_df, "ood_generalization_gap.png")

            # --- Task B: multi-class attack-type classification ---
            multi_res, multi_extractor, multi_le, multi_best = run_task(
                task_name="multiclass", train=train, test=test, label_col="attack_type",
                extractor_mode="combined", fig_prefix="multiclass",
            )

            # --- Feature importance for best tree model on binary task ---
            tree_model = "XGBoost" if "XGBoost" in bin_res["model"].values else bin_best
            plot_feature_importance("binary", tree_model, bin_extractor)

            # --- Feature ablation ---
            run_ablation(train, test)

            # --- Cross-validation of best model ---
            run_cross_validation(train, model_name="XGBoost")

            elapsed = time.perf_counter() - t_start
            log(f"\nAll experiments completed in {elapsed/60:.2f} minutes.")
            log(f"Best binary model: {bin_best}")
            log(f"Best multiclass model: {multi_best}")
        finally:
            sys.stdout = orig_stdout


if __name__ == "__main__":
    main()
