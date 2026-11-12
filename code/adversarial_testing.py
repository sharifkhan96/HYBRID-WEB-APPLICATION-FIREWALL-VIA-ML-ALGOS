"""
Black-box character-level adversarial evasion testing.

Addresses the "we did not evaluate adversarially crafted payloads" limitation.
Applies WAF-evasion-style string transforms to the malicious (anom) payloads
in the in-distribution test set and the OOD lexical test set, then measures
each model's evasion rate: of the payloads a model correctly flagged as
malicious *before* perturbation, what fraction does it now miss?

No retraining is performed anywhere -- this measures robustness of the
already-trained models (the same ones from run_experiments.py) to
adversarial input, a distinct robustness axis from the natural distribution
shift studied in the OOD experiment.

Transforms (applied independently, plus one "combined" transform that
chains all of them -- the strongest attack tested):
  - case_randomization : randomly re-cases each alphabetic character
  - whitespace_insertion: inserts SQL inline comments / extra whitespace
                          at token boundaries
  - url_encoding        : percent-encodes a random subset of characters
  - comment_injection   : inserts /**/ between adjacent word characters
                          (classic SQLi WAF-bypass technique)
  - combined            : case_randomization -> whitespace_insertion ->
                          url_encoding, applied in sequence

Usage: python3 adversarial_testing.py
"""
import random
import time
import urllib.parse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import WAFFeatureExtractor  # noqa: F401 (needed for unpickling)

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT.parent / "archive"
RESULTS = ROOT / "results"
METRICDIR = RESULTS / "metrics"
MODELDIR = RESULTS / "models"
FIGDIR = RESULTS / "figures"
LOGDIR = ROOT / "logs"

MODEL_NAMES = ["LogisticRegression", "MultinomialNB", "LinearSVM",
               "RandomForest", "XGBoost", "MLP"]

random.seed(RANDOM_STATE)
rng = np.random.default_rng(RANDOM_STATE)


# --------------------------------------------------------------- transforms
def case_randomization(s: str) -> str:
    return "".join(c.upper() if rng.random() < 0.5 else c.lower() for c in s)


def whitespace_insertion(s: str) -> str:
    # SQL treats /**/ as whitespace; insert it at existing space boundaries
    # and between adjacent word characters at a low random rate.
    out = []
    for i, c in enumerate(s):
        out.append(c)
        if c == " ":
            out.append("/**/" if rng.random() < 0.5 else "  ")
    return "".join(out)


def url_encoding(s: str, rate: float = 0.3) -> str:
    out = []
    for c in s:
        if c.isalnum() and rng.random() < rate:
            out.append(f"%{ord(c):02X}")
        else:
            out.append(c)
    return "".join(out)


def comment_injection(s: str) -> str:
    out = []
    chars = list(s)
    for i, c in enumerate(chars):
        out.append(c)
        if (i < len(chars) - 1 and c.isalpha() and chars[i + 1].isalpha()
                and rng.random() < 0.15):
            out.append("/**/")
    return "".join(out)


def combined(s: str) -> str:
    return url_encoding(whitespace_insertion(case_randomization(s)), rate=0.15)


# Realistic benign filler drawn from the actual training corpus (addresses,
# emails), used for a padding/dilution attack: a well-known technique for
# evading ratio-based features (special_char_ratio, digit_ratio, entropy)
# and diluting a short malicious payload's share of the TF-IDF vector by
# surrounding it with a much longer benign-looking string.
_PADDING_POOL = [
    "belmondo@artesania.org.ci", "calle fomento, 197, 3b",
    "c/ santo domingo 13 10 b", "vilcu-verschuer@eddydaniele.kz",
    "darbanville@chicasdiabolo.tr", "avenida de la constitucion 45",
    "contact us for more information about our services",
    "thank you for your recent order and continued business",
]


def padding_dilution(s: str) -> str:
    filler = " ".join(rng.choice(_PADDING_POOL, size=6, replace=True))
    return f"{filler} {s} {filler}"


TRANSFORMS = {
    "case_randomization": case_randomization,
    "whitespace_insertion": whitespace_insertion,
    "url_encoding": url_encoding,
    "comment_injection": comment_injection,
    "padding_dilution": padding_dilution,
    "combined": combined,
}


def load_split(name):
    df = pd.read_csv(DATA_DIR / name)
    df = df.dropna(subset=["payload", "label"]).reset_index(drop=True)
    df["payload"] = df["payload"].astype(str)
    return df


def evasion_rate(y_true_anom, y_pred_before, y_pred_after):
    """Of the samples that were true positives before perturbation, what
    fraction are no longer predicted positive after perturbation?"""
    was_caught = (y_true_anom == 1) & (y_pred_before == 1)
    if was_caught.sum() == 0:
        return np.nan, 0
    now_missed = was_caught & (y_pred_after == 0)
    return now_missed.sum() / was_caught.sum(), int(was_caught.sum())


def main():
    log_path = LOGDIR / "adversarial_testing.log"
    lines = []

    def log(msg=""):
        print(msg)
        lines.append(str(msg))

    t_start = time.perf_counter()

    extractor = joblib.load(MODELDIR / "binary_feature_extractor.joblib")
    label_encoder = joblib.load(MODELDIR / "binary_label_encoder.joblib")
    # LabelEncoder sorts alphabetically ('anom' < 'norm'), so the encoded
    # value meaning "anomalous" is NOT necessarily 1 -- look it up explicitly
    # and remap every prediction to the y_true convention (1 = anomalous).
    anom_idx = list(label_encoder.classes_).index("anom")
    models = {name: joblib.load(MODELDIR / f"binary_{name}.joblib") for name in MODEL_NAMES}

    splits = {
        "in_distribution_test": load_split("payload_test.csv"),
        "ood_lexical_test": load_split("payload_test_lexical.csv"),
    }

    rows = []
    examples = []

    for split_name, df in splits.items():
        y_true = (df["label"].values == "anom").astype(int)
        mal_mask = y_true == 1
        log(f"\n{'='*70}\n{split_name}: {mal_mask.sum()} malicious payloads\n{'='*70}")

        payloads = df["payload"].values
        X_before = extractor.transform(pd.Series(payloads))

        for model_name, model in models.items():
            y_pred_before = (model.predict(X_before) == anom_idx).astype(int)

            for transform_name, fn in TRANSFORMS.items():
                perturbed = np.array([
                    fn(p) if y_true[i] == 1 else p for i, p in enumerate(payloads)
                ])
                X_after = extractor.transform(pd.Series(perturbed))
                y_pred_after = (model.predict(X_after) == anom_idx).astype(int)

                rate, n_caught_before = evasion_rate(y_true, y_pred_before, y_pred_after)
                rows.append(dict(
                    split=split_name, model=model_name, transform=transform_name,
                    n_malicious=int(mal_mask.sum()), n_caught_before_perturbation=n_caught_before,
                    evasion_rate=rate,
                ))
                log(f"  {model_name:20s} {transform_name:22s} "
                    f"evasion_rate={rate:.3f}  (n_caught_before={n_caught_before})")

                if transform_name == "combined" and model_name == "XGBoost":
                    caught_idx = np.where((y_true == 1) & (y_pred_before == 1) & (y_pred_after == 0))[0]
                    for i in caught_idx[:5]:
                        examples.append(dict(
                            split=split_name, original=payloads[i], perturbed=perturbed[i],
                        ))

    result_df = pd.DataFrame(rows)
    result_df.to_csv(METRICDIR / "adversarial_evasion.csv", index=False)

    examples_df = pd.DataFrame(examples)
    examples_df.to_csv(METRICDIR / "adversarial_examples.csv", index=False)

    log(f"\nTotal elapsed: {(time.perf_counter()-t_start)/60:.1f} min")
    log("\n=== Mean evasion rate per transform (across models, in-distribution) ===")
    idd = result_df[result_df.split == "in_distribution_test"]
    log(idd.groupby("transform")["evasion_rate"].mean().sort_values(ascending=False).to_string())
    log("\n=== Mean evasion rate per model (across transforms, in-distribution) ===")
    log(idd.groupby("model")["evasion_rate"].mean().sort_values(ascending=False).to_string())

    with open(log_path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
