"""Feature engineering for WAF payload classification.

Combines character n-gram TF-IDF with hand-crafted lexical features that
capture structural properties known to correlate with injection attacks
(keyword usage, special-character density, entropy, bracket balance, ...).
"""
import re
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

SQL_KEYWORDS = [
    "select", "union", "where", "from", "and", "or", "order by", "group by",
    "insert", "update", "delete", "drop", "alter", "exec", "execute",
    "union all", "sleep", "waitfor", "benchmark", "information_schema",
    "having", "cast", "declare",
]
XSS_KEYWORDS = [
    "<script", "javascript:", "onerror", "onload", "onclick", "alert(",
    "document.cookie", "<img", "<svg", "eval(", "iframe", "</script>",
]
CMD_KEYWORDS = [
    ";", "|", "&&", "$(", "`", "wget", "curl", "nc ", "/bin/", "cmd.exe",
    "powershell", "chmod",
]
PATH_TRAVERSAL_PATTERNS = ["../", "..\\", "%2e%2e", "/etc/passwd", "boot.ini"]


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return float(-sum(p * np.log2(p) for p in probs))


def _count_hits(text: str, needles) -> int:
    t = text.lower()
    return sum(1 for n in needles if n in t)


def engineered_features(payloads: pd.Series) -> pd.DataFrame:
    """Hand-crafted lexical/structural features, all non-negative."""
    s = payloads.astype(str)
    feats = pd.DataFrame(index=s.index)
    feats["length"] = s.str.len()
    feats["special_char_ratio"] = s.apply(
        lambda x: len(re.findall(r"[^a-zA-Z0-9\s]", x)) / len(x) if len(x) else 0.0
    )
    feats["digit_ratio"] = s.apply(
        lambda x: len(re.findall(r"\d", x)) / len(x) if len(x) else 0.0
    )
    feats["upper_ratio"] = s.apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) else 0.0
    )
    feats["unique_char_ratio"] = s.apply(
        lambda x: len(set(x)) / len(x) if len(x) else 0.0
    )
    feats["entropy"] = s.apply(_shannon_entropy)
    feats["sql_keyword_count"] = s.apply(lambda x: _count_hits(x, SQL_KEYWORDS))
    feats["xss_keyword_count"] = s.apply(lambda x: _count_hits(x, XSS_KEYWORDS))
    feats["cmd_keyword_count"] = s.apply(lambda x: _count_hits(x, CMD_KEYWORDS))
    feats["path_traversal_hits"] = s.apply(lambda x: _count_hits(x, PATH_TRAVERSAL_PATTERNS))
    feats["parentheses_balance"] = s.apply(lambda x: abs(x.count("(") - x.count(")")))
    feats["quote_count"] = s.apply(lambda x: x.count("'") + x.count('"'))
    feats["space_ratio"] = s.apply(
        lambda x: x.count(" ") / len(x) if len(x) else 0.0
    )
    return feats


class WAFFeatureExtractor:
    """Fits a char n-gram TF-IDF vectorizer and concatenates engineered features."""

    def __init__(self, max_features=3000, ngram_range=(1, 4), mode="combined"):
        assert mode in ("tfidf", "engineered", "combined")
        self.mode = mode
        self.tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=ngram_range,
            max_features=max_features,
            strip_accents="unicode",
        )
        self.eng_columns_ = None

    def fit_transform(self, payloads: pd.Series):
        parts = []
        if self.mode in ("tfidf", "combined"):
            X_tfidf = self.tfidf.fit_transform(payloads.astype(str))
            parts.append(X_tfidf)
        if self.mode in ("engineered", "combined"):
            eng = engineered_features(payloads)
            self.eng_columns_ = eng.columns.tolist()
            parts.append(sparse.csr_matrix(eng.values.astype(np.float64)))
        return sparse.hstack(parts, format="csr") if len(parts) > 1 else parts[0]

    def transform(self, payloads: pd.Series):
        parts = []
        if self.mode in ("tfidf", "combined"):
            parts.append(self.tfidf.transform(payloads.astype(str)))
        if self.mode in ("engineered", "combined"):
            eng = engineered_features(payloads)
            parts.append(sparse.csr_matrix(eng.values.astype(np.float64)))
        return sparse.hstack(parts, format="csr") if len(parts) > 1 else parts[0]

    def feature_names(self):
        names = []
        if self.mode in ("tfidf", "combined"):
            names.extend([f"tfidf_{t}" for t in self.tfidf.get_feature_names_out()])
        if self.mode in ("engineered", "combined"):
            names.extend(self.eng_columns_)
        return names
