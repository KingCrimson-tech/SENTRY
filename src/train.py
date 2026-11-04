"""Train the SENTRY-style LightGBM detector."""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from .data import load_cache

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def eer(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thr = roc_curve(y, score)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2.0), float(thr[i])


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(MODELS / "sentry_lgbm.txt"))
    ap.add_argument("--meta", default=str(MODELS / "meta.json"))
    a = ap.parse_args()
    out_path, meta_path = Path(a.out), Path(a.meta)
    MODELS.mkdir(exist_ok=True)
    X, y, groups, names = load_cache()
    print(f"[train] X={X.shape} real={int((y == 0).sum())} fake={int((y == 1).sum())} "
          f"chan={int(np.char.startswith(np.array([str(g) for g in groups]), 'chan-').sum())} "
          f"tts={int(np.char.startswith(np.array([str(g) for g in groups]), 'tts-').sum())}")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    tr, te = next(gss.split(X, y, groups))
    # carve a validation slice out of train for early stopping
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=8)
    tr2, va = next(gss2.split(X[tr], y[tr], groups[tr]))
    tr2, va = tr[tr2], tr[va]

    params = dict(
        objective="binary",
        metric="auc",
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=20,
        feature_fraction=0.85,
        bagging_fraction=0.9,
        bagging_freq=1,
        n_jobs=12,
        verbose=-1,
        seed=7,
    )
    # up-weight 2026 neural-TTS clips: they are few but they are the demo target
    w = np.ones(len(y), dtype=np.float32)
    gs = np.array([str(g) for g in groups])
    w[np.char.startswith(gs, "tts-")] = 8.0
    w[np.char.startswith(gs, "chan-")] = 15.0
    dtr = lgb.Dataset(X[tr2], label=y[tr2], weight=w[tr2], feature_name=names)
    dva = lgb.Dataset(X[va], label=y[va], weight=w[va], reference=dtr, feature_name=names)
    model = lgb.train(
        params, dtr, num_boost_round=800, valid_sets=[dva],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(100)],
    )

    score = model.predict(X[te], num_iteration=model.best_iteration)
    pred = (score >= 0.5).astype(int)
    e, thr = eer(y[te], score)
    auc = roc_auc_score(y[te], score)
    acc = accuracy_score(y[te], pred)
    cm = confusion_matrix(y[te], pred)
    print(f"[eval] held-out groups: acc={acc:.4f} auc={auc:.4f} EER={e:.4f}")
    print("[eval] confusion [[TN FP],[FN TP]]:\n", cm)

    model.save_model(str(out_path))
    meta = {
        "n_features": len(names),
        "features": names,
        "thresholds": {"log": 0.40, "drop": 0.60},
        "metrics": {"accuracy": acc, "auc": float(auc), "eer": e, "eer_threshold": thr},
        "trained_on": "Fake-or-Real (norm) + optional local TTS augmentation",
        "paper": "SENTRY, arXiv:2610.24468",
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    imp = sorted(zip(names, model.feature_importance("gain")), key=lambda t: -t[1])[:15]
    print("[train] top features:", ", ".join(f"{k}" for k, _ in imp))
    print(f"[train] saved -> {out_path}")


if __name__ == "__main__":
    main()
