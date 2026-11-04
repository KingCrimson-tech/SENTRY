"""Evaluation experiments for the report.

1. FoR-only model  -> FoR held-out and modern TTS   (generalisation gap)
2. FoR + aug + TTS -> grouped held-out split        (final clean model)
3. Leave-one-voice-out cross-validation on the 5 edge-tts voices
4. Channel study: clean model on speaker->mic audio (should fail)
                  channel specialist, leave-one-clip-out
5. Figures: confusion matrix, feature importance, score distributions
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import load_cache

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"
MODELS = ROOT / "models"

PARAMS = dict(
    objective="binary", metric="auc", learning_rate=0.06, num_leaves=63,
    min_child_samples=20, feature_fraction=0.85, bagging_fraction=0.9,
    bagging_freq=1, n_jobs=12, verbose=-1, seed=7,
)


def eer(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def fit(X, y, rounds=500, params=None):
    d = lgb.Dataset(X, label=y)
    return lgb.train(params or PARAMS, d, num_boost_round=rounds)


def report(name, model, X, y):
    s = model.predict(X)
    pred = (s >= 0.5).astype(int)
    both = len(np.unique(y)) > 1
    e = eer(y, s)[0] if both else None
    out = {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, s)) if both else None,
        "eer": e,
        "confusion": confusion_matrix(y, pred).tolist() if both else None,
        "score_real_mean": float(s[y == 0].mean()) if (y == 0).any() else None,
        "score_fake_mean": float(s[y == 1].mean()) if (y == 1).any() else None,
    }
    print(
        f"[{name}] n={out['n']} acc={out['accuracy']:.4f} auc={out['auc']} "
        f"eer={e if e is None else round(e, 4)} "
        f"mean_real={out['score_real_mean']} mean_fake={out['score_fake_mean']}"
    )
    return out, s


def main():
    FIG.mkdir(exist_ok=True)
    X, y, groups, names = load_cache()
    gs = np.array([str(g) for g in groups])
    is_tts = np.char.startswith(gs, "tts-")            # clean generated TTS
    is_for = ~is_tts                                   # FoR real/fake recordings

    Xf, yf, gf = X[is_for], y[is_for], gs[is_for]
    Xt, yt, gt = X[is_tts], y[is_tts], gs[is_tts]

    # Channel experiment lives outside the main cache: it is a side study, not
    # part of the deployed model.
    chan_path = ROOT / "cache" / "experiments" / "chan.npz"
    if chan_path.exists():
        cd = np.load(chan_path, allow_pickle=True)
        Xc, yc, gc = cd["X"], cd["y"], np.array([str(g) for g in cd["groups"]])
    else:
        Xc = yc = gc = None

    print(f"[eval] FoR={len(yf)} clean-TTS={len(yt)} "
          f"channel={0 if yc is None else len(yc)}")
    results = {}

    # ---- 1. FoR-only baseline, tested on FoR and on modern TTS ----------
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=7).split(Xf, yf, gf))
    m_for = fit(Xf[tr], yf[tr])
    results["for_only__for_test"], _ = report("FoR-only -> FoR test", m_for, Xf[te], yf[te])
    if len(Xt):
        results["for_only__tts_test"], _ = report("FoR-only -> modern TTS", m_for, Xt, yt)

    # ---- 2. final clean model: FoR + clean TTS --------------------------
    keep = is_for | is_tts
    Xk, yk, gk = X[keep], y[keep], gs[keep]
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=7).split(Xk, yk, gk))
    m_all = fit(Xk[tr], yk[tr])
    results["combined__holdout"], s = report(
        "FoR+TTS -> grouped holdout", m_all, Xk[te], yk[te]
    )

    # confusion matrix figure
    cm = np.array(results["combined__holdout"]["confusion"])
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["real", "fake"]); ax.set_yticks([0, 1], ["real", "fake"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Grouped hold-out")
    fig.tight_layout(); fig.savefig(FIG / "cm.png", dpi=160); plt.close(fig)

    # feature importance figure
    imp = m_all.feature_importance("gain")
    order = np.argsort(imp)[-20:]
    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.barh(np.array(names)[order], imp[order], color="#0ea5e9")
    ax.set_title("Top-20 features (gain)"); fig.tight_layout()
    fig.savefig(FIG / "importance.png", dpi=160); plt.close(fig)

    # score distributions
    fig, ax = plt.subplots(figsize=(5.4, 3))
    ax.hist(s[yk[te] == 0], bins=30, alpha=0.65, label="real", color="#16a34a", density=True)
    ax.hist(s[yk[te] == 1], bins=30, alpha=0.65, label="fake", color="#dc2626", density=True)
    ax.axvline(0.4, ls="--", c="#f59e0b"); ax.axvline(0.6, ls="--", c="#dc2626")
    ax.set_xlabel("spoof probability"); ax.legend(); ax.set_title("Held-out scores")
    fig.tight_layout(); fig.savefig(FIG / "scores.png", dpi=160); plt.close(fig)

    # ---- 3. leave-one-voice-out on clean TTS ---------------------------
    if len(Xt):
        rows = {}
        for v in sorted(set(gt)):
            tr = gk != v
            te = gt == v
            mm = fit(Xk[tr], yk[tr])
            out, _ = report(f"unseen voice {v}", mm, Xt[te], yt[te])
            rows[v] = out
        results["leave_one_voice_out"] = rows

    # ---- 4. channel study ----------------------------------------------
    if yc is not None and len(yc):
        # (a) the deployed clean model on speaker->mic audio
        deployed = lgb.Booster(model_file=str(MODELS / "sentry_lgbm.txt"))
        results["clean_model__channel"] = report(
            "deployed clean model -> speaker/mic", deployed, Xc, yc
        )[0]

        # (b) channel specialist, group-aware cross-validation
        cparams = dict(
            objective="binary", verbosity=-1, n_jobs=12, num_leaves=15,
            learning_rate=0.08, min_child_samples=8, feature_fraction=0.8,
            bagging_fraction=0.9, bagging_freq=1, seed=7,
        )
        folds = list(GroupKFold(n_splits=5).split(Xc, yc, gc))
        accs = []
        for tr, te in folds:
            mm = fit(Xc[tr], yc[tr], rounds=250, params=cparams)
            sc = mm.predict(Xc[te])
            accs.append(float(((sc >= 0.5) == yc[te]).mean()))
        results["channel_specialist__loro"] = {
            "n": int(len(yc)),
            "n_clips": int(len(set(gc))),
            "accuracy": float(np.mean(accs)),
            "fold_accuracies": accs,
            "note": "leave-one-clip-out; overfits speaker identity",
        }
        print(f"[channel specialist] clips={len(set(gc))} loro_acc={np.mean(accs):.4f}")

    (MODELS / "eval.json").write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {MODELS/'eval.json'} and figures in {FIG}")


if __name__ == "__main__":
    main()
