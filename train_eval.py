
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from features import FEATURE_SETS, add_features

COST_FN, COST_FP = 100.0, 1.0  # a missed fraud costs ~100x a false alarm; adjust to your business


def time_split(df, train_q=0.70, val_q=0.85):
    a, b = df["step"].quantile([train_q, val_q])
    return df[df.step <= a], df[(df.step > a) & (df.step <= b)], df[df.step > b]


def make_pre(fs):
    prep = ColumnTransformer([
        ("num", StandardScaler(), fs["num"]),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), fs["cat"]),
        ("bin", "passthrough", fs["bin"]),
    ])
    return Pipeline([("fe", FunctionTransformer(add_features)), ("prep", prep)])


def best_threshold(y, p):
    """Threshold that minimises expected cost on VALIDATION data (never on test)."""
    ths = np.unique(np.quantile(p, np.linspace(0.5, 0.99999, 400)))
    costs = []
    for t in ths:
        pred = p >= t
        fn = ((~pred) & (y == 1)).sum()
        fp = (pred & (y == 0)).sum()
        costs.append(fn * COST_FN + fp * COST_FP)
    return float(ths[int(np.argmin(costs))])


def report(y, p, t):
    pred = (p >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        pr_auc=float(average_precision_score(y, p)),
        roc_auc=float(roc_auc_score(y, p)),
        threshold=float(t),
        precision=float(tp / max(tp + fp, 1)),
        recall=float(tp / max(tp + fn, 1)),
        tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn),
    )


def build_models(scale_pos_weight):
    return {
        "logreg": LogisticRegression(class_weight="balanced", max_iter=1000),
        "xgb": XGBClassifier(
            n_estimators=1000, learning_rate=0.05, max_depth=5, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5, scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", tree_method="hist", early_stopping_rounds=50,
            n_jobs=-1, random_state=42),
    }


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    tr, va, te = time_split(df)
    meta = dict(
        rows=int(len(df)), train_rows=int(len(tr)), val_rows=int(len(va)), test_rows=int(len(te)),
        overall_fraud_rate=float(df.isFraud.mean()),
        test_fraud_rate=float(te.isFraud.mean()), test_fraud_count=int(te.isFraud.sum()),
        test_step_range=[int(te.step.min()), int(te.step.max())],
        cost_fn=COST_FN, cost_fp=COST_FP, deployed_model="xgb/pre_txn",
        train_neg_frac=args.train_neg_frac,
    )
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | test fraud rate {te.isFraud.mean():.4%}")

    if args.train_neg_frac < 1.0:  # speed-up: downsample ONLY the training negatives
        neg = tr[tr.isFraud == 0].sample(frac=args.train_neg_frac, random_state=42)
        tr = pd.concat([neg, tr[tr.isFraud == 1]]).sample(frac=1, random_state=42)

    results, curves, deployed = {}, {}, None
    for fs_name, fs in FEATURE_SETS.items():
        pre = make_pre(fs)
        Xtr, Xva, Xte = pre.fit_transform(tr), pre.transform(va), pre.transform(te)
        spw = float((tr.isFraud == 0).sum() / max((tr.isFraud == 1).sum(), 1))

        for name, clf in build_models(spw).items():
            if name == "xgb":
                clf.fit(Xtr, tr.isFraud, eval_set=[(Xva, va.isFraud)], verbose=False)
            else:
                clf.fit(Xtr, tr.isFraud)
            t = best_threshold(va.isFraud.values, clf.predict_proba(Xva)[:, 1])
            p_te = clf.predict_proba(Xte)[:, 1]
            key = f"{name}/{fs_name}"
            results[key] = report(te.isFraud.values, p_te, t)
            pr, rc, _ = precision_recall_curve(te.isFraud, p_te)
            curves[key] = (rc, pr)
            print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in results[key].items()})
            if key == meta["deployed_model"]:
                deployed = (Pipeline([("pre", pre), ("clf", clf)]), t)

    flagged = te.isFlaggedFraud == 1
    hit = (flagged & (te.isFraud == 1)).sum()
    bank_rule = dict(recall=float(hit / max(te.isFraud.sum(), 1)),
                     precision=float(hit / max(flagged.sum(), 1)), flagged=int(flagged.sum()))

    (out / "metrics.json").write_text(json.dumps(
        dict(meta=meta, models=results, bank_rule_baseline=bank_rule), indent=2))

    plt.figure(figsize=(7, 5))
    for k, (rc, pr) in curves.items():
        plt.plot(rc, pr, label=f"{k} (AP={results[k]['pr_auc']:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-recall on time-based test set")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "pr_curves.png", dpi=130); plt.close()

    pipe, threshold = deployed
    joblib.dump(dict(pipeline=pipe, threshold=threshold, meta=meta), out / "model_v2.pkl")
    print(f"saved model_v2.pkl, metrics.json, pr_curves.png to {out.resolve()}")

    try:  # SHAP global importance for the deployed model
        import shap
        sample = te.sample(min(5000, len(te)), random_state=1)
        Xs = pipe.named_steps["pre"].transform(sample)
        Xs = Xs.toarray() if hasattr(Xs, "toarray") else Xs
        names = pipe.named_steps["pre"].named_steps["prep"].get_feature_names_out()
        sv = shap.TreeExplainer(pipe.named_steps["clf"]).shap_values(Xs)
        shap.summary_plot(sv, Xs, feature_names=names, plot_type="bar", show=False)
        plt.tight_layout(); plt.savefig(out / "shap_importance.png", dpi=130); plt.close()
    except Exception as exc:  # shap is optional
        print("SHAP skipped:", exc)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Fraud.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--train-neg-frac", type=float, default=1.0,
                    help="fraction of NON-fraud training rows to keep (test set is never downsampled)")
    main(ap.parse_args())
