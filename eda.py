"""Exploratory analysis that JUSTIFIES the feature list in features.py.

Run this BEFORE (or alongside) train_eval.py:
    python eda.py --csv Fraud.csv

Unlike a notebook that eyeballs the whole dataset, this computes every number
and chart on the TRAINING split only (same time_split as train_eval.py) so
none of this "EDA" is secretly peeking at the test set — a common but real
form of leakage in EDA-then-train workflows.

Outputs (in ./eda_output/):
    fraud_by_type.png       - fraud rate per transaction type
    fraud_by_hour.png       - fraud rate + volume by hour of day
    amount_distribution.png - amount, fraud vs legit (log scale)
    eda_summary.json        - every number quoted in PROJECT_NOTES.md
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from train_eval import time_split  # reuse the EXACT same split as training


def main(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df["hour"] = df["step"] % 24
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)

    tr, _, _ = time_split(df)  # EDA only looks at training data, never val/test
    summary = {"rows_analyzed": int(len(tr)), "note": "computed on the TRAIN split only"}

    # ---- 1. Fraud rate by transaction type ----------------------------------
    by_type = tr.groupby("type")["isFraud"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    summary["fraud_rate_by_type"] = by_type["mean"].to_dict()
    fig, ax = plt.subplots(figsize=(7, 4))
    by_type["mean"].mul(100).plot(kind="bar", ax=ax, color="#c0392b")
    ax.set_ylabel("Fraud rate (%)"); ax.set_title("Fraud rate by transaction type (train split)")
    plt.tight_layout(); plt.savefig(out / "fraud_by_type.png", dpi=130); plt.close()
    only_transfer_cashout = tr[tr.type.isin(["TRANSFER", "CASH_OUT"])].isFraud.sum() == tr.isFraud.sum()
    summary["fraud_confined_to_transfer_cashout"] = bool(only_transfer_cashout)
    print("Fraud rate by type:\n", (by_type["mean"] * 100).round(3), "\n")
    print(f"-> Every fraud is TRANSFER/CASH_OUT: {only_transfer_cashout}")
    print("   JUSTIFIES: keeping `type` in the feature set (features.py -> PRE_TXN['cat'])\n")

    # ---- 2. Fraud rate by hour / night vs day --------------------------------
    by_hour = tr.groupby("hour")["isFraud"].agg(["mean", "count"])
    summary["fraud_rate_by_hour"] = (by_hour["mean"] * 100).round(3).to_dict()
    night_rate = tr[tr.is_night == 1].isFraud.mean()
    day_rate = tr[tr.is_night == 0].isFraud.mean()
    summary["night_fraud_rate_pct"] = round(night_rate * 100, 3)
    summary["day_fraud_rate_pct"] = round(day_rate * 100, 3)
    summary["night_vs_day_ratio"] = round(night_rate / max(day_rate, 1e-9), 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(by_hour.index, by_hour["mean"] * 100, color="#e67e22")
    axes[0].set(title="Fraud rate by hour", xlabel="Hour", ylabel="Fraud rate (%)")
    axes[1].bar(by_hour.index, by_hour["count"], color="#2980b9")
    axes[1].set(title="Transaction volume by hour", xlabel="Hour", ylabel="Count")
    plt.tight_layout(); plt.savefig(out / "fraud_by_hour.png", dpi=130); plt.close()
    print(f"Night fraud rate: {night_rate:.4%}  |  Day fraud rate: {day_rate:.4%}  "
          f"|  Ratio: {summary['night_vs_day_ratio']}x")
    print("   JUSTIFIES: keeping `hour`/`is_night` (features.py -> PRE_TXN['num'], ['bin'])\n")

    # ---- 3. Amount distribution ----------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    tr[tr.isFraud == 0].amount.clip(upper=tr.amount.quantile(0.99)).plot(
        kind="hist", bins=60, alpha=0.6, label="Legit", ax=ax, color="#2980b9", log=True)
    tr[tr.isFraud == 1].amount.clip(upper=tr.amount.quantile(0.99)).plot(
        kind="hist", bins=60, alpha=0.6, label="Fraud", ax=ax, color="#c0392b", log=True)
    ax.set_title("Amount distribution, fraud vs legit (99th pct clipped, log y)")
    ax.set_xlabel("Amount"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "amount_distribution.png", dpi=130); plt.close()
    summary["median_amount_fraud"] = float(tr[tr.isFraud == 1].amount.median())
    summary["median_amount_legit"] = float(tr[tr.isFraud == 0].amount.median())
    print(f"Median fraud amount: {summary['median_amount_fraud']:,.0f}  "
          f"vs legit: {summary['median_amount_legit']:,.0f}")
    print("   JUSTIFIES: keeping `amount` and `amount_ratio` as numeric features\n")

    # ---- 4. Zero-balance patterns (pre-transaction only — no leakage) -------
    orig_empty_fraud_rate = tr[tr.oldbalanceOrg == 0].isFraud.mean()
    dest_empty_fraud_rate = tr[tr.oldbalanceDest == 0].isFraud.mean()
    summary["orig_empty_fraud_rate_pct"] = round(orig_empty_fraud_rate * 100, 3)
    summary["dest_empty_fraud_rate_pct"] = round(dest_empty_fraud_rate * 100, 3)
    print(f"Fraud rate when SENDER balance starts at 0: {orig_empty_fraud_rate:.4%}")
    print(f"Fraud rate when RECEIVER balance starts at 0: {dest_empty_fraud_rate:.4%}")
    print("   JUSTIFIES: `orig_empty`/`dest_empty` flags (features.py -> PRE_TXN['bin'])\n")

    # ---- 5. What was EXCLUDED, and why ---------------------------------------
    summary["excluded_features"] = {
        "newbalanceOrig": "only known AFTER the transfer completes -- unusable at decision time",
        "newbalanceDest": "same reason",
        "balanceDiffOrig/Dest": "derived from the two features above -- inherits the leakage",
        "isFlaggedFraud": "the bank's OWN rule output, not a real feature -- used only as a baseline to beat",
        "nameOrig/nameDest": "unique per-account IDs -- can't generalise, and identify individuals",
    }

    (out / "eda_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"Saved charts + eda_summary.json to {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="Fraud.csv")
    ap.add_argument("--out", default="eda_output")
    main(ap.parse_args())
