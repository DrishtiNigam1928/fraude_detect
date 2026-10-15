"""Single source of truth for feature engineering.

Imported by BOTH train_eval.py (training) and app.py (serving), and used inside the
saved sklearn Pipeline, so training and serving can never drift apart.
"""
import pandas as pd


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hour" not in df.columns:
        df["hour"] = df["step"] % 24  # PaySim: 1 step = 1 hour

    # ---- Features known BEFORE a payment is approved --------------------------
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 22, 23]).astype(int)
    df["amount_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
    df["orig_empty"] = (df["oldbalanceOrg"] == 0).astype(int)
    df["dest_empty"] = (df["oldbalanceDest"] == 0).astype(int)

    # ---- Post-transaction features (only exist after the money has moved) -----
    # Created only when the raw columns are present (training/ablation), so the
    # live API can score a transaction without them.
    if {"newbalanceOrig", "newbalanceDest"} <= set(df.columns):
        df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
        df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]
        df["is_zero_orig"] = ((df["oldbalanceOrg"] == 0) & (df["newbalanceOrig"] == 0)).astype(int)
        df["is_zero_dest"] = ((df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)).astype(int)
    return df


# Deployable feature set: nothing that is only known after the transaction.
PRE_TXN = dict(
    num=["amount", "oldbalanceOrg", "oldbalanceDest", "hour", "amount_ratio"],
    cat=["type"],
    bin=["is_night", "orig_empty", "dest_empty"],
)

# Original notebook feature set (uses post-transaction balances). Trained only to
# measure how much accuracy comes from information a live system would not have.
FULL = dict(
    num=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
         "hour", "balanceDiffOrig", "balanceDiffDest", "amount_ratio"],
    cat=["type"],
    bin=["is_night", "is_zero_orig", "is_zero_dest"],
)

FEATURE_SETS = {"pre_txn": PRE_TXN, "full": FULL}
TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
