"""
Feature Preparation for Fairness-Accuracy Tradeoff Benchmark
=============================================================
Handles:
  1. Dropping degenerate columns (single unique value after missingness)
  2. Defining outcome-specific leak lists
  3. Categorizing features (numeric vs categorical)
  4. SES quintile derivation (per-fold, leak-free)
  5. Preprocessing pipeline construction

Usage:
    from prepare_features import FeatureConfig, get_features, derive_ses, mk_prep
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from dataclasses import dataclass, field
from typing import Dict, List, Set

# ─────────────────────────────────────────────────────────────────────────────
# Columns to drop (degenerate or redundant)
# ─────────────────────────────────────────────────────────────────────────────
DROP_COLS = [
    # Degenerate: only 1 unique non-null value
    "partner_in_hh",    # 31.8% missing, nunique=1 → no information
    "dep_pessimism",    # 17.8% missing, nunique=1 → no information

    # Redundant with max_grip_strength (correlation ~1.0)
    "maxgrip",          # keep max_grip_strength instead
]

# ─────────────────────────────────────────────────────────────────────────────
# ID and metadata columns (never used as features)
# ─────────────────────────────────────────────────────────────────────────────
ID_COLS = ["mergeid"]

# ─────────────────────────────────────────────────────────────────────────────
# Audit/protected attributes (excluded from features, used for fairness eval)
# ─────────────────────────────────────────────────────────────────────────────
AUDIT_ATTRS = ["country_id", "female"]
SES_COLS = ["ses_q", "ses_bin", "hh_income"]  # derived during CV

# ─────────────────────────────────────────────────────────────────────────────
# Outcome columns
# ─────────────────────────────────────────────────────────────────────────────
OUTCOMES = ["SRH_poor", "EUROD_dep", "Multimorbid", "Polypharmacy"]

# ─────────────────────────────────────────────────────────────────────────────
# LEAK LISTS — features that directly construct or are identical to each outcome
# ─────────────────────────────────────────────────────────────────────────────
# Universal: all outcome columns excluded from all predictions
UNIVERSAL_LEAK = set(OUTCOMES)

# SRH_poor = ph003_ >= 4 (fair/poor self-rated health)
# sphus and health_self_raw are the same underlying variable
SRH_LEAK = {
    "sphus",              # US-label SRH (identical to ph003_ mapping)
    "health_self_raw",    # Raw SRH scale (direct source of outcome)
}

# EUROD_dep = eurodcat == 1 (EURO-D >= 4)
# eurod is the sum of all dep_* items; eurodcat is the binary threshold
EUROD_LEAK = {
    "eurod",              # Continuous EURO-D score (outcome threshold applied to this)
    "eurodcat",           # Binary EURO-D caseness (THIS IS the outcome)
    "dep_sadness",        # EURO-D item 1
    "dep_tearfulness",    # EURO-D item 2
    "dep_guilt",          # EURO-D item 3
    "dep_interest_loss",  # EURO-D item 4
    "dep_irritability",   # EURO-D item 5
    "dep_appetite_loss",  # EURO-D item 6
    "dep_fatigue",        # EURO-D item 7
}
# Note: dep_pessimism already in DROP_COLS (degenerate)
# Note: dep_sleep, dep_suicidality, dep_concentration, dep_no_enjoyment
#       were dropped during cleaning (>50% missing)

# Multimorbid = chronic2w9 == 1 (2+ chronic conditions)
# chronicw9 is the count; chronic2w9 is the binary flag
# Individual conditions ARE the components that sum to chronicw9
MULTIMORBID_LEAK = {
    "chronicw9",          # Chronic condition count (outcome source)
    "chronic2w9",         # Binary 2+ indicator (THIS IS the outcome)
    # Individual conditions — summed to create chronicw9
    "heart",
    "hypertension",
    "cholesterol",
    "stroke",
    "diabetes",
    "lung_disease",
    "cancer",
    "ulcer",
    "parkinsons",
    "cataracts",
    "hip_fracture",
    "other_fracture",
    "osteoarthritis",
    "rheumatoid_arthritis",
    "kidney_disease",
    "emotional_disorder",
    "dementia",
}

# Polypharmacy = 5+ daily medication types
# n_drug_types is the count from which the threshold is derived
POLYPHARMACY_LEAK = {
    "n_drug_types",       # Number of drug types (outcome threshold applied to this)
}

LEAK_LISTS = {
    "SRH_poor": SRH_LEAK,
    "EUROD_dep": EUROD_LEAK,
    "Multimorbid": MULTIMORBID_LEAK,
    "Polypharmacy": POLYPHARMACY_LEAK,
}


def get_features(df, outcome: str) -> List[str]:
    """Get valid feature columns for a given outcome.

    Excludes: IDs, outcomes, audit/SES attributes, degenerate cols, leak cols.
    """
    exclude = set()
    exclude.update(ID_COLS)
    exclude.update(OUTCOMES)
    exclude.update(AUDIT_ATTRS)
    exclude.update(SES_COLS)
    exclude.update(DROP_COLS)
    exclude.update(UNIVERSAL_LEAK)
    exclude.update(LEAK_LISTS.get(outcome, set()))

    features = [c for c in df.columns
                if c not in exclude and not c.startswith("_")]
    return features


def col_types(df, cols):
    """Split columns into numeric (>10 unique) and categorical (<=10 unique)."""
    num = [c for c in cols if c in df.columns
           and df[c].dtype in ['int64', 'float64'] and df[c].nunique() > 10]
    cat = [c for c in cols if c in df.columns and c not in num]
    return num, cat


def mk_prep(df, cols, scaler="standard"):
    """Build sklearn preprocessing pipeline: impute + scale/encode."""
    num, cat = col_types(df, cols)
    sc = MinMaxScaler() if scaler == "minmax" else StandardScaler()
    transformers = []
    if num:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", sc),
        ])
        transformers.append(("num", num_pipe, num))
    if cat:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", cat_pipe, cat))
    return ColumnTransformer(transformers, remainder="drop")


def derive_ses(df, train_idx, force_recompute=False):
    """Compute within-country SES quintiles from training data only.

    Uses hh_income (multiply-imputed, 0% missing in SHARE W9).
    Quintile boundaries derived from train_idx to prevent leakage.

    Parameters
    ----------
    df : DataFrame with hh_income and country_id columns
    train_idx : array of training indices
    force_recompute : if True, always recompute

    Returns
    -------
    df with ses_q (1-5) and ses_bin (0/1) columns added
    """
    if ("ses_q" in df.columns and df["ses_q"].notna().sum() > len(df) * 0.5
            and not force_recompute):
        valid_ses = df["ses_q"].dropna()
        if valid_ses.min() >= 1 and valid_ses.max() <= 5:
            df["ses_bin"] = (df["ses_q"] == 5).astype(int)
            return df

    inc_col = "hh_income"
    cty_col = "country_id"
    train_set = set(train_idx)

    df["ses_q"] = np.nan

    for c in df[cty_col].unique():
        train_mask = df.index.isin(train_set) & (df[cty_col] == c)
        train_inc = df.loc[train_mask, inc_col].dropna()

        if len(train_inc) < 50:
            continue

        q20 = train_inc.quantile(0.2)
        q40 = train_inc.quantile(0.4)
        q60 = train_inc.quantile(0.6)
        q80 = train_inc.quantile(0.8)

        country_mask = df[cty_col] == c
        df.loc[country_mask, "ses_q"] = pd.cut(
            df.loc[country_mask, inc_col],
            bins=[-np.inf, q20, q40, q60, q80, np.inf],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)

    # Fill missing with within-country median or global fallback
    for c in df[cty_col].unique():
        mask = (df[cty_col] == c) & df["ses_q"].isna()
        if mask.sum() > 0:
            med = df.loc[(df[cty_col] == c) & df["ses_q"].notna(), "ses_q"].median()
            df.loc[mask, "ses_q"] = med if pd.notna(med) else 3

    df["ses_q"] = df["ses_q"].fillna(3).astype(int)
    df["ses_bin"] = (df["ses_q"] == 5).astype(int)  # Q5 = underprivileged
    return df


def prep_data(df, outcome, splits, features):
    """Prepare train/cal/test splits for one outcome.

    Returns dict with Xtr, ytr, Xv, yv, Xte, yte, feats, ses_q_te.
    """
    tr, val, te = splits["train"], splits["val"], splits["test"]

    # Filter features to those actually present
    feats = [f for f in features if f in df.columns]

    return {
        "Xtr": df.loc[tr],
        "ytr": df.loc[tr, outcome].values,
        "Xv": df.loc[val],
        "yv": df.loc[val, outcome].values,
        "Xte": df.loc[te],
        "yte": df.loc[te, outcome].values,
        "feats": feats,
        "ses_q_te": df.loc[te, "ses_q"].values if "ses_q" in df.columns else None,
        "ses_bin_te": df.loc[te, "ses_bin"].values if "ses_bin" in df.columns else None,
    }


def print_feature_summary(df):
    """Print summary of features per outcome and leak exclusions."""
    print("=" * 60)
    print("FEATURE PREPARATION SUMMARY")
    print("=" * 60)

    print(f"\nDataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Dropped degenerate: {DROP_COLS}")

    # Missingness in remaining features
    all_feats = get_features(df, "SRH_poor")  # use any outcome for base count
    miss = df[all_feats].isnull().mean()
    high_miss = miss[miss > 0.05].sort_values(ascending=False)
    if len(high_miss) > 0:
        print(f"\nFeatures with >5% missing (will be median/mode imputed):")
        for col, rate in high_miss.items():
            print(f"  {col}: {rate:.1%}")

    print(f"\nFeatures per outcome:")
    for out in OUTCOMES:
        feats = get_features(df, out)
        leak = LEAK_LISTS.get(out, set())
        print(f"  {out:15s}: {len(feats)} features "
              f"(excluded {len(leak)} leak vars)")

    print("=" * 60)


if __name__ == "__main__":
    df = pd.read_csv("share_w9_cleaned.csv.zip")
    print_feature_summary(df)

    # Verify no leaks
    for out in OUTCOMES:
        feats = get_features(df, out)
        leak = LEAK_LISTS.get(out, set())
        overlap = set(feats) & leak
        assert len(overlap) == 0, f"{out}: leak features in feature list: {overlap}"
    print("\nLeak check passed for all outcomes.")
