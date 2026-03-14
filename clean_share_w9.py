"""
SHARE Wave 9 Data Cleaning Pipeline
=====================================
Builds a clean, analysis-ready dataset from raw SHARE Wave 9 Stata modules.

Outcomes (4 binary, clinically-backed thresholds):
  1. SRH_poor:      Self-rated health fair/poor (ph003_ >= 4)
  2. EUROD_dep:     EURO-D depression caseness (eurodcat == 1, i.e. eurod >= 4)
  3. Multimorbid:   2+ chronic conditions (chronic2w9 == 1)
  4. Polypharmacy:  5+ daily medications (ph082_ == 1)

Protected attribute (SES):
  - Derived from SHARE multiply-imputed household income (thinc, 5 implicates averaged)
  - Quintiles computed per-fold during analysis (NOT here) to prevent leakage

Data sources:
  - gv_health.dta:       Validated generated health variables (EUROD, ADL, IADL, BMI, etc.)
  - gv_imputations.dta:  Multiply-imputed income/wealth (5 implicates, 0% missing)
  - gv_isced.dta:        Harmonized education (ISCED-97)
  - cv_r.dta:            Cover screen (age, gender, country)
  - dn.dta:              Demographics (marital status, migration, citizenship)
  - ph.dta:              Physical health (SRH, chronic conditions, mobility, ADL, polypharmacy)
  - mh.dta:              Mental health (EURO-D items, loneliness)
  - cf.dta:              Cognitive function (memory, verbal fluency, orientation)
  - br.dta:              Behavioral risks (smoking, drinking, physical activity, diet)
  - hc.dta:              Healthcare utilization (doctor visits, hospital, dentist, drugs)
  - ac.dta:              Activities (life satisfaction, CASP-12, social activities)
  - co.dta:              Consumption (making ends meet)
  - sp.dta:              Social participation (help given/received)
  - sn.dta:              Social networks (network size, satisfaction)
  - ep.dta:              Employment (job situation)
  - gv_big5.dta:         Big 5 personality traits
  - gs.dta:              Grip strength

Author: Auto-generated cleaning pipeline
Date: 2026-03-14
"""

import pandas as pd
import numpy as np
import pyreadstat
import zipfile
import os
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
RAW_DIR = "raw_data/wave9"
OUTPUT_FILE = "share_w9_cleaned.csv"
OUTPUT_ZIP = "share_w9_cleaned.csv.zip"
CLEANING_LOG = "cleaning_log.json"

# SHARE negative codes indicating missing/refused/don't know
SHARE_MISSING_CODES = {-1, -2, -3, -4, -5, -7, -9, -10, -11, -12, -13, -14, -15, -16}


def load_module(name, usecols=None):
    """Load a SHARE Wave 9 Stata module."""
    path = os.path.join(RAW_DIR, f"sharew9_rel9-0-0_{name}.dta")
    df, meta = pyreadstat.read_dta(path, usecols=usecols, apply_value_formats=False)
    return df, meta


def replace_share_missing(df, cols=None):
    """Replace SHARE negative missing codes with NaN."""
    if cols is None:
        cols = df.select_dtypes(include=[np.number]).columns
    for col in cols:
        if col in df.columns:
            df[col] = df[col].where(~df[col].isin(SHARE_MISSING_CODES), np.nan)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load base population from cover screen
# ─────────────────────────────────────────────────────────────────────────────
def load_base_population():
    """Load cover screen and filter to interviewed respondents."""
    print("Step 1: Loading base population from cv_r...")
    df, meta = load_module("cv_r", usecols=[
        "mergeid", "gender", "yrbirth", "age_int", "country",
        "interview", "hhsize", "partnerinhh"
    ])

    # Keep only respondents who completed interview in wave 9
    n_total = len(df)
    df = df[df["interview"] == 1].copy()
    n_interviewed = len(df)

    # Clean gender (1=Male, 2=Female)
    df["female"] = (df["gender"] == 2).astype(int)

    # Age at interview
    df["age"] = df["age_int"]
    df.loc[df["age"].isin(SHARE_MISSING_CODES), "age"] = np.nan

    # Country
    df["country_id"] = df["country"]

    # Partner in household
    df["partner_in_hh"] = df["partnerinhh"].map({1: 1, 5: 0})

    # Household size
    df["hhsize"] = df["hhsize"].where(~df["hhsize"].isin(SHARE_MISSING_CODES), np.nan)

    log = {
        "total_in_cv_r": n_total,
        "interviewed_w9": n_interviewed,
    }

    keep_cols = ["mergeid", "female", "age", "country_id", "partner_in_hh", "hhsize"]
    return df[keep_cols].copy(), log


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load generated health variables (validated constructs)
# ─────────────────────────────────────────────────────────────────────────────
def load_gv_health():
    """Load SHARE's official generated health variables."""
    print("Step 2: Loading generated health variables (gv_health)...")
    df, meta = load_module("gv_health", usecols=[
        "mergeid",
        # Outcomes
        "sphus",          # Self-perceived health (US version, 1-5)
        "eurod",          # EURO-D score (0-12)
        "eurodcat",       # EURO-D caseness (binary)
        "chronic2w9",     # 2+ chronic diseases (binary)
        "chronicw9",      # Chronic disease count
        # Additional health indicators
        "adl", "adl2",    # ADL limitations
        "iadl", "iadl2",  # IADL limitations
        "mobility", "mobilit2", "mobilit3",  # Mobility limitations
        "gali",           # Global Activity Limitation Indicator
        "bmi", "bmi2",    # BMI
        "maxgrip",        # Grip strength
        "casp",           # CASP-12 quality of life
        "loneliness",     # UCLA loneliness scale
        "phactiv",        # Physical inactivity
        "cf008tot",       # Immediate word recall
        "cf016tot",       # Delayed word recall
        "orienti",        # Orientation
        "numeracy2",      # Numeracy
    ])

    replace_share_missing(df)

    # ── OUTCOME 1: Self-Rated Health (fair/poor) ──
    # sphus: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor
    df["SRH_poor"] = (df["sphus"] >= 4).astype(float)
    df.loc[df["sphus"].isna(), "SRH_poor"] = np.nan

    # ── OUTCOME 2: EURO-D Depression Caseness ──
    # eurodcat: 1 = depressed (eurod >= 4), 0 = not depressed
    df["EUROD_dep"] = df["eurodcat"].copy()

    # ── OUTCOME 3: Multimorbidity ──
    # chronic2w9: 1 = 2+ chronic diseases
    df["Multimorbid"] = df["chronic2w9"].copy()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load physical health module (polypharmacy + individual items)
# ─────────────────────────────────────────────────────────────────────────────
def load_physical_health():
    """Load physical health variables including polypharmacy outcome."""
    print("Step 3: Loading physical health module (ph)...")

    # Chronic conditions, mobility, ADL items, aids, pain, polypharmacy
    usecols = ["mergeid", "ph003_", "ph004_", "ph005_", "ph082_",
               "ph011dno",  # Takes no drugs at all (for polypharmacy imputation)
               "ph084_", "ph085_",
               # Chronic conditions (individual)
               "ph006d1", "ph006d2", "ph006d3", "ph006d4", "ph006d5",
               "ph006d6", "ph006d10", "ph006d11", "ph006d12", "ph006d13",
               "ph006d14", "ph006d15", "ph006d16", "ph006d18", "ph006d19",
               "ph006d20", "ph006d21",
               # Mobility items
               "ph048d1", "ph048d2", "ph048d3", "ph048d4", "ph048d5",
               "ph048d6", "ph048d7", "ph048d8", "ph048d9", "ph048d10",
               # ADL items
               "ph049d1", "ph049d2", "ph049d3", "ph049d4", "ph049d5",
               "ph049d6",
               # IADL items
               "ph049d7", "ph049d8", "ph049d9", "ph049d10", "ph049d11",
               "ph049d12", "ph049d13", "ph049d14", "ph049d15",
               # Assistive devices
               "ph059d1", "ph059d2", "ph059d3", "ph059d4", "ph059d5",
               "ph059d6", "ph059d7", "ph059d8", "ph059d9", "ph059d10",
               # Vision/hearing
               "ph041_", "ph043_", "ph044_", "ph045_", "ph046_",
               # Frailty indicators
               "ph089d1", "ph089d2", "ph089d3", "ph089d4",
               # Drug types
               "ph011d1", "ph011d2", "ph011d3", "ph011d4", "ph011d6",
               "ph011d7", "ph011d8", "ph011d9", "ph011d10", "ph011d11",
               "ph011d13", "ph011d14", "ph011d15",
               # Weight/height
               "ph012_", "ph013_",
               # Work limitation
               "ph061_",
               ]

    df, meta = load_module("ph", usecols=usecols)
    replace_share_missing(df)

    # ── OUTCOME 4: Polypharmacy ──
    # ph082_: 1=Yes (5+ drugs/day), 5=No
    # NaN when respondent takes no drugs at all (ph011dno=1) → code as 0
    df["Polypharmacy"] = df["ph082_"].map({1: 1, 5: 0})
    # If ph011dno==1 (takes no drugs), polypharmacy is definitely 0
    no_drugs = df["ph011dno"].fillna(0) == 1
    df.loc[no_drugs & df["Polypharmacy"].isna(), "Polypharmacy"] = 0

    # ── Self-rated health (raw, for comparison) ──
    # ph003_: 1=Excellent, 2=Very good, 3=Good, 4=Fair, 5=Poor
    df["health_self_raw"] = df["ph003_"]

    # ── Long-term illness ──
    df["long_term_illness"] = df["ph004_"].map({1: 1, 5: 0})

    # ── Activity limitation (GALI) ──
    # ph005_: 1=Severely limited, 2=Limited not severely, 3=Not limited
    df["gali_raw"] = df["ph005_"]
    df["limited_activities"] = (df["ph005_"].isin([1, 2])).astype(float)
    df.loc[df["ph005_"].isna(), "limited_activities"] = np.nan

    # ── Chronic conditions (binary, 1=yes for each) ──
    chronic_map = {
        "ph006d1": "heart", "ph006d2": "hypertension", "ph006d3": "cholesterol",
        "ph006d4": "stroke", "ph006d5": "diabetes", "ph006d6": "lung_disease",
        "ph006d10": "cancer", "ph006d11": "ulcer", "ph006d12": "parkinsons",
        "ph006d13": "cataracts", "ph006d14": "hip_fracture",
        "ph006d15": "other_fracture", "ph006d16": "dementia",
        "ph006d18": "emotional_disorder", "ph006d19": "rheumatoid_arthritis",
        "ph006d20": "osteoarthritis", "ph006d21": "kidney_disease",
    }
    for raw, clean in chronic_map.items():
        # In SHARE: 1=yes, 0=no (already binary flag variables)
        df[clean] = df[raw].where(df[raw].isin([0, 1]), np.nan)

    # ── Vision ──
    df["wears_glasses"] = df["ph041_"].map({1: 1, 5: 0})
    # ph043_/ph044_: 1=Excellent...5=Poor (after correction)
    df["vision_distance"] = df["ph043_"]
    df["vision_close"] = df["ph044_"]

    # ── Hearing ──
    df["uses_hearing_aid"] = df["ph045_"].map({1: 1, 3: 1, 5: 0})  # 1=yes, 3=cochlear, 5=no
    df["hearing"] = df["ph046_"]  # 1=Excellent...5=Poor

    # ── Pain ──
    df["has_pain"] = df["ph084_"].map({1: 1, 5: 0})
    df["pain_level"] = df["ph085_"]  # 1=Mild, 2=Moderate, 3=Severe (conditional)

    # ── Assistive devices (binary) ──
    aid_map = {
        "ph059d1": "uses_cane", "ph059d2": "uses_walker",
        "ph059d3": "uses_manual_wheelchair", "ph059d4": "uses_electric_wheelchair",
        "ph059d5": "uses_buggy", "ph059d6": "uses_eating_utensils",
        "ph059d7": "uses_personal_alarm", "ph059d8": "uses_grab_bars",
        "ph059d9": "uses_raised_toilet", "ph059d10": "uses_incontinence_pads",
    }
    for raw, clean in aid_map.items():
        df[clean] = df[raw].where(df[raw].isin([0, 1]), np.nan)

    # ── Frailty symptoms ──
    df["fall"] = df["ph089d1"].where(df["ph089d1"].isin([0, 1]), np.nan)
    df["fear_of_falling"] = df["ph089d2"].where(df["ph089d2"].isin([0, 1]), np.nan)
    df["dizziness"] = df["ph089d3"].where(df["ph089d3"].isin([0, 1]), np.nan)
    df["fatigue_frailty"] = df["ph089d4"].where(df["ph089d4"].isin([0, 1]), np.nan)

    # ── Drug types (binary) ──
    drug_cols = [c for c in df.columns if c.startswith("ph011d")]
    for c in drug_cols:
        df[c] = df[c].where(df[c].isin([0, 1]), np.nan)

    # Number of drug types
    drug_items = ["ph011d1", "ph011d2", "ph011d3", "ph011d4", "ph011d6",
                  "ph011d7", "ph011d8", "ph011d9", "ph011d10", "ph011d11",
                  "ph011d13", "ph011d14", "ph011d15"]
    df["n_drug_types"] = df[drug_items].sum(axis=1, min_count=1)

    # ── Health limits work ──
    df["health_limits_work"] = df["ph061_"].map({1: 1, 5: 0})

    # Drop raw columns, keep cleaned versions
    drop_cols = (list(chronic_map.keys()) + list(aid_map.keys()) +
                 ["ph003_", "ph004_", "ph005_", "ph041_", "ph043_", "ph044_",
                  "ph045_", "ph046_", "ph082_", "ph011dno", "ph084_", "ph085_",
                  "ph061_"] +
                 [f"ph048d{i}" for i in range(1, 11)] +
                 [f"ph049d{i}" for i in range(1, 16)] +
                 [f"ph089d{i}" for i in range(1, 5)] +
                 drug_cols +
                 ["ph012_", "ph013_"])
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Mental health module
# ─────────────────────────────────────────────────────────────────────────────
def load_mental_health():
    """Load mental health variables (EURO-D items, loneliness)."""
    print("Step 4: Loading mental health module (mh)...")
    usecols = ["mergeid",
               # EURO-D items
               "mh002_", "mh003_", "mh004_", "mh005_", "mh007_", "mh008_",
               "mh010_", "mh011_", "mh013_", "mh014_", "mh016_", "mh017_",
               # Loneliness items
               "mh034_", "mh035_", "mh036_", "mh037_",
               ]
    df, meta = load_module("mh", usecols=usecols)
    replace_share_missing(df)

    # EURO-D individual items (all binary: 1=symptom present)
    eurod_map = {
        "mh002_": "dep_sadness", "mh003_": "dep_pessimism",
        "mh004_": "dep_suicidality", "mh007_": "dep_sleep",
        "mh010_": "dep_irritability", "mh013_": "dep_fatigue",
        "mh016_": "dep_no_enjoyment", "mh017_": "dep_tearfulness",
    }
    for raw, clean in eurod_map.items():
        df[clean] = df[raw].map({1: 1, 5: 0})

    # Guilt: mh005_ 1=marked, 2=vague, 3=no → binary (1 if marked or vague)
    df["dep_guilt"] = df["mh005_"].map({1: 1, 2: 1, 3: 0})

    # Interest: mh008_ 1=less, 2=same, 3=more → binary (1 if less)
    df["dep_interest_loss"] = df["mh008_"].map({1: 1, 2: 0, 3: 0})

    # Appetite: mh011_ 1=diminished, 2=same, 3=increased → binary
    df["dep_appetite_loss"] = df["mh011_"].map({1: 1, 2: 0, 3: 0})

    # Concentration: mh014_ 1=yes, 5=no
    df["dep_concentration"] = df["mh014_"].map({1: 1, 5: 0})

    # Loneliness items (1=often, 2=some of the time, 3=hardly ever)
    lone_map = {
        "mh034_": "lonely_companionship",
        "mh035_": "lonely_leftout",
        "mh036_": "lonely_isolated",
        "mh037_": "lonely_feels_lonely",
    }
    for raw, clean in lone_map.items():
        df[clean] = df[raw]  # Keep ordinal 1-3

    # Drop raw columns
    drop_raw = list(eurod_map.keys()) + list(lone_map.keys()) + [
        "mh005_", "mh008_", "mh011_", "mh014_"
    ]
    df = df.drop(columns=[c for c in drop_raw if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Cognitive function
# ─────────────────────────────────────────────────────────────────────────────
def load_cognitive():
    """Load cognitive function variables."""
    print("Step 5: Loading cognitive function module (cf)...")
    usecols = ["mergeid",
               "cf003_", "cf004_", "cf005_", "cf006_",  # Orientation (date, month, year, day)
               "cf010_",   # Verbal fluency (animals)
               "cf103_",   # Self-rated memory
               "cf012_",   # Serial 7s: 93
               "cf013_",   # Serial 7s: 86
               "cf014_",   # Serial 7s: 79
               "cf015_",   # Serial 7s: 72
               ]
    df, meta = load_module("cf", usecols=usecols)
    replace_share_missing(df)

    # Verbal fluency (continuous)
    df["verbal_fluency"] = df["cf010_"]

    # Self-rated memory (1=Excellent...5=Poor)
    df["memory_self"] = df["cf103_"]

    # Serial 7 subtraction score (count correct out of 4 in Wave 9)
    serial7_items = ["cf012_", "cf013_", "cf014_", "cf015_"]
    for col in serial7_items:
        df[col] = df[col].map({1: 1, 5: 0})  # 1=correct, 5=incorrect
    df["serial7_score"] = df[serial7_items].sum(axis=1, min_count=1)

    # Orientation score (0-4)
    orient_items = ["cf003_", "cf004_", "cf005_", "cf006_"]
    for col in orient_items:
        df[col] = df[col].map({1: 1, 5: 0})  # 1=correct, 5=incorrect
    df["orientation_score"] = df[orient_items].sum(axis=1, min_count=1)

    drop_cols = serial7_items + orient_items + ["cf010_", "cf103_"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Behavioral risks
# ─────────────────────────────────────────────────────────────────────────────
def load_behavioral_risks():
    """Load behavioral risk variables."""
    print("Step 6: Loading behavioral risks module (br)...")
    usecols = ["mergeid",
               "br001_",   # Ever smoked
               "br002_",   # Smoke at present
               "br015_",   # Vigorous physical activity
               "br016_",   # Moderate physical activity
               "br039_",   # Alcohol last 7 days
               "br026_",   # Dairy frequency
               "br027_",   # Legumes/eggs frequency
               "br028_",   # Meat/fish frequency
               "br029_",   # Fruits/vegetables frequency
               ]
    df, meta = load_module("br", usecols=usecols)
    replace_share_missing(df)

    # Smoking status: 0=never, 1=former, 2=current
    df["smoking_status"] = np.nan
    df.loc[df["br001_"] == 5, "smoking_status"] = 0  # Never smoked
    df.loc[(df["br001_"] == 1) & (df["br002_"] == 5), "smoking_status"] = 1  # Former
    df.loc[(df["br001_"] == 1) & (df["br002_"] == 1), "smoking_status"] = 2  # Current

    # Physical activity (1=more than once a week...4=hardly ever)
    df["vigorous_activity"] = df["br015_"]
    df["moderate_activity"] = df["br016_"]

    # Alcohol in last 7 days
    df["alcohol_last7d"] = df["br039_"].map({1: 1, 5: 0})

    # Diet frequency (1=every day...5=less than once a week, keep ordinal)
    df["diet_dairy"] = df["br026_"]
    df["diet_legumes_eggs"] = df["br027_"]
    df["diet_meat_fish"] = df["br028_"]
    df["diet_fruit_veg"] = df["br029_"]

    drop_cols = ["br001_", "br002_", "br015_", "br016_", "br039_",
                 "br026_", "br027_", "br028_", "br029_"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Healthcare utilization
# ─────────────────────────────────────────────────────────────────────────────
def load_healthcare():
    """Load healthcare utilization variables."""
    print("Step 7: Loading healthcare module (hc)...")
    usecols = ["mergeid",
               "hc010_",   # Dentist visit
               "hc012_",   # Hospitalized
               "hc013_",   # Times hospitalized
               "hc029_",   # Nursing home
               "hc113_",   # Supplementary insurance
               "hc602_",   # Doctor visits
               "hc884_",   # Flu vaccination
               "hc889_",   # Health literacy
               "hc841d1",  # Forgone dental due cost
               "hc841d2",  # Forgone prescribed drugs due cost
               "hc841d3",  # Forgone doctor visit due cost
               "hc841dno", # None forgone
               ]
    df, meta = load_module("hc", usecols=usecols)
    replace_share_missing(df)

    df["dentist_visit"] = df["hc010_"].map({1: 1, 5: 0})
    df["hospitalized"] = df["hc012_"].map({1: 1, 5: 0})
    df["n_hospitalizations"] = df["hc013_"].where(df["hc013_"] >= 0, np.nan)
    df["nursing_home"] = df["hc029_"].map({1: 1, 3: 1, 5: 0})  # 1=yes currently, 3=temporarily
    df["has_suppl_insurance"] = df["hc113_"].map({1: 1, 5: 0})
    df["n_doctor_visits"] = df["hc602_"].where(df["hc602_"] >= 0, np.nan)
    df["flu_vaccination"] = df["hc884_"].map({1: 1, 5: 0})
    df["health_literacy"] = df["hc889_"]  # 1=always...5=never (need help)

    # Forgone care due to cost (any type)
    forgo_cols = ["hc841d1", "hc841d2", "hc841d3"]
    for c in forgo_cols:
        df[c] = df[c].where(df[c].isin([0, 1]), np.nan)
    df["forgone_care_cost"] = df[forgo_cols].max(axis=1)

    drop_cols = ["hc010_", "hc012_", "hc013_", "hc029_", "hc113_",
                 "hc602_", "hc884_", "hc889_"] + forgo_cols + ["hc841dno"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Activities (life satisfaction, social activities)
# ─────────────────────────────────────────────────────────────────────────────
def load_activities():
    """Load activities variables."""
    print("Step 8: Loading activities module (ac)...")
    usecols = ["mergeid",
               "ac012_",    # Life satisfaction (0-10)
               # Social activities (Wave 9 available items)
               "ac035d1",   # Voluntary work
               "ac035d4",   # Education/training course
               "ac035d5",   # Sport/social club
               "ac035d7",   # Political/community organization
               "ac035d8",   # Read books/magazines
               "ac035d9",   # Word/number games
               "ac035d10",  # Played cards/board games
               "ac035dno",  # None of these
               ]
    df, meta = load_module("ac", usecols=usecols)
    replace_share_missing(df)

    df["life_satisfaction"] = df["ac012_"]

    # Social activities (binary)
    act_map = {
        "ac035d1": "does_voluntary", "ac035d4": "takes_course",
        "ac035d5": "sport_club", "ac035d7": "political_org",
        "ac035d8": "reads_books", "ac035d9": "word_games",
        "ac035d10": "card_games",
    }
    for raw, clean in act_map.items():
        df[clean] = df[raw].where(df[raw].isin([0, 1]), np.nan)

    # Number of social activities
    act_items = list(act_map.keys())
    df["n_social_activities"] = df[act_items].sum(axis=1, min_count=1)

    drop_cols = ["ac012_"] + act_items + ["ac035dno"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Demographics (detailed)
# ─────────────────────────────────────────────────────────────────────────────
def load_demographics():
    """Load detailed demographics.

    Note: In SHARE Wave 9, the dn module is only asked of NEW respondents
    (~22% of sample). Panel respondents carry forward from prior waves.
    We extract what's available and accept that some demographics will have
    high missingness for single-wave analysis. Education is obtained from
    gv_isced (available for all respondents).
    """
    print("Step 9: Loading demographics module (dn)...")
    usecols = ["mergeid",
               "dn004_",   # Born in country of interview
               "dn007_",   # Citizenship
               "dn014_",   # Marital status
               "dn041_",   # Years of education
               "dn042_",   # Gender confirmation (available for all)
               "dn044_",   # Marital status changed
               ]
    df, meta = load_module("dn", usecols=usecols)
    replace_share_missing(df)

    df["born_in_country"] = df["dn004_"].map({1: 1, 5: 0})
    df["is_citizen"] = df["dn007_"].map({1: 1, 5: 0})

    # Marital status: only available for ~22% (new respondents)
    # 1=married, 2=registered partner, 3=married/separated, 4=never married,
    # 5=divorced, 6=widowed
    df["marital_status"] = df["dn014_"].where(df["dn014_"].between(1, 6), np.nan)

    # For panel respondents with dn044_==5 (marital status NOT changed),
    # we know they maintained their previous status but don't have the value.
    # Flag this for transparency.
    df["marital_unchanged"] = (df["dn044_"] == 5).astype(float)
    df.loc[df["dn044_"].isna(), "marital_unchanged"] = np.nan

    # Years of education (available for ~22%)
    df["years_education"] = df["dn041_"].where(df["dn041_"] >= 0, np.nan)

    df = df.drop(columns=["dn004_", "dn007_", "dn014_", "dn041_", "dn042_", "dn044_"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 10: Education (ISCED harmonized)
# ─────────────────────────────────────────────────────────────────────────────
def load_education():
    """Load harmonized ISCED education classification."""
    print("Step 10: Loading education module (gv_isced)...")
    df, meta = load_module("gv_isced", usecols=[
        "mergeid", "isced1997_r",
    ])
    replace_share_missing(df)

    # ISCED-97: 0=pre-primary, 1=primary, 2=lower secondary, 3=upper secondary,
    # 4=post-secondary non-tertiary, 5=first stage tertiary, 6=second stage tertiary
    # 95=still in school, 97=other
    df["isced"] = df["isced1997_r"].where(
        df["isced1997_r"].between(0, 6), np.nan
    )

    # Group into 3 levels for analysis: low (0-1), medium (2-3-4), high (5-6)
    df["edu_level"] = pd.cut(
        df["isced"], bins=[-0.1, 1, 4, 6],
        labels=[0, 1, 2]  # 0=low, 1=medium, 2=high
    ).astype(float)

    df = df.drop(columns=["isced1997_r"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 11: Income and wealth (multiply imputed)
# ─────────────────────────────────────────────────────────────────────────────
def load_income_wealth():
    """Load multiply-imputed income and wealth, average across 5 implicates."""
    print("Step 11: Loading imputed income/wealth (gv_imputations)...")

    financial_vars = [
        "mergeid", "implicat",
        "thinc",    # Total household income
        "hnetw",    # Household net worth
        "hrass",    # Household real assets
        "hgfass",   # Household gross financial assets
        "hnfass",   # Household net financial assets
        "home",     # Value of main residence
        "bacc",     # Bank accounts
        "bsmf",     # Bonds/stocks/mutual funds
        "slti",     # Savings for long-term investments
        "vbus",     # Value of own business
        "car",      # Value of cars
        "liab",     # Financial liabilities
        "ydip",     # Earnings from employment
        "yind",     # Earnings from self-employment
        "ypen1",    # Old age/retirement pensions
        "ypen2",    # Private/occupational pensions
        "ypen3",    # Disability pensions
        "ypen4",    # Unemployment benefits
        "ypen5",    # Social assistance
        "ypen6",    # Sickness benefits
        "yreg1",    # Regular private payments
    ]

    df, meta = load_module("gv_imputations", usecols=financial_vars)

    # Average across 5 implicates (Rubin's rules — point estimate is the mean)
    agg_cols = [c for c in financial_vars if c not in ("mergeid", "implicat")]
    df_avg = df.groupby("mergeid")[agg_cols].mean().reset_index()

    # Rename for clarity
    rename_map = {
        "thinc": "hh_income", "hnetw": "hh_net_worth",
        "hrass": "hh_real_assets", "hgfass": "hh_gross_fin_assets",
        "hnfass": "hh_net_fin_assets", "home": "home_value",
        "bacc": "bank_accounts", "bsmf": "bonds_stocks",
        "slti": "long_term_savings", "vbus": "business_value",
        "car": "car_value", "liab": "financial_liabilities",
        "ydip": "income_employment", "yind": "income_self_employment",
        "ypen1": "pension_old_age", "ypen2": "pension_occupational",
        "ypen3": "pension_disability", "ypen4": "unemployment_benefits",
        "ypen5": "social_assistance", "ypen6": "sickness_benefits",
        "yreg1": "private_transfers",
    }
    df_avg = df_avg.rename(columns=rename_map)

    return df_avg


# ─────────────────────────────────────────────────────────────────────────────
# Step 12: Consumption
# ─────────────────────────────────────────────────────────────────────────────
def load_consumption():
    """Load consumption/financial hardship variables."""
    print("Step 12: Loading consumption module (co)...")
    usecols = ["mergeid",
               "co007_",   # Make ends meet (1=great difficulty...4=easily)
               ]
    df, meta = load_module("co", usecols=usecols)
    replace_share_missing(df)

    df["make_ends_meet"] = df["co007_"]  # Keep ordinal 1-4
    df = df.drop(columns=["co007_"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 13: Social networks
# ─────────────────────────────────────────────────────────────────────────────
def load_social_networks():
    """Load social network variables from sn module + generated networks."""
    print("Step 13: Loading social networks (sn + gv_networks)...")

    # sn012_ has network satisfaction
    df_sn, _ = load_module("sn", usecols=["mergeid", "sn012_"])
    replace_share_missing(df_sn)
    df_sn["network_satisfaction"] = df_sn["sn012_"]
    df_sn = df_sn.drop(columns=["sn012_"])

    # gv_networks has network size and combined satisfaction
    df_gv, _ = load_module("gv_networks", usecols=[
        "mergeid", "sn_size_w9", "sn_satisfaction"
    ])
    replace_share_missing(df_gv)
    df_gv = df_gv.rename(columns={"sn_size_w9": "network_size"})
    df_gv = df_gv.drop(columns=["sn_satisfaction"])  # Use raw sn012_ instead

    df = df_sn.merge(df_gv, on="mergeid", how="outer")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 14: Employment
# ─────────────────────────────────────────────────────────────────────────────
def load_employment():
    """Load employment variables."""
    print("Step 14: Loading employment module (ep)...")
    usecols = ["mergeid",
               "ep005_",   # Current job situation
               ]
    df, meta = load_module("ep", usecols=usecols)
    replace_share_missing(df)

    # ep005_: 1=retired, 2=employed, 3=unemployed, 4=sick/disabled,
    #         5=homemaker, 97=other
    df["job_situation"] = df["ep005_"].where(df["ep005_"].between(1, 5), np.nan)
    df = df.drop(columns=["ep005_"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 15: Social participation / support
# ─────────────────────────────────────────────────────────────────────────────
def load_social_participation():
    """Load social participation variables."""
    print("Step 15: Loading social participation module (sp)...")
    usecols = ["mergeid",
               "sp002_",   # Received help from outside
               "sp008_",   # Given help to others
               ]
    df, meta = load_module("sp", usecols=usecols)
    replace_share_missing(df)

    df["received_help"] = df["sp002_"].map({1: 1, 5: 0})
    df["given_help"] = df["sp008_"].map({1: 1, 5: 0})

    df = df.drop(columns=["sp002_", "sp008_"])
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 16: Grip strength
# ─────────────────────────────────────────────────────────────────────────────
def load_grip_strength():
    """Load grip strength measurements."""
    print("Step 16: Loading grip strength module (gs)...")
    usecols = ["mergeid",
               "gs006_", "gs007_",   # Left hand trials
               "gs008_", "gs009_",   # Right hand trials
               ]
    df, meta = load_module("gs", usecols=usecols)
    replace_share_missing(df)

    # Max grip across all valid trials
    grip_cols = ["gs006_", "gs007_", "gs008_", "gs009_"]
    for c in grip_cols:
        df[c] = df[c].where(df[c] > 0, np.nan)
    df["max_grip_strength"] = df[grip_cols].max(axis=1)

    df = df.drop(columns=grip_cols)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Step 17: Big 5 personality
# ─────────────────────────────────────────────────────────────────────────────
def load_personality():
    """Load Big 5 personality traits (generated)."""
    print("Step 17: Loading personality (gv_big5)...")
    df, meta = load_module("gv_big5")
    replace_share_missing(df)

    # Keep generated Big 5 scores if available
    big5_cols = [c for c in df.columns if c.startswith("bfi10_")]
    if big5_cols:
        keep = ["mergeid"] + big5_cols
        rename = {}
        for c in big5_cols:
            trait = c.replace("bfi10_", "")
            rename[c] = f"personality_{trait}"
        df = df[keep].rename(columns=rename)
    else:
        df = df[["mergeid"]].copy()

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("SHARE Wave 9 Data Cleaning Pipeline")
    print("=" * 70)

    cleaning_log = {"timestamp": datetime.now().isoformat(), "steps": {}}

    # Load all modules
    base, base_log = load_base_population()
    cleaning_log["steps"]["base_population"] = base_log
    print(f"  Base population: {len(base)} respondents")

    gv_health = load_gv_health()
    ph = load_physical_health()
    mh = load_mental_health()
    cf = load_cognitive()
    br = load_behavioral_risks()
    hc = load_healthcare()
    ac = load_activities()
    dn = load_demographics()
    edu = load_education()
    income = load_income_wealth()
    co = load_consumption()
    sn = load_social_networks()
    ep = load_employment()
    sp = load_social_participation()
    gs = load_grip_strength()
    big5 = load_personality()

    # ── Merge all modules ──
    print("\nMerging all modules on mergeid...")
    df = base.copy()
    modules = [gv_health, ph, mh, cf, br, hc, ac, dn, edu, income, co, sn, ep, sp, gs, big5]
    for mod in modules:
        df = df.merge(mod, on="mergeid", how="left")

    print(f"  After merge: {df.shape[0]} rows x {df.shape[1]} columns")
    cleaning_log["steps"]["after_merge"] = {
        "n_rows": df.shape[0], "n_cols": df.shape[1]
    }

    # ── Exclusion criteria ──
    print("\nApplying exclusion criteria...")
    n_before = len(df)

    # 1. Must have age >= 50 (SHARE target population)
    df = df[df["age"] >= 50].copy()
    n_age = len(df)
    print(f"  Age >= 50: {n_before} → {n_age} (dropped {n_before - n_age})")

    # 2. Must have all 4 outcomes non-missing
    outcomes = ["SRH_poor", "EUROD_dep", "Multimorbid", "Polypharmacy"]
    for outcome in outcomes:
        n_pre = len(df)
        df = df[df[outcome].notna()].copy()
        n_post = len(df)
        if n_pre > n_post:
            print(f"  Non-missing {outcome}: {n_pre} → {n_post} (dropped {n_pre - n_post})")

    n_final = len(df)
    cleaning_log["steps"]["exclusions"] = {
        "before_exclusions": n_before,
        "after_age_filter": n_age,
        "after_outcome_filter": n_final,
        "total_excluded": n_before - n_final,
    }

    # ── Document SES impact of exclusions ──
    print("\n  Checking SES impact of exclusions...")
    if "hh_income" in df.columns:
        # Compare income distribution of included vs excluded
        full_income = income["hh_income"]
        kept_income = df["hh_income"]
        print(f"  Included sample mean income: {kept_income.mean():.0f}")
        print(f"  Full sample mean income: {full_income.mean():.0f}")
        cleaning_log["steps"]["ses_impact"] = {
            "included_mean_income": float(kept_income.mean()),
            "full_mean_income": float(full_income.mean()),
        }

    # ── Outcome distributions ──
    print("\nOutcome distributions:")
    for outcome in outcomes:
        prev = df[outcome].mean()
        n_pos = df[outcome].sum()
        n_neg = len(df) - n_pos
        print(f"  {outcome}: {prev*100:.1f}% positive ({int(n_pos)} / {int(n_pos + n_neg)})")
        cleaning_log["steps"][f"outcome_{outcome}"] = {
            "prevalence": float(prev),
            "n_positive": int(n_pos),
            "n_negative": int(n_neg),
        }

    # ── Drop features with >50% missing ──
    # These are mostly from the dn module (panel respondents not re-asked),
    # personality (only 20% coverage), and conditional questions.
    feature_cols = [c for c in df.columns if c not in (["mergeid"] + outcomes)]
    drop_high_missing = []
    for col in feature_cols:
        pct = df[col].isna().mean() * 100
        if pct > 50:
            drop_high_missing.append((col, pct))
    if drop_high_missing:
        print(f"\n  Dropping {len(drop_high_missing)} features with >50% missing:")
        for col, pct in drop_high_missing:
            print(f"    {col}: {pct:.1f}% missing")
        df = df.drop(columns=[col for col, _ in drop_high_missing])
    cleaning_log["steps"]["dropped_high_missing"] = {
        col: f"{pct:.1f}%" for col, pct in drop_high_missing
    }

    # ── Feature missingness report (remaining) ──
    print("\nRemaining feature missingness (>5%):")
    feature_cols = [c for c in df.columns if c not in (["mergeid"] + outcomes)]
    high_missing = []
    for col in feature_cols:
        pct = df[col].isna().mean() * 100
        if pct > 5:
            high_missing.append((col, pct))
            print(f"  {col}: {pct:.1f}% missing")
    cleaning_log["steps"]["remaining_high_missing"] = {
        col: f"{pct:.1f}%" for col, pct in high_missing
    }

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print(f"Final dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Outcomes: {outcomes}")
    print(f"Features: {df.shape[1] - len(outcomes) - 1}")  # -1 for mergeid
    print(f"{'=' * 70}")

    # ── Save ──
    print(f"\nSaving to {OUTPUT_ZIP}...")
    df.to_csv(OUTPUT_FILE, index=False)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(OUTPUT_FILE)
    os.remove(OUTPUT_FILE)

    # Save cleaning log
    cleaning_log["final"] = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "outcomes": outcomes,
        "n_features": df.shape[1] - len(outcomes) - 1,
        "columns": list(df.columns),
    }
    with open(CLEANING_LOG, "w") as f:
        json.dump(cleaning_log, f, indent=2)

    print(f"Cleaning log saved to {CLEANING_LOG}")
    print("Done!")

    return df


if __name__ == "__main__":
    df = main()
