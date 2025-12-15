import pandas as pd
import numpy as np

# ======================================================
# cardiometabolic cutoffs (NCEP ATP III)
# ======================================================
FASTING_CUTOFFS = {
    "trig": 1.7,
    "pglu": 5.6,
    "systolic": 130,
    "diastolic": 85,
    "hdl": {"male": 1.0, "female": 1.3},
    "waist_cm": {"male": 102, "female": 88},
}

POSTPRANDIAL_CUTOFFS = {
    "trig_360": 1.7,
    "pglu_360": 7.8,
    "systolic": 130,
    "diastolic": 85,
    "hdl_360": {"male": 1.0, "female": 1.3},
}

# ======================================================
# count abnormal markers
# ======================================================
def count_abnormal(row, cutoffs, mode="fasting"):
    count = 0
    for marker, cutoff in cutoffs.items():
        if isinstance(cutoff, dict):  # sex-specific
            sex = row["sex"]
            if row[marker] is not np.nan:
                if ("hdl" in marker and row[marker] < cutoff[sex]) or \
                   ("waist" in marker and row[marker] > cutoff[sex]):
                    count += 1
        else:
            if row[marker] > cutoff:
                count += 1
    return count


# ======================================================
# BP completeness rule
# ======================================================
def apply_bp_rules(df):
    df = df.copy()
    df["valid_bp_count"] = df[["systolic", "diastolic"]].notna().sum(axis=1)
    df.loc[df["valid_bp_count"] == 0, "valid_bp_count"] = np.nan

    cond1 = (df["valid_bp_count"] == 2) & (df.notna().sum(axis=1) >= 4)
    cond2 = (df["valid_bp_count"] == 1) & (df.notna().sum(axis=1) >= 5)
    cond3 = (df["valid_bp_count"].isna()) & (df.notna().sum(axis=1) >= 3)

    return df[cond1 | cond2 | cond3]


# ======================================================
# process one dataset
# ======================================================
def process_dataset(
    df,
    marker_columns,
    confounders,
    cutoffs,
    mode="fasting",
    use_bp=True,
):
    markers = df[marker_columns]

    if use_bp:
        markers = apply_bp_rules(markers)
    else:
        markers = markers[markers.notna().sum(axis=1) >= 3]

    out = markers.join(df[confounders])

    out["unhealthy"] = out.apply(
        count_abnormal,
        axis=1,
        cutoffs=cutoffs,
        mode=mode,
    )

    out["group"] = np.where(out["unhealthy"] >= 3, "high_risk", "low_risk")
    return out


# ======================================================
# data processing of each dataset
# ======================================================
# dataset1: fasting + BP
dataset1 = process_dataset(
    df=dataset1_df,
    marker_columns=["waist_cm", "hdl", "trig", "pglu", "systolic", "diastolic"],
    confounders=["age", "sex", "bmi", "height"],
    cutoffs=FASTING_CUTOFFS,
    mode="fasting",
    use_bp=True,
)

# dataset2: fasting + BP (no waist)
dataset2 = process_dataset(
    df=dataset2_df,
    marker_columns=["hdl", "trig", "pglu", "systolic", "diastolic"],
    confounders=["age", "sex", "bmi", "height"],
    cutoffs=FASTING_CUTOFFS,
    mode="fasting",
    use_bp=True,
)

# dataset3: postprandial + BP
dataset3 = process_dataset(
    df=dataset3_df,
    marker_columns=["hdl_360", "trig_360", "pglu_360", "systolic", "diastolic"],
    confounders=["age", "sex", "bmi", "height"],
    cutoffs=POSTPRANDIAL_CUTOFFS,
    mode="postprandial",
    use_bp=True,
)

# dataset4: postprandial + BP (no glucose)
dataset4 = process_dataset(
    df=dataset4_df,
    marker_columns=["hdl_360", "trig_360", "systolic", "diastolic"],
    confounders=["age", "sex", "bmi", "height"],
    cutoffs=POSTPRANDIAL_CUTOFFS,
    mode="postprandial",
    use_bp=True,
)

# dataset5: fasting, no BP
dataset5 = process_dataset(
    df=dataset5_df,
    marker_columns=["waist_cm", "hdl", "trig", "pglu"],
    confounders=["age", "sex", "bmi", "height"],
    cutoffs=FASTING_CUTOFFS,
    mode="fasting",
    use_bp=False,
)
