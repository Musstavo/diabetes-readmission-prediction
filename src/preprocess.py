import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def map_diags(code):
    if code == "?" or code is np.nan:
        return "Other"
    try:
        val = float(code)
        if 390 <= val <= 459 or val == 785:
            return "Circulatory"
        if 460 <= val <= 519 or val == 786:
            return "Respiratory"
        if 520 <= val <= 579 or val == 787:
            return "Digestive"
        if 250 <= val < 251:
            return "Diabetes"
        if 800 <= val <= 999:
            return "Injury"
        if 710 <= val <= 749:
            return "Musculoskeletal"
        if 580 <= val <= 629 or val == 788:
            return "Genitourinary"
        if 140 <= val <= 239:
            return "Neoplasms"
        return "Other"
    except ValueError:
        return "Other"


def run_pipeline(df, is_train=True, label_encoder=None):
    drop_cols = [
        "encounter_id",
        "patient_nbr",
        "weight",
        "medical_specialty",
        "payer_code",
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
    ]
    df = df.drop(drop_cols, axis=1, errors="ignore")

    df["max_glu_serum"] = df["max_glu_serum"].fillna("None")
    df["A1Cresult"] = df["A1Cresult"].fillna("None")

    for col in ["diag_1", "diag_2", "diag_3"]:
        df[col] = df[col].apply(map_diags)

    cols_to_encode = [
        "race",
        "gender",
        "max_glu_serum",
        "A1Cresult",
        "change",
        "diabetesMed",
        "diag_1",
        "diag_2",
        "diag_3",
    ]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True)

    return df
