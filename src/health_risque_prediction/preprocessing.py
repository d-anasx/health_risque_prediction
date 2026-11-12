from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
TRAIN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "train_data.csv"
TEST_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "test_data.csv"

TARGET_COLUMN = "Mental_Health_Risk"
ID_COLUMN = "Student_ID"

NOMINAL_COLS = [
    "Gender",
    "Part_Time_Job",
    "Previous_Mental_Health_History",
]

ORDINAL_COLS = [
    "Physical_Activity",
    "Diet_Quality",
]

ORDINAL_CATEGORIES = [
    ["Low", "Moderate", "High"],
    ["Poor", "Average", "Good"],
]


def split_and_save_data(
    cleaned_data_path=CLEANED_DATA_PATH,
    train_output_path=TRAIN_DATA_PATH,
    test_output_path=TEST_DATA_PATH,
):
    df_clean = pd.read_csv(cleaned_data_path)

    X = df_clean.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = df_clean[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    min_screen_time = X_train["Screen_Time_Hours"].min()

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train["Social_Media_Hours"] = (
        X_train["Social_Media_Hours"].fillna(min_screen_time)
    )

    X_test["Social_Media_Hours"] = (
        X_test["Social_Media_Hours"].fillna(min_screen_time)
    )

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_output_path = Path(train_output_path)
    test_output_path = Path(test_output_path)

    train_output_path.parent.mkdir(parents=True, exist_ok=True)
    test_output_path.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"Train: {train_df.shape}")
    print(f"Test: {test_df.shape}")
    print(f"Imputation value: {min_screen_time}")

    return X_train, X_test, y_train, y_test


def create_preprocessor(X_train):
    numerical_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    nominal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(
                categories=ORDINAL_CATEGORIES,
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
        ),
    ])

    preprocessor = ColumnTransformer([
        ("numerical", numerical_pipeline, numerical_cols),
        ("nominal", nominal_pipeline, NOMINAL_COLS),
        ("ordinal", ordinal_pipeline, ORDINAL_COLS),
    ])

    return preprocessor


if __name__ == "__main__":
    split_and_save_data()