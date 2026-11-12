from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student-mental-health.csv"
CLEANED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"


def load_raw_data(dataset_path=RAW_DATA_PATH):
    return pd.read_csv(dataset_path)


def clean_data(df):
    invalid_rows = df[
        df["Screen_Time_Hours"] < df["Social_Media_Hours"]
    ].index

    return df.drop(index=invalid_rows).copy()


def prepare_dataset(
    dataset_path=RAW_DATA_PATH,
    output_path=CLEANED_DATA_PATH,
):
    df = load_raw_data(dataset_path)
    df_clean = clean_data(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    print(f"Raw: {df.shape}")
    print(f"Cleaned: {df_clean.shape}")

    return df_clean


if __name__ == "__main__":
    prepare_dataset()