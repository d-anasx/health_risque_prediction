import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_data(path):
    df = pd.read_csv(path)

    # Remove invalid rows 
    invalid_rows = (
        df["Social_Media_Hours"] > df["Screen_Time_Hours"]
    )
    df = df.loc[~invalid_rows].copy()

    X = df.drop(columns=["Mental_Health_Risk", "Student_ID"])
    y = df["Mental_Health_Risk"]

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )