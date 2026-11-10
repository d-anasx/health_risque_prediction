import numpy as np


def get_categorical_columns(X):
    return X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

def get_numerical_columns(X):
    return X.select_dtypes(include= np.number).columns.tolist()