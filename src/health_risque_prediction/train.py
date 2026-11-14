from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from health_risque_prediction.preprocessing import (
    TARGET_COLUMN,
    TRAIN_DATA_PATH,
    create_preprocessor,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

BEST_MODEL_PATH = MODELS_DIR / "best_mental_health_model.joblib"
TUNING_RESULTS_PATH = REPORTS_DIR / "tuning_results.csv"


def load_train_data(train_data_path=TRAIN_DATA_PATH):
    train_df = pd.read_csv(train_data_path)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    return X_train, y_train


def build_models(preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
        ),
        "SVC": SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            random_state=42,
        ),
    }

    pipelines = {}

    for name, model in models.items():
        pipelines[name] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

    return pipelines


def get_param_grids():
    return {
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1, 10, 100],
            "classifier__max_iter" : [1000, 2000],
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 5, 10],
            "classifier__min_samples_split": [5, 10],
        },
        "SVC": {
            "classifier__C": [0.1, 1, 10],
            "classifier__gamma": ["scale", "auto"],
            "classifier__kernel": ["rbf", "linear"],
        },
    }


def train_and_tune(
    train_data_path=TRAIN_DATA_PATH,
    model_output_path=BEST_MODEL_PATH,
    tuning_output_path=TUNING_RESULTS_PATH,
):
    X_train, y_train = load_train_data(train_data_path)

    preprocessor = create_preprocessor(X_train)
    pipelines = build_models(preprocessor)
    param_grids = get_param_grids()

    best_models = {}
    tuning_results = []

    for name, pipeline in pipelines.items():
        print(f"Training: {name}")

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grids[name],
            scoring="f1_weighted",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X_train, y_train)

        best_models[name] = grid_search.best_estimator_

        tuning_results.append({
            "model": name,
            "best F1-Score": grid_search.best_score_,
            "best parameters": grid_search.best_params_,
        })

        print(f"Best F1: {grid_search.best_score_:.4f}")

    tuning_df = pd.DataFrame(tuning_results)

    tuning_df = tuning_df.sort_values(
        by="best F1-Score",
        ascending=False,
    )

    best_model_name = tuning_df.iloc[0]["model"]
    best_model = best_models[best_model_name]

    model_output_path = Path(model_output_path)
    tuning_output_path = Path(tuning_output_path)

    model_output_path.parent.mkdir(parents=True, exist_ok=True)
    tuning_output_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_output_path)
    tuning_df.to_csv(tuning_output_path, index=False)

    print(f"Best model: {best_model_name}")
    print("Model saved.")
    print("Tuning results saved.")

    return best_model_name, best_model, tuning_df


if __name__ == "__main__":
    train_and_tune()