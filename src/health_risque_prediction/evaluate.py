from pathlib import Path
import json

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import LabelBinarizer

from health_risque_prediction.preprocessing import (
    TARGET_COLUMN,
    TEST_DATA_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BEST_MODEL_PATH = (
    PROJECT_ROOT / "models" / "best_mental_health_model.joblib"
)

REPORTS_DIR = PROJECT_ROOT / "reports"


def evaluate_best_model(
    test_data_path=TEST_DATA_PATH,
    model_path=BEST_MODEL_PATH,
    reports_dir=REPORTS_DIR,
):
    test_df = pd.read_csv(test_data_path)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    best_model = joblib.load(model_path)

    y_test_pred = best_model.predict(X_test)

    metrics = {
        "accuracy": float(
            accuracy_score(y_test, y_test_pred)
        ),
        "precision_weighted": float(
            precision_score(
                y_test,
                y_test_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_test,
                y_test_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_test,
                y_test_pred,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    y_proba = best_model.predict_proba(X_test)

    metrics["roc_auc_ovr_weighted"] = float(
        roc_auc_score(
            y_test,
            y_proba,
            labels=best_model.classes_,
            multi_class="ovr",
            average="weighted",
        )
    )

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(
        reports_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    report = classification_report(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    (reports_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    cm = confusion_matrix(y_test, y_test_pred)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=best_model.classes_,
    )

    disp.plot(cmap="Blues")

    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        reports_dir / "confusion_matrix.png",
        dpi=150,
    )
    plt.close()

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(best_model.classes_)

    y_test_bin = label_binarizer.transform(y_test)
    classes = label_binarizer.classes_

    fig, axes = plt.subplots(
        1,
        len(classes),
        figsize=(6 * len(classes), 5),
    )

    if len(classes) == 1:
        axes = [axes]

    for class_id, class_name in enumerate(classes):
        ax = axes[class_id]

        RocCurveDisplay.from_predictions(
            y_test_bin[:, class_id],
            y_proba[:, class_id],
            name="Best model",
            ax=ax,
        )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Random classifier",
        )

        ax.set_title(f"{class_name} vs Rest")
        ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(
        reports_dir / "roc_curves.png",
        dpi=150,
    )
    plt.close()

    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    print("Evaluation complete.")

    return metrics


if __name__ == "__main__":
    evaluate_best_model()