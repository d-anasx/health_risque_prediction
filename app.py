
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "best_mental_health_model.joblib"
REPORTS_DIR = BASE_DIR / "reports"

METRICS_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_PATH = REPORTS_DIR / "confusion_matrix.png"
ROC_PATH = REPORTS_DIR / "roc_curves.png"
CLASSIFICATION_PATH = REPORTS_DIR / "classification_report.txt"


st.set_page_config(
    page_title="Student Mental Health Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("Student Mental Health Dashboard")

st.warning(
    "Educational prototype only. Predictions are not clinical diagnoses "
    "or screening results."
)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def show_performance():
    st.header("Model Performance")

    if not METRICS_PATH.exists():
        st.info("Evaluation metrics are not available yet.")
        return

    with open(METRICS_PATH, encoding="utf-8") as file:
        metrics = json.load(file)

    columns = st.columns(len(metrics))

    for column, (name, value) in zip(columns, metrics.items()):
        column.metric(
            name.replace("_", " ").title(),
            f"{value:.3f}",
        )

    st.header("Model Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Confusion Matrix")

        if CONFUSION_PATH.exists():
            st.image(str(CONFUSION_PATH), use_container_width=True)

    with col2:
        st.subheader("ROC Curves")

        if ROC_PATH.exists():
            st.image(str(ROC_PATH), use_container_width=True)

    if CLASSIFICATION_PATH.exists():
        with st.expander("Classification Report"):
            report = CLASSIFICATION_PATH.read_text(encoding="utf-8")
            st.code(report)


def show_prediction():
    st.header("Test a New Student")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            age = st.number_input("Age", min_value=10, max_value=100, value=20)
            gender = st.selectbox(
                "Gender", ["Male", "Female", "Non-binary"]
            )
            year = st.number_input(
                "Year of Study", min_value=1, max_value=10, value=1
            )
            sleep = st.number_input(
                "Sleep Hours", min_value=0.0, max_value=24.0, value=7.0
            )
            study = st.number_input(
                "Study Hours Per Week", min_value=0.0, value=20.0
            )
            gpa = st.number_input(
                "GPA", min_value=0.0, max_value=4.0, value=3.0
            )

        with col2:
            social_media = st.number_input(
                "Social Media Hours", min_value=0.0, value=2.0
            )
            screen_time = st.number_input(
                "Screen Time Hours", min_value=0.0, value=5.0
            )
            physical_activity = st.selectbox(
                "Physical Activity", ["Low", "Moderate", "High"]
            )
            financial_stress = st.number_input(
                "Financial Stress", min_value=0.0, value=3.0
            )
            family_support = st.number_input(
                "Family Support", min_value=0.0, value=3.0
            )
            attendance = st.number_input(
                "Attendance Rate", min_value=0.0, max_value=100.0, value=85.0
            )

        part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

        diet_quality = st.selectbox(
            "Diet Quality", ["Poor", "Average", "Good"]
        )

        previous_history = st.selectbox(
            "Previous Mental Health History", ["No", "Yes"]
        )

        submitted = st.form_submit_button("Predict")

    if submitted:

        if social_media > screen_time:
            st.error("Social media hours cannot exceed screen time.")
            return

        student = pd.DataFrame([{
            "Age": age,
            "Gender": gender,
            "Year_of_Study": year,
            "Sleep_Hours": sleep,
            "Study_Hours_Per_Week": study,
            "GPA": gpa,
            "Social_Media_Hours": social_media,
            "Screen_Time_Hours": screen_time,
            "Physical_Activity": physical_activity,
            "Financial_Stress": financial_stress,
            "Family_Support": family_support,
            "Attendance_Rate": attendance,
            "Part_Time_Job": part_time_job,
            "Diet_Quality": diet_quality,
            "Previous_Mental_Health_History": previous_history,
        }])

        model = load_model()

        prediction = model.predict(student)[0]

        st.subheader("Prediction Result")
        st.success(f"Predicted category: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(student)[0]

            st.subheader("Class Probabilities")

            probability_df = pd.DataFrame({
                "Category": model.classes_,
                "Probability": probabilities,
            })

            st.bar_chart(
                probability_df.set_index("Category")
            )


show_performance()

st.divider()

show_prediction()
