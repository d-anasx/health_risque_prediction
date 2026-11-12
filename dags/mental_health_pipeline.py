import sys
from datetime import datetime

from airflow.decorators import dag, task

sys.path.insert(0, "/opt/airflow/src")

from health_risque_prediction.data_loader import prepare_dataset
from health_risque_prediction.preprocessing import split_and_save_data
from health_risque_prediction.train import train_and_tune
from health_risque_prediction.evaluate import evaluate_best_model


@dag(
    dag_id="mental_health_pipeline",
    start_date=datetime(2026, 11, 12),
    schedule=None,
    catchup=False,
    tags=["machine-learning"],
)
def mental_health_pipeline():

    @task
    def prepare_data():
        prepare_dataset()

    @task
    def preprocess_data():
        split_and_save_data()

    @task
    def train_model():
        train_and_tune()

    @task
    def evaluate_model():
        evaluate_best_model()

    data = prepare_data()
    processed = preprocess_data()
    model = train_model()
    evaluation = evaluate_model()

    data >> processed >> model >> evaluation


mental_health_pipeline()