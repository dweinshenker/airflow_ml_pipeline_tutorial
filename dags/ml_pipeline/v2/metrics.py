import logging
from airflow.stats import Stats

ML_PIPELINE_ACCURACY = "ml_pipeline_accuracy"

def generate_ml_metrics(accuracy):
    try:
        Stats.gauge(f"{ML_PIPELINE_ACCURACY}", accuracy)
    except:
        logging.exception("Caught exception during statistics logging")