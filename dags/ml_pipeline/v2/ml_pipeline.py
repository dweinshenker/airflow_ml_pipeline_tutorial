# This DAG will use Airflow to run the various load, train, and test tasks

# Airflow is meant for workflow orchestration

# Prequisites
# Install Airflow => https://airflow.apache.org/docs/apache-airflow/stable/start.html

from datetime import timedelta

from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash_operator import BashOperator


# Define DAG parameters
# Include overrides for defaults in `default_args`

with DAG(
    dag_id="fashion_image_categorization",
    default_args={
        'start_date': days_ago(2),
    },
    schedule_interval='0 0 * * *',
    dagrun_timeout=timedelta(minutes=60)
) as dag:
    # we are breaking our load, train, and test steps into tasks
    load_data = BashOperator(
        task_id="load",
        dag=dag,
        bash_command="python3 ~/airflow/dags/ml_pipeline/v2/load.py"
    )
    train = BashOperator(
        task_id="train",
        dag=dag,
        bash_command="python3 ~/airflow/dags/ml_pipeline/v2/train.py"
    )
    test = BashOperator(
        task_id="test",
        dag=dag,
        bash_command="python3 ~/airflow/dags/ml_pipeline/v2/test.py"
    )
    load_data >> train >> test
