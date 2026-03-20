import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import joblib
import logging
from train_model import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = 'https://raw.githubusercontent.com/selva86/datasets/master/Cars93_miss.csv'
RAW_DATA_PATH = '/tmp/cars_raw.csv'
CLEAN_DATA_PATH = '/tmp/cleaned_data.csv'


def download_data(**context):
    response = requests.get(DATA_URL, timeout=10)
    response.raise_for_status()

    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)

    context['ti'].xcom_push(key='data_shape', value=str(df.shape))
    return RAW_DATA_PATH


def clear_data(**context):
    df = pd.read_csv(RAW_DATA_PATH)

    df = df[['Manufacturer', 'Model', 'Type', 'Price', 'EngineSize', 'Horsepower', 'MPG.city', 'MPG.highway']]

    df = df.fillna(df.median(numeric_only=True))
    df = df.fillna('Unknown')

    cat_columns = ['Manufacturer', 'Model', 'Type']

    ordinal = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    df[cat_columns] = ordinal.fit_transform(df[cat_columns])

    df.to_csv(CLEAN_DATA_PATH, index=False)
    joblib.dump(ordinal, '/tmp/ordinal_encoder.pkl')

    return CLEAN_DATA_PATH


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


dag = DAG(
    'train_pipe',
    default_args=default_args,
    schedule=timedelta(days=1),
    catchup=False,
    tags=['ml', 'cars', 'training'],
)


download_task = PythonOperator(
    task_id='download_dataset',
    python_callable=download_data,
    dag=dag
)


clear_task = PythonOperator(
    task_id='clear_data',
    python_callable=clear_data,
    dag=dag
)


train_task = PythonOperator(
    task_id='train_model',
    python_callable=train,
    dag=dag
)


download_task >> clear_task >> train_task
