import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, PowerTransformer
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
import requests
import os
import logging
import joblib
from train_model import train

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_URL = 'https://raw.githubusercontent.com/dayekb/Basic_ML_Alg/main/cars_moldova_no_dup.csv'
RAW_DATA_PATH = '/tmp/cars_raw.csv'
CLEAN_DATA_PATH = '/tmp/cleaned_data.csv'

def download_data(**context):
    response = requests.get(DATA_URL, timeout=10)
    response.raise_for_status()
    df = pd.read_csv(DATA_URL)
    df.to_csv(RAW_DATA_PATH, index=False)
    context['ti'].xcom_push(key='data_shape', value=str(df.shape))
    return RAW_DATA_PATH

def validate_data(df):
    required_columns = ['Make', 'Model', 'Year', 'Price(euro)', 'Distance']
    missing_columns = [col for col in required_columns if col not in df.columns]

def clear_data(**context):
    df = pd.read_csv(RAW_DATA_PATH)
    validate_data(df)

    initial_size = len(df)
    context['ti'].xcom_push(key='initial_size', value=initial_size)

    cat_columns = ['Make', 'Model', 'Style', 'Fuel_type', 'Transmission']
    num_columns = ['Year', 'Distance', 'Engine_capacity(cm3)', 'Price(euro)']

    df = df[~((df.Year < 2021) & (df.Distance < 1100))]
    df = df[df.Distance <= 1e6]
    df = df[(df["Engine_capacity(cm3)"] >= 200) & (df["Engine_capacity(cm3)"] <= 5000)]
    df = df[(df["Price(euro)"] >= 101) & (df["Price(euro)"] <= 1e5)]
    df = df[df.Year >= 1971]
    df = df.reset_index(drop=True)

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
