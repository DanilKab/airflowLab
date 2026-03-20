import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
from mlflow.models import infer_signature
import joblib
from datetime import datetime
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TARGET_COLUMN = 'Price'


def scale_frame(frame):
    df = frame.copy()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    scaler = StandardScaler()
    power_trans = PowerTransformer()

    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))

    return X_scaled, y_scaled, scaler, power_trans


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def save_metrics(metrics, filename='/tmp/model_metrics.json'):
    with open(filename, 'w') as f:
        json.dump(metrics, f, indent=4)


def train():
    df = pd.read_csv("/tmp/cleaned_data.csv")

    X_scaled, y_scaled, scaler, power_trans = scale_frame(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=0.3, random_state=42
    )

    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'penalty': ["l1", "l2", "elasticnet"],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [False, True],
        'max_iter': [1000, 2000]
    }

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("linear_model_cars")

    with mlflow.start_run(run_name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4, scoring='r2', verbose=1)
        clf.fit(X_train, y_train.ravel())

        best_model = clf.best_estimator_

        y_pred_scaled = best_model.predict(X_val)
        y_pred_original = power_trans.inverse_transform(y_pred_scaled.reshape(-1, 1))
        y_val_original = power_trans.inverse_transform(y_val)

        rmse, mae, r2 = eval_metrics(y_val_original, y_pred_original)

        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'best_params': clf.best_params_,
            'timestamp': datetime.now().isoformat()
        }

        save_metrics(metrics)

        for param, value in clf.best_params_.items():
            mlflow.log_param(param, value)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        predictions = best_model.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best_model, "model", signature=signature)

        artifacts = {
            'model': best_model,
            'scaler': scaler,
            'power_transformer': power_trans
        }

        for name, artifact in artifacts.items():
            filename = f"/tmp/{name}.pkl"
            joblib.dump(artifact, filename)
            mlflow.log_artifact(filename)

        return True
