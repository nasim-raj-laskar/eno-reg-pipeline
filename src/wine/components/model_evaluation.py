import os
import mlflow
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow.sklearn
import numpy as np
import joblib
import json
from urllib.parse import urlparse

from src.datascience.constants import *
from src.datascience.entity.config_entity import ModelEvaluationConfig
from src.datascience.utils.common import read_yaml,create_directories,save_json
from src.datascience import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
    
    def evaluate_model(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        mse = mean_squared_error(actual, pred)
        return rmse, mae, r2, mse
    
    def log_into_mlflow(self):
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
        
        # Load model comparison results
        comparison_file = os.path.join(os.path.dirname(self.config.model_path), 'model_comparison.json')
        model_results = {}
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                model_results = json.load(f)

        test_x = test_data.drop(columns=[self.config.target_column], axis=1)
        test_y = test_data[self.config.target_column].values

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2, mse) = self.evaluate_model(test_y, predicted_qualities)

            # Save final model metrics
            scores = {
                "best_model": model_results.get('best_model', 'Unknown'),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "mse": mse,
                "model_comparison": model_results
            }
            save_json(path=Path(self.config.metric_file_name), data=scores)

            # Log best model info
            if 'best_model' in model_results:
                mlflow.log_param("best_model_type", model_results['best_model'])
                best_model_metrics = model_results.get(model_results['best_model'], {})
                if best_model_metrics:
                    mlflow.log_params(best_model_metrics.get('best_params', {}))

            # Log final metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mse", mse)
            
            # Log all model comparison results
            for model_name, metrics in model_results.items():
                if model_name != 'best_model' and isinstance(metrics, dict):
                    mlflow.log_metric(f"{model_name}_test_mse", metrics.get('test_mse', 0))
                    mlflow.log_metric(f"{model_name}_test_r2", metrics.get('test_r2', 0))
                    mlflow.log_metric(f"{model_name}_test_mae", metrics.get('test_mae', 0))

            # Register model with appropriate name
            model_name = model_results.get('best_model', 'BestModel')
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name=f"{model_name}WineQuality")
            else:
                mlflow.sklearn.log_model(model, "model")
                
            logger.info(f"Model evaluation completed. Best model: {model_results.get('best_model', 'Unknown')}")