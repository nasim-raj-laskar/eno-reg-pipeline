import joblib
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from src.datascience import logger

class PredictionPipeline:
    def __init__(self):
        self.model_path = Path("artifacts/model_trainer/model.joblib")
        self.comparison_path = Path("artifacts/model_trainer/model_comparison.json")
        
        try:
            self.model = joblib.load(self.model_path)
            self.model_info = self._load_model_info()
            logger.info(f"Loaded model: {self.model_info.get('best_model', 'Unknown')}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_model_info(self):
        """Load model comparison information if available"""
        try:
            if self.comparison_path.exists():
                with open(self.comparison_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load model info: {str(e)}")
        return {}
    
    def predict(self, data):
        """Make prediction with input validation"""
        try:
            # Validate input
            if data is None or len(data) == 0:
                raise ValueError("Input data cannot be empty")
            
            # Ensure data is numpy array
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Check feature count (should be 11 for wine dataset)
            if data.shape[1] != 11:
                raise ValueError(f"Expected 11 features, got {data.shape[1]}")
            
            prediction = self.model.predict(data)
            
            # Ensure prediction is within reasonable range (wine quality 0-10)
            prediction = np.clip(prediction, 0, 10)
            
            logger.info(f"Prediction made successfully: {prediction}")
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_model_info(self):
        """Return information about the loaded model"""
        return {
            'model_type': self.model_info.get('best_model', 'Unknown'),
            'model_path': str(self.model_path),
            'available': self.model_path.exists()
        }
    
