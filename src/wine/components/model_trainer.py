import pandas as pd
import os
import numpy as np
from src.datascience import logger
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import joblib
import json
from src.datascience.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.best_model = None
        self.best_score = float('inf')
        self.model_results = {}
    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop(self.config.target_column, axis=1)
        test_x = test_data.drop(self.config.target_column, axis=1)
        train_y = train_data[self.config.target_column].values
        test_y = test_data[self.config.target_column].values

        # Train multiple models with hyperparameter tuning
        self._train_elasticnet(train_x, train_y, test_x, test_y)
        self._train_random_forest(train_x, train_y, test_x, test_y)
        self._train_xgboost(train_x, train_y, test_x, test_y)
        
        # Save best model and results
        joblib.dump(self.best_model, os.path.join(self.config.root_dir, self.config.model_name))
        
        # Save model comparison results
        with open(os.path.join(self.config.root_dir, 'model_comparison.json'), 'w') as f:
            json.dump(self.model_results, f, indent=2)
        
        logger.info(f"Best model: {self.model_results['best_model']} with MSE: {self.best_score:.4f}")
    
    def _train_elasticnet(self, train_x, train_y, test_x, test_y):
        param_grid = {
            'alpha': self.config.elasticnet_params['alpha'],
            'l1_ratio': self.config.elasticnet_params['l1_ratio']
        }
        
        model = ElasticNet(random_state=self.config.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=self.config.cv_folds, 
                                 scoring=self.config.scoring, n_jobs=-1)
        grid_search.fit(train_x, train_y)
        
        # Evaluate on test set
        test_pred = grid_search.predict(test_x)
        mse = mean_squared_error(test_y, test_pred)
        
        self.model_results['ElasticNet'] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_mse': mse,
            'test_r2': r2_score(test_y, test_pred),
            'test_mae': mean_absolute_error(test_y, test_pred)
        }
        
        if mse < self.best_score:
            self.best_score = mse
            self.best_model = grid_search.best_estimator_
            self.model_results['best_model'] = 'ElasticNet'
    
    def _train_random_forest(self, train_x, train_y, test_x, test_y):
        param_grid = {
            'n_estimators': self.config.random_forest_params['n_estimators'],
            'max_depth': self.config.random_forest_params['max_depth'],
            'min_samples_split': self.config.random_forest_params['min_samples_split'],
            'min_samples_leaf': self.config.random_forest_params['min_samples_leaf']
        }
        
        model = RandomForestRegressor(random_state=self.config.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=self.config.cv_folds,
                                 scoring=self.config.scoring, n_jobs=-1)
        grid_search.fit(train_x, train_y)
        
        test_pred = grid_search.predict(test_x)
        mse = mean_squared_error(test_y, test_pred)
        
        self.model_results['RandomForest'] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_mse': mse,
            'test_r2': r2_score(test_y, test_pred),
            'test_mae': mean_absolute_error(test_y, test_pred)
        }
        
        if mse < self.best_score:
            self.best_score = mse
            self.best_model = grid_search.best_estimator_
            self.model_results['best_model'] = 'RandomForest'
    
    def _train_xgboost(self, train_x, train_y, test_x, test_y):
        param_grid = {
            'n_estimators': self.config.xgboost_params['n_estimators'],
            'max_depth': self.config.xgboost_params['max_depth'],
            'learning_rate': self.config.xgboost_params['learning_rate'],
            'subsample': self.config.xgboost_params['subsample']
        }
        
        model = xgb.XGBRegressor(random_state=self.config.random_state)
        grid_search = GridSearchCV(model, param_grid, cv=self.config.cv_folds,
                                 scoring=self.config.scoring, n_jobs=-1)
        grid_search.fit(train_x, train_y)
        
        test_pred = grid_search.predict(test_x)
        mse = mean_squared_error(test_y, test_pred)
        
        self.model_results['XGBoost'] = {
            'best_params': grid_search.best_params_,
            'cv_score': grid_search.best_score_,
            'test_mse': mse,
            'test_r2': r2_score(test_y, test_pred),
            'test_mae': mean_absolute_error(test_y, test_pred)
        }
        
        if mse < self.best_score:
            self.best_score = mse
            self.best_model = grid_search.best_estimator_
            self.model_results['best_model'] = 'XGBoost'