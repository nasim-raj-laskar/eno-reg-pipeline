# Enological Quality Regression Framework

## Architecture
**Heterogeneous ensemble ML pipeline** implementing automated hyperparameter optimization via exhaustive grid search across regularized linear (ElasticNet), bagging (RandomForest), and gradient boosting (XGBoost) regressors. Employs stratified k-fold cross-validation (k=5) with negative MSE objective for model selection.

## ETL & Training Pipeline
```
Data Ingestion → Schema Validation → Feature Engineering → Ensemble Training → Model Selection → Inference Deployment
```

## Technical Implementation
- **Regressors**: ElasticNet (L1/L2 elastic regularization), RandomForest (bootstrap aggregating), XGBoost (gradient tree boosting)
- **Hyperparameter Optimization**: Exhaustive grid search with cross-validated performance estimation
- **Validation Framework**: Schema-driven data contracts, statistical performance metrics (RMSE, MAE, R²-score)
- **MLOps Stack**: MLflow experiment tracking, model registry, artifact lineage
- **Inference Engine**: Flask WSGI application with JSON schema validation

## Configuration Management
- `config.yaml`: Artifact storage paths, pipeline orchestration
- `params.yaml`: Hyperparameter search space definitions
- `schema.yaml`: Feature type contracts, data validation rules

## Pipeline Execution
```bash
# End-to-end training pipeline
python main.py

# Model performance benchmarking
python test_enhanced_model.py

# Production inference server
python app.py
```

## Model Selection Algorithm
Automated model selection via test set MSE minimization. Serialized results in `artifacts/model_trainer/model_comparison.json` containing cross-validation statistics and optimal hyperparameter configurations.

## REST API Specification
- `GET /`: Frontend interface
- `POST /predict`: Feature vector ingestion → quality prediction [0,10]
- `GET /train`: Asynchronous pipeline trigger