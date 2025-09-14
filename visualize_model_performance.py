#!/usr/bin/env python3
"""
Model Performance Visualization Dashboard
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def load_model_results():
    """Load model comparison results"""
    comparison_file = Path("artifacts/model_trainer/model_comparison.json")
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            return json.load(f)
    return {}

def create_performance_dashboard():
    """Generate comprehensive model performance visualizations"""
    
    # Load data and model
    test_data = pd.read_csv("artifacts/data_transformation/test.csv")
    model = joblib.load("artifacts/model_trainer/model.joblib")
    results = load_model_results()
    
    # Prepare data
    X_test = test_data.drop('quality', axis=1)
    y_test = test_data['quality']
    y_pred = model.predict(X_test)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Performance Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Model Comparison Bar Chart
    ax1 = axes[0, 0]
    models = [k for k in results.keys() if k != 'best_model' and isinstance(results[k], dict)]
    mse_scores = [results[model]['test_mse'] for model in models]
    r2_scores = [results[model]['test_r2'] for model in models]
    
    x_pos = np.arange(len(models))
    ax1_twin = ax1.twinx()
    
    bars1 = ax1.bar(x_pos - 0.2, mse_scores, 0.4, label='MSE', color='lightcoral', alpha=0.8)
    bars2 = ax1_twin.bar(x_pos + 0.2, r2_scores, 0.4, label='R²', color='skyblue', alpha=0.8)
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('MSE', color='red')
    ax1_twin.set_ylabel('R² Score', color='blue')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(models, rotation=45)
    
    # Add value labels on bars
    for bar, score in zip(bars1, mse_scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Actual vs Predicted Scatter Plot
    ax2 = axes[0, 1]
    ax2.scatter(y_test, y_pred, alpha=0.6, color='darkblue')
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax2.set_xlabel('Actual Quality')
    ax2.set_ylabel('Predicted Quality')
    ax2.set_title('Actual vs Predicted Values')
    ax2.text(0.05, 0.95, f'R² = {r2_score(y_test, y_pred):.3f}', 
             transform=ax2.transAxes, bbox=dict(boxstyle="round", facecolor='wheat'))
    
    # 3. Residuals Plot
    ax3 = axes[0, 2]
    residuals = y_test - y_pred
    ax3.scatter(y_pred, residuals, alpha=0.6, color='green')
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Analysis')
    
    # 4. Feature Importance (if available)
    ax4 = axes[1, 0]
    if hasattr(model, 'feature_importances_'):
        feature_names = X_test.columns
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        ax4.barh(range(len(indices)), importances[indices], color='orange', alpha=0.7)
        ax4.set_yticks(range(len(indices)))
        ax4.set_yticklabels([feature_names[i] for i in indices])
        ax4.set_xlabel('Feature Importance')
        ax4.set_title('Top 10 Feature Importances')
    else:
        ax4.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Feature Importance')
    
    # 5. Distribution of Predictions
    ax5 = axes[1, 1]
    ax5.hist(y_test, bins=20, alpha=0.7, label='Actual', color='blue', density=True)
    ax5.hist(y_pred, bins=20, alpha=0.7, label='Predicted', color='red', density=True)
    ax5.set_xlabel('Wine Quality Score')
    ax5.set_ylabel('Density')
    ax5.set_title('Distribution Comparison')
    ax5.legend()
    
    # 6. Cross-Validation Scores
    ax6 = axes[1, 2]
    cv_scores = [results[model]['cv_score'] for model in models]
    colors = ['gold', 'lightgreen', 'lightblue']
    
    bars = ax6.bar(models, cv_scores, color=colors[:len(models)], alpha=0.8)
    ax6.set_ylabel('CV Score (neg_MSE)')
    ax6.set_title('Cross-Validation Performance')
    ax6.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, score in zip(bars, cv_scores):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('artifacts/model_evaluation/performance_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Best Model: {results.get('best_model', 'Unknown')}")
    print(f"Test RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")
    print(f"Test R²: {r2_score(y_test, y_pred):.4f}")
    print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
    print("="*60)

if __name__ == "__main__":
    create_performance_dashboard()