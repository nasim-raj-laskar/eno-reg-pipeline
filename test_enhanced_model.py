#!/usr/bin/env python3
"""
Test script for enhanced model training with hyperparameter tuning
"""

import sys
import os
sys.path.append('src')

from src.datascience.config.configuration import ConfigurationManager
from src.datascience.components.model_trainer import ModelTrainer
from src.datascience import logger
import json

def test_enhanced_training():
    """Test the enhanced model training with multiple algorithms"""
    try:
        # Initialize configuration
        config_manager = ConfigurationManager()
        model_trainer_config = config_manager.get_model_trainer_config()
        
        # Initialize and train models
        logger.info("Starting enhanced model training with hyperparameter tuning...")
        model_trainer = ModelTrainer(config=model_trainer_config)
        model_trainer.train()
        
        # Check if model comparison file was created
        comparison_file = os.path.join(model_trainer_config.root_dir, 'model_comparison.json')
        if os.path.exists(comparison_file):
            with open(comparison_file, 'r') as f:
                results = json.load(f)
            
            print("\n" + "="*50)
            print("MODEL COMPARISON RESULTS")
            print("="*50)
            
            for model_name, metrics in results.items():
                if model_name != 'best_model' and isinstance(metrics, dict):
                    print(f"\n{model_name}:")
                    print(f"  Best Parameters: {metrics.get('best_params', {})}")
                    print(f"  CV Score: {metrics.get('cv_score', 0):.4f}")
                    print(f"  Test MSE: {metrics.get('test_mse', 0):.4f}")
                    print(f"  Test R²: {metrics.get('test_r2', 0):.4f}")
                    print(f"  Test MAE: {metrics.get('test_mae', 0):.4f}")
            
            print(f"\nBEST MODEL: {results.get('best_model', 'Unknown')}")
            print("="*50)
            
        logger.info("Enhanced model training completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced model training failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_enhanced_training()
    if success:
        print("\n✅ Enhanced model training test PASSED!")
    else:
        print("\n❌ Enhanced model training test FAILED!")
        sys.exit(1)