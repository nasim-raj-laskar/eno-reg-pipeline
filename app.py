from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import os
import pandas as pd
import numpy as np
from src.datascience.pipeline.prediction_pipline import PredictionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'wine_quality_predictor_2024'

@app.route('/', methods=['GET'])
def homepage():
    return render_template('index.html')
   
@app.route('/train', methods=['GET'])
def training():
    try:
        logger.info("Starting model training...")
        os.system("python main.py")
        flash("Model training completed successfully!", "success")
        return redirect(url_for('homepage'))
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        flash(f"Training failed: {str(e)}", "error")
        return redirect(url_for('homepage'))

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Extract and validate input data
            features = [
                'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar',
                'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 
                'density', 'pH', 'sulphates', 'alcohol'
            ]
            
            data = []
            for feature in features:
                value = request.form.get(feature)
                if not value:
                    raise ValueError(f"Missing value for {feature}")
                data.append(float(value))
            
            # Basic validation
            if data[8] < 0 or data[8] > 14:  # pH should be between 0-14
                raise ValueError("pH value should be between 0 and 14")
            if data[10] < 0 or data[10] > 100:  # Alcohol should be reasonable
                raise ValueError("Alcohol percentage should be between 0 and 100")
            
            # Reshape data for prediction
            data_array = np.array(data).reshape(1, 11)
            
            # Make prediction
            obj = PredictionPipeline()
            prediction = obj.predict(data_array)
            
            # Round prediction to 1 decimal place
            prediction_value = round(float(prediction[0]), 1)
            
            logger.info(f"Prediction made: {prediction_value}")
            return render_template('results.html', prediction=prediction_value)
        
        except ValueError as ve:
            logger.warning(f"Validation error: {str(ve)}")
            flash(f"Input validation error: {str(ve)}", "error")
            return redirect(url_for('homepage'))
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            flash(f"Prediction failed: {str(e)}", "error")
            return redirect(url_for('homepage'))
    else:
        return render_template('index.html')

@app.errorhandler(404)
def not_found(error):
    return render_template('index.html'), 404

@app.errorhandler(500)
def internal_error(error):
    flash("An internal error occurred. Please try again.", "error")
    return render_template('index.html'), 500
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)