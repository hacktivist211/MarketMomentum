import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score
)
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_test_data(file_path, scaler):
    """
    Load and preprocess the test data for model evaluation.
    
    Parameters:
    -----------
    file_path : str
        Path to the test dataset CSV file
    scaler : StandardScaler
        Pre-fitted scaler to transform features
    
    Returns:
    --------
    tuple: (preprocessed_data, feature_names)
        Preprocessed test data and list of features to use
    """
    print("Loading test data...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        return None, None
    
    print("Preprocessing test data...")
    
    # Ensure 'time' column is in datetime format
    data['time'] = pd.to_datetime(data['date'])
    
    # Feature names to use for scaling and prediction
    feature_names = [
        'open', 'high', 'low', 'volume',
        'daily_return', 'volatility', 'rsi',
        'macd', 'signal', 'company_encoded',
        'sin_day_of_week', 'cos_day_of_week',
        'sin_month', 'cos_month',
        'price_to_sma_50', 'price_to_sma_200',
        'normalized_momentum', 'momentum_oscillator',
        'trend_strength', 'trend_direction',
        'bollinger_percent_b', 'opening_gap',
        'price_range_percentage', 'volume_trend',
        'ema_12', 'ema_26', 'ema_50', 'ema_100',
        'macd_histogram', 'volume_ratio',
        'is_month_start', 'is_month_end',
        'price_change', 'sma_50', 'sma_200',
        'ema_50', 'ema_100', 'momentum',
        'volume_change'
    ]
    
    # Verify feature existence and remove missing features
    missing_features = [f for f in feature_names if f not in data.columns]
    if missing_features:
        print(f"Warning: Missing features in the dataset: {missing_features}")
        feature_names = [f for f in feature_names if f in data.columns]
    
    # Ensure all required features exist
    if not feature_names:
        print("Error: No valid features found in the dataset.")
        return None, None
    
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Scale features
    try:
        data[feature_names] = scaler.transform(data[feature_names])
    except Exception as e:
        print(f"Error during feature scaling: {e}")
        return None, None
    
    return data, feature_names

def evaluate_model(model_path, test_file_path, scaler_path):
    """
    Evaluate the trained model on unseen test data.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved machine learning model
    test_file_path : str
        Path to the test dataset
    scaler_path : str
        Path to the saved feature scaler
    """
    # Load the model
    print("Loading trained model...")
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load the scaler
    print("Loading scaler...")
    try:
        scaler = joblib.load(scaler_path)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return
    
    # Load and preprocess test data
    test_data, features = load_and_preprocess_test_data(test_file_path, scaler)
    
    if test_data is None or features is None:
        print("Data preprocessing failed.")
        return
    
    # Extract features and target
    X_test = test_data[features]
    y_test = test_data['close']
    
    # Make predictions
    print("Making predictions...")
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error during prediction: {e}")
        return
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print performance metrics
    print("\nModel Performance Metrics on Unseen Data:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    
    # Display predictions vs actuals
    print("\nFirst 10 Predictions vs Actual Values:")
    for i in range(min(10, len(y_pred))):
        print(f"Prediction: {y_pred[i]:.4f}, Actual: {y_test.iloc[i]:.4f}")

def main():
    """
    Main execution function to run model evaluation.
    """
    # Paths to files - REPLACE THESE WITH YOUR ACTUAL PATHS
    model_path = r"C:\Users\Niraj\Documents\Projects\StockPrediction\XGBoost\final_xgboost_model.joblib"
    test_file_path = r"C:\Users\Niraj\Documents\Projects\StockPrediction\test_dataset.csv"
    scaler_path = r"C:\Users\Niraj\Documents\Projects\StockPrediction\XGBoost\scaler.joblib"
    
    # Evaluate the model
    evaluate_model(model_path, test_file_path, scaler_path)

# ============================
# Application Entry Point
# ============================
if __name__ == "__main__":
    main()