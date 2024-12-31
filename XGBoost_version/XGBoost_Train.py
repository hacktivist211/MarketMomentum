import os
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    mean_absolute_percentage_error, 
    r2_score
)
import matplotlib.pyplot as plt
from tqdm import tqdm

def advanced_feature_engineering(data):
    """Add comprehensive features with enhanced pattern recognition"""
    # Time-based features
    data['time'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    data['quarter'] = data['time'].dt.quarter
    data['is_month_start'] = (data['time'].dt.day == 1).astype(int)
    data['is_month_end'] = (data['time'].dt.day == data['time'].dt.days_in_month).astype(int)
    
    # Price-based features
    data['price_change'] = data['close'] - data['open']
    data['daily_return'] = data['close'].pct_change()
    data['volatility'] = data['daily_return'].rolling(window=5).std()
    
    # Trend and seasonality features
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    data['trend_strength'] = data['sma_50'] - data['sma_200']
    data['trend_direction'] = np.sign(data['trend_strength'])
    
    # Exponential moving averages
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_100'] = data['close'].ewm(span=100, adjust=False).mean()
    
    # Comparative features
    data['price_to_sma_50'] = data['close'] / data['sma_50']
    data['price_to_sma_200'] = data['close'] / data['sma_200']
    
    # Momentum and volatility indicators
    data['momentum'] = data['close'].pct_change(periods=14)
    data['momentum_oscillator'] = data['momentum'].rolling(window=9).mean()
    data['normalized_momentum'] = (data['momentum'] - data['momentum'].rolling(window=50).mean()) / data['momentum'].rolling(window=50).std()
    
    # Technical indicators
    data['rsi'] = calculate_rsi(data['close'], window=14)
    data['macd'], data['signal'] = calculate_macd(data['close'])
    data['macd_histogram'] = data['macd'] - data['signal']
    
    # Bollinger Bands
    data['rolling_std'] = data['close'].rolling(window=20).std()
    data['upper_bollinger'] = data['sma_50'] + (2 * data['rolling_std'])
    data['lower_bollinger'] = data['sma_50'] - (2 * data['rolling_std'])
    data['bollinger_width'] = data['upper_bollinger'] - data['lower_bollinger']
    data['bollinger_percent_b'] = (data['close'] - data['lower_bollinger']) / (data['upper_bollinger'] - data['lower_bollinger'])
    
    # Gap and price range analysis
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['price_range_percentage'] = (data['high'] - data['low']) / data['close']
    
    # Volume-based features
    data['volume_change'] = data['volume'].pct_change()
    data['volume_mavg'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_mavg']
    data['volume_trend'] = data['volume'].ewm(span=10, adjust=False).mean() / data['volume_mavg']
    
    # Cyclical and seasonal features
    data['sin_day_of_week'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['cos_day_of_week'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Categorical encoding
    data['company_encoded'] = pd.Categorical(data['company']).codes
    
    # Fill missing values using ffill and bfill methods
    data = data.ffill().bfill()
    
    return data

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index with more robust calculation"""
    delta = prices.diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    relative_strength = gain / loss
    rsi = 100.0 - (100.0 / (1.0 + relative_strength))
    
    return rsi

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate Moving Average Convergence Divergence with enhanced filtering"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock data from CSV file
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading the CSV file: {e}")
        raise
    
    # Basic data validation
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'company']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Sort data by date
    df = df.sort_values('date')
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Handle missing values
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].ffill().bfill()
    
    # If any missing values remain, fill with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Apply advanced feature engineering
    df = advanced_feature_engineering(df)
    
    # Optional: Log transformation for volume to handle skewness
    df['volume'] = np.log1p(df['volume'])
    
    # Remove rows with infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    print("Columns:", list(df.columns))
    
    return df

def define_xgboost_dataset(data):
    """
    Advanced dataset preparation for XGBoost
    """
    features = [
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
        'price_range_percentage', 'volume_change'
    ]
    
    X = data[features].values
    y = data['close'].values
    
    # Stratified time-series split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Save the scaler
    scaler_path = r"{path for scaler}"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    print(f"Dataset split completed:")
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"Validation data: {X_val.shape[0]} samples")
    print(f"Test data: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def train_xgboost_model(
    X_train, X_val, X_test, 
    y_train, y_val, y_test,
    feature_names=None
):
    """
    Train and evaluate XGBoost model with advanced hyperparameters and detailed progress tracking
    """
    print("Configuring XGBoost model...")
    
    # Advanced Progress Tracking Callback
    class AdvancedProgressCallback:
        def __init__(self, total_epochs):
            self.total_epochs = total_epochs
            self.progress_bar = None
            self.start_time = time.time()
            self.best_val_loss = float('inf')
        
        def __call__(self, env):
            if self.progress_bar is None:
                self.progress_bar = tqdm(
                    total=self.total_epochs, 
                    desc='🚀 XGBoost Training', 
                    unit='epoch', 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, 🔥{postfix}]',
                    ncols=100,  # Fixed width for better readability
                    colour='green'  # Colorful progress bar
                )
            
            # Current epoch and loss values
            train_loss = env.evaluation_result_list[0][1]
            val_loss = env.evaluation_result_list[1][1]
            
            # Track best validation loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                improvement_marker = '📈'
            else:
                improvement_marker = ''
            
            # Estimate total training time
            elapsed_time = time.time() - self.start_time
            estimated_total_time = (elapsed_time / (env.iteration + 1)) * self.total_epochs
            remaining_time = estimated_total_time - elapsed_time
            
            # Update progress bar with enhanced metrics
            self.progress_bar.set_postfix_str(
                f'Train:{train_loss:.4f}, Val:{val_loss:.4f} {improvement_marker} '
                f'ETA:{remaining_time/60:.1f}m',
                refresh=False
            )
            self.progress_bar.update(1)
            
            # Close progress bar when training completes
            if env.iteration == self.total_epochs - 1:
                self.progress_bar.close()
    
    # XGBoost model configuration - Updated to remove GPU-specific settings
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.01,
        max_depth=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=50
    )
    
    # Progress tracking callback
    progress_callback = AdvancedProgressCallback(total_epochs=50)
    
    # Time the training
    start_time = time.time()
    
    # Fit the model with progress tracking
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=True  # This will show progress
    )
    
    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time / 60:.2f} minutes.")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Advanced evaluation metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate accuracy (for regression, this is R-squared)
    accuracy = r2 * 100  # Convert R-squared to percentage
    
    print("\nDetailed Model Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    print(f"Model Accuracy: {accuracy:.2f}%")
    
    # Feature importance visualization
    feature_importance = model.feature_importances_
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(len(feature_importance))]
    
    # Sort features by importance
    feature_sorted_idx = np.argsort(feature_importance)[::-1]
    sorted_features = [feature_names[i] for i in feature_sorted_idx]
    sorted_importance = feature_importance[feature_sorted_idx]
    
    # Visualization of feature importance
    plt.figure(figsize=(15,6))
    plt.bar(range(len(sorted_importance[:10])), sorted_importance[:10])
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(len(sorted_importance[:10])), sorted_features[:10], rotation=45, ha='right')
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(r'{path for directory}', exist_ok=True)
    
    plt.savefig(r'{path for graph}')
    plt.close()
    
    # Prediction vs Actual Plot
    plt.figure(figsize=(15,6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('Stock Price Prediction: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'{path for prediction comparision}')
    plt.close()
    
    print("\nTop 10 Most Important Features:")
    for feat, imp in zip(sorted_features[:10], sorted_importance[:10]):
        print(f"{feat}: {imp:.4f}")
    
    # Save the model
    import joblib
    model_path = r'{model path}'
    joblib.dump(model, model_path)
    print(f"Final model saved to {model_path}")
    
    return model

def main():
    """
    Main execution function to orchestrate the entire process
    """
    # File path for your stock data
    file_path = r"E:\CodeFusion\combined_stock_data.csv"
    
    # Load and preprocess data
    processed_data = load_and_preprocess_data(file_path)
    
    # Feature names for feature importance
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
        'price_range_percentage', 'volume_change'
    ]
    
    # Prepare dataset for XGBoost
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = define_xgboost_dataset(processed_data)
    
    # Train the model
    model = train_xgboost_model(
        X_train, X_val, X_test, 
        y_train, y_val, y_test,
        feature_names=feature_names
    )

if __name__ == "__main__":
    main()
