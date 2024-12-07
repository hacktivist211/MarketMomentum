import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import datetime
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
from sklearn.decomposition import PCA
from tensorflow.keras.regularizers import l1_l2

def advanced_feature_engineering(data):
    """Add more sophisticated features"""
    # Time-based features
    data['time'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    
    # Advanced price-based features
    data['price_change'] = data['close'] - data['open']
    data['daily_return'] = data['close'].pct_change()
    data['volatility'] = data['daily_return'].rolling(window=5).std()
    
    # Trend indicators
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    data['trend_strength'] = data['sma_50'] - data['sma_200']
    
    # Momentum indicators
    data['momentum'] = data['close'].pct_change(periods=14)
    
    # Exponential moving averages
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    
    # Volume-based features
    data['volume_change'] = data['volume'].pct_change()
    data['volume_mavg'] = data['volume'].rolling(window=5).mean()
    
    # Technical indicators
    data['rsi'] = calculate_rsi(data['close'], window=14)
    data['macd'], data['signal'] = calculate_macd(data['close'])
    
    # Bollinger Bands
    data['rolling_std'] = data['close'].rolling(window=20).std()
    data['upper_bollinger'] = data['sma_50'] + (2 * data['rolling_std'])
    data['lower_bollinger'] = data['sma_50'] - (2 * data['rolling_std'])
    
    # Percentage change features
    data['pct_change_high'] = (data['high'] - data['close']) / data['close']
    data['pct_change_low'] = (data['low'] - data['close']) / data['close']
    
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Categorical encoding for company
    data['company_encoded'] = pd.Categorical(data['company']).codes
    
    # Encode cyclical features
    data['sin_day_of_week'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['cos_day_of_week'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
    
    return data

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, slow=26, fast=12, signal=9):
    """Calculate Moving Average Convergence Divergence"""
    exp1 = prices.ewm(span=fast, adjust=False).mean()
    exp2 = prices.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def load_and_preprocess_data(file_path):
    """
    Enhanced data loading and preprocessing with feature engineering
    """
    print("Loading data...")
    try:
        # Use pandas chunking to show progress
        data_chunks = pd.read_csv(file_path, chunksize=10000)
        
        # Combine chunks with progress bar
        full_data = []
        total_rows = sum(1 for _ in open(file_path)) - 1  # Count total rows, subtract header
        
        with tqdm(total=total_rows, desc="Loading Rows", unit="row") as pbar:
            for chunk in data_chunks:
                full_data.append(chunk)
                pbar.update(len(chunk))
        
        # Combine all chunks
        data = pd.concat(full_data, ignore_index=True)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        exit()
    
    print("\nPreprocessing data...")
    
    # Apply advanced feature engineering
    data = advanced_feature_engineering(data)
    
    # Create a MinMaxScaler for non-price features
    non_price_scaler = MinMaxScaler()
    
    # Comprehensive list of features for scaling
    feature_columns = [
        'open', 'high', 'low', 'close', 'volume', 
        'daily_return', 'volatility', 'rsi', 'macd', 
        'sma_50', 'sma_200', 'trend_strength', 
        'momentum', 'ema_12', 'ema_26',
        'volume_change', 'volume_mavg',
        'rolling_std', 'upper_bollinger', 'lower_bollinger'
    ]
    
    # Separate scaling for features
    data[feature_columns] = StandardScaler().fit_transform(data[feature_columns])
    
    print("\nData processed.")
    return data, non_price_scaler

def define_dataset(data):
    """
    Prepare the dataset for training with comprehensive features
    """
    print("Defining dataset...")
    
    # More comprehensive feature selection
    features = [
        'open', 'high', 'low', 'volume', 
        'daily_return', 'volatility', 'rsi', 
        'macd', 'signal', 'company_encoded', 
        'sin_day_of_week', 'cos_day_of_week', 
        'sin_month', 'cos_month',
        'pct_change_high', 'pct_change_low',
        'sma_50', 'sma_200', 'trend_strength', 
        'momentum', 'ema_12', 'ema_26',
        'volume_change', 'volume_mavg',
        'rolling_std', 'upper_bollinger', 'lower_bollinger'
    ]
    
    X = data[features].values
    y = data['close'].values
    
    print("Splitting dataset...")
    # Stratified time-series split to preserve temporal information
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=0.95)  # Retain 95% of variance
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    
    print(f"Dataset split completed:")
    print(f"Training data: {X_train_pca.shape[0]} rows")
    print(f"Validation data: {X_val_pca.shape[0]} rows")
    print(f"Test data: {X_test_pca.shape[0]} rows")
    
    return (X_train_pca, X_val_pca, X_test_pca, y_train, y_val, y_test)

def build_more_complex_model(input_shape):
    """
    More sophisticated neural network architecture
    """
    model = keras.Sequential([
        # Input layer
        layers.InputLayer(input_shape=input_shape),
        
        # More complex feature extraction
        layers.Dense(512, activation='selu', 
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        # Additional layers with increased complexity
        layers.Dense(256, activation='selu', 
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='selu', 
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='selu', 
                     kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)),
        layers.BatchNormalization(),
        
        # Output layer
        layers.Dense(1, activation='linear')
    ])
    
    # Use standard Adam optimizer
    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Robust loss function
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(), 
            tf.keras.metrics.RootMeanSquaredError(),
            'mae'  # Mean Absolute Error
        ]
    )
    
    return model

def train_model(X_train, X_val, X_test, y_train, y_val, y_test, 
                batch_size=128, max_epochs=100):
    """
    Enhanced model training with multiple callbacks and advanced techniques
    """
    print("Building advanced model...")
    
    model = build_more_complex_model(input_shape=(X_train.shape[1],))
    
    # More aggressive early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=25,
        restore_best_weights=True,
        min_delta=0.0001
    )
    
    # Model checkpoint with more flexibility
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        f"models/best_model_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.h5", 
        monitor='val_loss', 
        save_best_only=True,
        mode='min'
    )
    
    # Advanced learning rate scheduling
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    os.makedirs("models", exist_ok=True)

    print("Training advanced model...")
    start_time = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[early_stop, model_checkpoint, reduce_lr],
        verbose=1
    )
    
    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time / 60:.2f} minutes.")

    # Advanced model evaluation
    print("Evaluating model on test data...")
    test_metrics = model.evaluate(X_test, y_test)
    
    y_pred = model.predict(X_test).flatten()
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print("\nTest Metrics:")
    print(f"Test Loss: {test_metrics[0]:.4f}")
    print(f"Mean Absolute Error: {test_metrics[1]:.4f}")
    print(f"Root Mean Squared Error: {test_metrics[2]:.4f}")
    print(f"Model Accuracy (1 - MAPE): {100 - mape:.2f}%")

    # Optional: Ensemble Prediction
    def create_ensemble_models(X_train, X_val, X_test, y_train, y_val, y_test, n_models=5):
        ensemble_models = []
        ensemble_predictions = []
        
        for i in range(n_models):
            # Create slightly different models
            ensemble_model = build_more_complex_model(input_shape=(X_train.shape[1],))
            ensemble_model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=75,
                batch_size=batch_size,
                verbose=0
            )
            ensemble_models.append(ensemble_model)
            ensemble_predictions.append(ensemble_model.predict(X_test))
        
        # Ensemble prediction (average)
        final_ensemble_prediction = np.mean(ensemble_predictions, axis=0)
        
        # Evaluate ensemble performance
        ensemble_mape = np.mean(np.abs((y_test - final_ensemble_prediction.flatten()) / y_test)) * 100
        print(f"\nEnsemble Model Accuracy: {100 - ensemble_mape:.2f}%")
        
        return ensemble_models, final_ensemble_prediction

    # Create ensemble models (optional)
    ensemble_models, ensemble_predictions = create_ensemble_models(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    return model, history

if __name__ == "__main__":
    file_path = r"E:\CodeFusion\combined_stock_data.csv"
    
    # Step 1: Load and preprocess the data
    combined_data, scaler = load_and_preprocess_data(file_path)
    
    # Step 2: Define the dataset
    X_train, X_val, X_test, y_train, y_val, y_test = define_dataset(combined_data)
    
    # Step 3: Train the model
    model, history = train_model(X_train, X_val, X_test, y_train, y_val, y_test, 
                                 batch_size=128, max_epochs=100)
    
    # Step 4: Save the model
    model_path = r"E:\CodeFusion\Projects\StockPrediction\advanced_ensemble_model.h5"
    model.save(model_path)
    print(f"Model saved to {model_path}")