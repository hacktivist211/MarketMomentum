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
from tensorflow.keras.layers import MultiHeadAttention

def advanced_feature_engineering(data):
    """Add more sophisticated features with enhanced pattern recognition"""
    # Existing time-based and advanced features from previous implementation
    data['time'] = pd.to_datetime(data['date'])
    data['day_of_week'] = data['time'].dt.dayofweek
    data['month'] = data['time'].dt.month
    data['quarter'] = data['time'].dt.quarter
    data['is_month_start'] = (data['time'].dt.day == 1).astype(int)
    data['is_month_end'] = (data['time'].dt.day == data['time'].dt.days_in_month).astype(int)
    
    # Enhanced price-based features with more complex pattern recognition
    data['price_change'] = data['close'] - data['open']
    data['daily_return'] = data['close'].pct_change()
    data['volatility'] = data['daily_return'].rolling(window=5).std()
    
    # Advanced trend and seasonality features
    data['sma_50'] = data['close'].rolling(window=50).mean()
    data['sma_200'] = data['close'].rolling(window=200).mean()
    data['trend_strength'] = data['sma_50'] - data['sma_200']
    data['trend_direction'] = np.sign(data['trend_strength'])
    
    # Exponential moving averages for multiple timeframes
    data['ema_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['ema_50'] = data['close'].ewm(span=50, adjust=False).mean()
    data['ema_100'] = data['close'].ewm(span=100, adjust=False).mean()
    
    # Comparative features for pattern recognition
    data['price_to_sma_50'] = data['close'] / data['sma_50']
    data['price_to_sma_200'] = data['close'] / data['sma_200']
    
    # Advanced momentum and volatility indicators
    data['momentum'] = data['close'].pct_change(periods=14)
    data['momentum_oscillator'] = data['momentum'].rolling(window=9).mean()
    data['normalized_momentum'] = (data['momentum'] - data['momentum'].rolling(window=50).mean()) / data['momentum'].rolling(window=50).std()
    
    # More complex technical indicators
    data['rsi'] = calculate_rsi(data['close'], window=14)
    data['macd'], data['signal'] = calculate_macd(data['close'])
    data['macd_histogram'] = data['macd'] - data['signal']
    
    # Advanced Bollinger Bands with more context
    data['rolling_std'] = data['close'].rolling(window=20).std()
    data['upper_bollinger'] = data['sma_50'] + (2 * data['rolling_std'])
    data['lower_bollinger'] = data['sma_50'] - (2 * data['rolling_std'])
    data['bollinger_width'] = data['upper_bollinger'] - data['lower_bollinger']
    data['bollinger_percent_b'] = (data['close'] - data['lower_bollinger']) / (data['upper_bollinger'] - data['lower_bollinger'])
    
    # Enhanced gap and price range analysis
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['price_range_percentage'] = (data['high'] - data['low']) / data['close']
    
    # More advanced volume-based features
    data['volume_change'] = data['volume'].pct_change()
    data['volume_mavg'] = data['volume'].rolling(window=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_mavg']
    data['volume_trend'] = data['volume'].ewm(span=10, adjust=False).mean() / data['volume_mavg']
    
    # New advanced cyclical and seasonal features
    data['sin_day_of_week'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
    data['cos_day_of_week'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
    data['sin_month'] = np.sin(2 * np.pi * data['month'] / 12)
    data['cos_month'] = np.cos(2 * np.pi * data['month'] / 12)
    
    # Categorical encoding for additional context
    data['company_encoded'] = pd.Categorical(data['company']).codes
    
    # Fill missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
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

# Keep the load_and_preprocess_data function similar to the previous implementation
# with potentially minor modifications to the feature selection

def prepare_lstm_sequences(X, y, sequence_length=15):
    """
    Enhanced sequence preparation with multi-step sequence generation
    """
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i+sequence_length])
        y_sequences.append(y[i+sequence_length])
    
    return np.array(X_sequences), np.array(y_sequences)

def define_dataset(data, sequence_length=15):
    """
    Advanced dataset preparation with more comprehensive feature selection
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
        'macd_histogram', 'volume_ratio'
    ]
    
    X = data[features].values
    y = data['close'].values
    
    # Stratified time-series split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
    
    # Prepare sequences for LSTM
    X_train_seq, y_train_seq = prepare_lstm_sequences(X_train, y_train, sequence_length)
    X_val_seq, y_val_seq = prepare_lstm_sequences(X_val, y_val, sequence_length)
    X_test_seq, y_test_seq = prepare_lstm_sequences(X_test, y_test, sequence_length)
    
    print(f"Dataset split completed:")
    print(f"Training data: {X_train_seq.shape[0]} sequences")
    print(f"Validation data: {X_val_seq.shape[0]} sequences")
    print(f"Test data: {X_test_seq.shape[0]} sequences")
    
    return (X_train_seq, X_val_seq, X_test_seq, y_train_seq, y_val_seq, y_test_seq)

class AttentionLSTMModel(keras.Model):
    """
    Advanced LSTM model with multi-head self-attention and residual connections
    """
    def __init__(self, input_shape):
        super().__init__()
        
        # First LSTM layer with more units and regularization
        self.lstm1 = layers.LSTM(
            256, 
            return_sequences=True, 
            kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3),
            recurrent_regularizer=l1_l2(l1=1e-4, l2=1e-3)
        )
        self.batch_norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.4)
        
        # Multi-head self-attention layer
        self.multi_head_attention = MultiHeadAttention(
            num_heads=4, 
            key_dim=64, 
            dropout=0.3
        )
        
        # Second LSTM layer
        self.lstm2 = layers.LSTM(
            192, 
            return_sequences=True,
            kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3),
            recurrent_regularizer=l1_l2(l1=1e-4, l2=1e-3)
        )
        self.batch_norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.4)
        
        # Third LSTM layer
        self.lstm3 = layers.LSTM(
            128, 
            kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3),
            recurrent_regularizer=l1_l2(l1=1e-4, l2=1e-3)
        )
        self.batch_norm3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.3)
        
        # Dense layers for feature extraction
        self.dense1 = layers.Dense(
            64, 
            activation='selu', 
            kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)
        )
        self.batch_norm4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(0.2)
        
        self.dense2 = layers.Dense(
            32, 
            activation='selu', 
            kernel_regularizer=l1_l2(l1=1e-4, l2=1e-3)
        )
        self.batch_norm5 = layers.BatchNormalization()
        
        # Output layer
        self.output_layer = layers.Dense(1, activation='linear')
    
    def call(self, inputs):
        # First LSTM layer
        x = self.lstm1(inputs)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        
        # Multi-head self-attention
        x_attention = self.multi_head_attention(x, x, x)
        x = x + x_attention  # Residual connection
        
        # Second LSTM layer
        x = self.lstm2(x)
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        
        # Third LSTM layer
        x = self.lstm3(x)
        x = self.batch_norm3(x)
        x = self.dropout3(x)
        
        # Dense layers
        x = self.dense1(x)
        x = self.batch_norm4(x)
        x = self.dropout4(x)
        
        x = self.dense2(x)
        x = self.batch_norm5(x)
        
        # Output prediction
        return self.output_layer(x)

def train_lstm_model(
    X_train, X_val, X_test, 
    y_train, y_val, y_test, 
    batch_size=64, 
    max_epochs=200
):
    """
    Enhanced LSTM model training with advanced techniques
    """
    print("Building advanced LSTM model...")
    
    model = AttentionLSTMModel(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    # Advanced optimizer with adaptive learning rate and gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=1e-3,
        clipnorm=1.0  # Gradient clipping to prevent exploding gradients
    )
    
    model.compile(
        optimizer=optimizer,
        loss='huber',  # Robust loss function
        metrics=[
            tf.keras.metrics.MeanAbsoluteError(), 
            tf.keras.metrics.RootMeanSquaredError(),
            'mae'
        ]
    )
    
    # Advanced callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=50,
        restore_best_weights=True,
        min_delta=0.0001
    )
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        r'C:\Users\Niraj\Documents\Projects\StockPrediction\LSTM\best_lstm_attention_model_{epoch:02d}_{val_loss:.2f}.h5', 
        monitor='val_loss', 
        save_best_only=True,
        mode='min'
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=20,
        min_lr=1e-6,
        verbose=1
    )
    
    os.makedirs(r'C:\Users\Niraj\Documents\Projects\StockPrediction\LSTM', exist_ok=True)

    print("Training advanced LSTM model...")
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

    print("Evaluating model on test data...")
    test_metrics = model.evaluate(X_test, y_test)
    
    y_pred = model.predict(X_test).flatten()
    
    # Advanced evaluation metrics
    from sklearn.metrics import (
        mean_absolute_error, 
        mean_squared_error, 
        mean_absolute_percentage_error, 
        r2_score
    )
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print("\nDetailed Model Performance Metrics:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.4f}")
    print(f"R-squared Score: {r2:.4f}")
    
    # Visualization of predictions
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15,6))
    plt.plot(y_test, label='Actual Prices', color='blue')
    plt.plot(y_pred, label='Predicted Prices', color='red', alpha=0.7)
    plt.title('Stock Price Prediction: Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.tight_layout()
    plt.savefig(r'C:\Users\Niraj\Documents\Projects\StockPrediction\LSTM\prediction_comparison.png')
    plt.close()
    
    # Feature importance analysis
    def get_feature_importance(model, X, feature_names):
        """
        Basic feature importance approximation
        Note: This is a simplified approach and may not be as accurate as 
        feature importance in tree-based models
        """
        # Create a smaller subset of the data for importance calculation
        X_subset = X[:100]  # Use a subset to avoid computational complexity
        
        # Calculate gradients
        with tf.GradientTape() as tape:
            tape.watch(X_subset)
            predictions = model(X_subset)
        
        # Compute gradients
        gradients = tape.gradient(predictions, X_subset)
        
        # Calculate absolute importance
        feature_importance = np.abs(gradients).mean(axis=0).mean(axis=0)
        
        # Normalize
        feature_importance = feature_importance / np.sum(feature_importance)
        
        # Sort features by importance
        feature_sorted_idx = np.argsort(feature_importance)[::-1]
        
        return feature_importance[feature_sorted_idx], [feature_names[i] for i in feature_sorted_idx]
    
    # Define feature names (should match the features used in define_dataset)
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
        'macd_histogram', 'volume_ratio'
    ]
    
    # Get feature importance
    importance, sorted_features = get_feature_importance(model, X_test, feature_names)
    
    # Visualize feature importance
    plt.figure(figsize=(15,6))
    plt.bar(range(len(importance[:10])), importance[:10])
    plt.title('Top 10 Most Important Features')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.xticks(range(len(importance[:10])), sorted_features[:10], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(r'C:\Users\Niraj\Documents\Projects\StockPrediction\LSTM\feature_importance.png')
    plt.close()
    
    print("\nTop 10 Most Important Features:")
    for feat, imp in zip(sorted_features[:10], importance[:10]):
        print(f"{feat}: {imp:.4f}")
    
    # Save the final model
    final_model_path = r'C:\Users\Niraj\Documents\Projects\StockPrediction\LSTM\final_lstm_model.h5'
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    return model, history

def load_and_preprocess_data(file_path):
    """
    Load and preprocess stock data from CSV file
    
    Parameters:
    file_path (str): Path to the CSV file containing stock data
    
    Returns:
    pandas.DataFrame: Preprocessed stock data
    """
    # Import required libraries
    import pandas as pd
    import numpy as np
    
    # Load the CSV file
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
    # Fill numeric columns with forward fill, then backward fill
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_columns] = df[numeric_columns].fillna(method='ffill').fillna(method='bfill')
    
    # If any missing values remain, fill with median
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
    
    # Apply advanced feature engineering (using the function you already defined)
    df = advanced_feature_engineering(df)
    
    # Optional: Log transformation for volume to handle skewness
    df['volume'] = np.log1p(df['volume'])
    
    # Remove rows with infinite values
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    print(f"Data loaded and preprocessed. Shape: {df.shape}")
    print("Columns:", list(df.columns))
    
    return df

def main():
    """
    Main execution function to orchestrate the entire process
    """
    # File path for your stock data
    file_path = r"E:\CodeFusion\combined_stock_data.csv"
    
    # Load and preprocess data
    processed_data = load_and_preprocess_data(file_path)
    
    # Prepare dataset for LSTM
    X_train, X_val, X_test, y_train, y_val, y_test = define_dataset(processed_data)
    
    # Train the model
    model, training_history = train_lstm_model(
        X_train, X_val, X_test, 
        y_train, y_val, y_test
    )

if __name__ == "__main__":
    main()