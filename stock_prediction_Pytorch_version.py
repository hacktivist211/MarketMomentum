import os
import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models.temporal_fusion_transformer import TemporalFusionTransformer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import datetime
import time

# ============================
# Utility Functions
# ============================

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset.
    """
    print("Loading data...")
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}. Please check the path.")
        exit()
    
    print("Preprocessing data...")
    data['volume_change'] = data['volume'].pct_change()  # Percentage change in volume
    data = data.fillna(method='ffill').fillna(method='bfill')  # Fill missing values
    data['company'] = data['company'].astype('category').cat.codes  # Convert company to categorical
    data['time'] = pd.to_datetime(data['time'])  # Ensure 'time' is in datetime format
    print("Data loaded and preprocessed.")
    return data


def define_dataset(data, max_encoder_length=60, max_prediction_length=10):
    """
    Define the TimeSeriesDataSet for the Temporal Fusion Transformer.
    """
    print("Defining dataset...")
    try:
        dataset = TimeSeriesDataSet(
            data,
            time_idx="time",  # Time index column
            target="close",  # Target column to predict
            group_ids=["company"],  # Group by company
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            static_categoricals=["company"],  # Static feature: company
            time_varying_known_reals=["open", "high", "low", "volume"],  # Known at prediction time
            time_varying_unknown_reals=["close"],  # Target to predict
        )
    except Exception as e:
        print(f"Error while defining dataset: {e}")
        exit()
    print("Dataset defined.")
    return dataset


def setup_callbacks():
    """
    Set up callbacks for early stopping and model checkpointing.
    """
    print("Setting up callbacks...")
    # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=5,
        verbose=True,
        mode="min"
    )
    
    # Model checkpoint callback
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models",
        filename=f"tft_model_{timestamp}_{{epoch:02d}}_{{val_loss:.2f}}",
        save_top_k=1,
        mode="min",
    )
    
    os.makedirs("models", exist_ok=True)  # Ensure models directory exists
    return [early_stop_callback, checkpoint_callback]


def train_model(dataset, batch_size=64, max_epochs=50):
    """
    Train the Temporal Fusion Transformer model.
    Includes live updates for epoch progress and total time tracking.
    """
    print("Splitting dataset...")
    train_data, val_data = dataset.split_before(0.8)  # 80% training, 20% validation/test
    val_data, test_data = val_data.split_before(0.5)  # Split remaining 20% into validation and test
    
    # Create DataLoader objects
    train_dataloader = train_data.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_dataloader = val_data.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    test_dataloader = test_data.to_dataloader(train=False, batch_size=batch_size, num_workers=4)
    
    print("Defining model...")
    tft = TemporalFusionTransformer.from_dataset(
        dataset,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
    )
    
    # Set up logger and callbacks
    logger = TensorBoardLogger("logs", name="tft_model")  # Set up TensorBoard
    callbacks = setup_callbacks()
    
    # Set up the trainer
    trainer = Trainer(
        max_epochs=max_epochs,
        gpus=1,
        logger=logger,
        callbacks=callbacks
    )
    
    print("Training model...")
    start_time = time.time()  # Start timer
    
    # Wrap the training process with tqdm for live updates
    for epoch in tqdm(range(1, max_epochs + 1), desc="Training Progress"):
        epoch_start = time.time()
        
        trainer.fit(tft, train_dataloader, val_dataloader)
        
        # Calculate epoch time and estimated time remaining
        epoch_duration = time.time() - epoch_start
        remaining_epochs = max_epochs - epoch
        estimated_time_left = epoch_duration * remaining_epochs
        
        print(f"\nEpoch {epoch}/{max_epochs} completed in {epoch_duration:.2f} seconds.")
        print(f"Estimated time left: {estimated_time_left / 60:.2f} minutes.")
    
    # Total training time
    total_training_time = time.time() - start_time
    print(f"\nTotal training time: {total_training_time / 60:.2f} minutes.")
    
    print("Testing model...")
    trainer.test(tft, test_dataloader)
    
    print("Saving trained model...")
    model_path = r"E:\CodeFusion\Projects\StockPrediction\tft_model.pth"
    torch.save(tft.state_dict(), model_path)
    print(f"Model saved to {model_path}")


# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    # File path for your combined CSV file
    file_path = r"E:\CodeFusion\combined_stock_data.csv"
    
    # Step 1: Load and preprocess the data
    combined_data = load_and_preprocess_data(file_path)
    
    # Step 2: Define the dataset
    dataset = define_dataset(combined_data)
    
    # Step 3: Train the model
    train_model(dataset)
    
    print("Process complete.")