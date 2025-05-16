#!/usr/bin/env python3
"""
Stock Price Prediction Model Training Script

This script trains an LSTM model to predict stock prices based on historical data.
It loads data from the raw data directory, creates features, scales them,
trains the model, and saves both the model and scaler for later use.

Usage:
    python train_model.py

The trained model will be saved to the models directory.
"""

import os
import logging
import argparse
import pandas as pd
import numpy as np
import joblib
import time
import glob
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import configurations
from app.config import (
    STOCK_SYMBOL,
    MODEL_PATH,
    STOCK_DATA_FILE,
    MODEL_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    LOG_LEVEL,
    TEST_SIZE,
    MIN_TRAINING_SAMPLES
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(DATA_DIR, 'logs', 'train_model.log'))
    ]
)
logger = logging.getLogger(__name__)

# LSTM model parameters (can be moved to config.py for better configurability)
SEQUENCE_LENGTH = 5
EPOCHS = 100
BATCH_SIZE = 16
LSTM_UNITS = 50
DROPOUT_RATE = 0.2
PATIENCE = 15

# Real-time training parameters
WATCH_INTERVAL = 60  # seconds between checking for new data
MIN_NEW_SAMPLES = 5  # minimum number of new samples to trigger retraining


def create_sequences(X, y, seq_length):
    """
    Create sequences for LSTM model from the original data
    
    Args:
        X (numpy.ndarray): Features array
        y (numpy.ndarray): Target array
        seq_length (int): Sequence length for LSTM
        
    Returns:
        tuple: (X_seq, y_seq) with sequence data
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
        
    return np.array(X_seq), np.array(y_seq)


def create_features(df):
    """
    Create additional features from the original price data
    
    Args:
        df (pandas.DataFrame): Original dataframe with price data
        
    Returns:
        pandas.DataFrame: Dataframe with additional features
    """
    # Copy dataframe to avoid modifying the original
    df_feat = df.copy()
    
    # Ensure dataframe is sorted by timestamp
    df_feat['timestamp'] = pd.to_datetime(df_feat['timestamp'])
    df_feat = df_feat.sort_values('timestamp')
    
    # Add price change features
    df_feat['price_change'] = df_feat['current_price'].diff()
    df_feat['price_change_pct'] = df_feat['current_price'].pct_change() * 100
    
    # Add price difference features
    df_feat['high_low_diff'] = df_feat['high_price'] - df_feat['low_price']
    df_feat['close_open_diff'] = df_feat['current_price'] - df_feat['open_price']
    
    # Add moving averages
    df_feat['ma5'] = df_feat['current_price'].rolling(window=5).mean()
    df_feat['ma10'] = df_feat['current_price'].rolling(window=10).mean()
    
    # Add volatility (standard deviation over a window)
    df_feat['volatility5'] = df_feat['current_price'].rolling(window=5).std()
    
    # Add momentum indicators
    df_feat['momentum'] = df_feat['current_price'] - df_feat['current_price'].shift(5)
    
    # Add day of week (0-4 for weekdays, assuming we only have weekday data)
    df_feat['day_of_week'] = df_feat['timestamp'].dt.dayofweek
    
    # Remove rows with NaN values (caused by diff and rolling operations)
    df_feat = df_feat.dropna()
    
    logger.info(f"Created features dataframe with {len(df_feat)} rows and {len(df_feat.columns)} columns")
    
    return df_feat


def load_data():
    """
    Load stock data from CSV file
    
    Returns:
        pandas.DataFrame or None: Loaded dataframe or None if loading failed
    """
    try:
        df = pd.read_csv(STOCK_DATA_FILE)
        logger.info(f"Loaded {len(df)} data points from {STOCK_DATA_FILE}")
        
        if len(df) < MIN_TRAINING_SAMPLES:
            logger.warning(f"Not enough data for training, need at least {MIN_TRAINING_SAMPLES} samples")
            return None
            
        return df
    except (FileNotFoundError, pd.errors.EmptyDataError) as e:
        logger.error(f"Failed to load data: {e}")
        return None


def build_lstm_model(input_shape):
    """
    Build an LSTM model for stock price prediction
    
    Args:
        input_shape (tuple): Shape of input data (sequence_length, n_features)
        
    Returns:
        tensorflow.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(LSTM_UNITS, return_sequences=True, input_shape=input_shape),
        Dropout(DROPOUT_RATE),
        LSTM(LSTM_UNITS),
        Dropout(DROPOUT_RATE),
        Dense(20, activation='relu'),
        Dense(1)
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )
    
    logger.info(f"Built LSTM model with input shape {input_shape}")
    return model


def save_model_and_scaler(model, scaler):
    """
    Save the trained Keras model and scaler
    
    Args:
        model (tensorflow.keras.Model): Trained LSTM model
        scaler (sklearn.preprocessing.StandardScaler): Fitted scaler
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        # Create model directory if it doesn't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Define file paths
        model_keras_path = os.path.join(MODEL_DIR, f"{STOCK_SYMBOL.lower()}_lstm_model.h5")
        model_scaler_path = os.path.join(MODEL_DIR, f"{STOCK_SYMBOL.lower()}_scaler.joblib")
        
        # Save Keras model
        model.save(model_keras_path)
        logger.info(f"Saved Keras model to {model_keras_path}")
        
        # Save scaler
        joblib.dump(scaler, model_scaler_path)
        logger.info(f"Saved scaler to {model_scaler_path}")
        
        # Save the combined model data for compatibility with existing code
        model_data = {
            'model_type': 'lstm',
            'model_path': model_keras_path,
            'scaler': scaler
        }
        joblib.dump(model_data, MODEL_PATH)
        logger.info(f"Saved combined model data to {MODEL_PATH}")
        
        return True
    except Exception as e:
        logger.error(f"Failed to save model: {e}")
        return False


def train(force=False, epochs=None, batch_size=None, sequence_length=None):
    """
    Main training function to load data, create features, train and save the model
    
    Args:
        force (bool): Force training even if model exists
        epochs (int): Number of epochs for training, uses default if None
        batch_size (int): Batch size for training, uses default if None
        sequence_length (int): Sequence length for LSTM, uses default if None
        
    Returns:
        dict: Training metrics and status
    """
    # Use provided parameters or defaults
    actual_epochs = epochs if epochs is not None else EPOCHS
    actual_batch_size = batch_size if batch_size is not None else BATCH_SIZE
    actual_sequence_length = sequence_length if sequence_length is not None else SEQUENCE_LENGTH
    
    logger.info(f"Starting model training for {STOCK_SYMBOL}")
    
    # Check if model exists and we're not forcing a retrain
    if os.path.exists(MODEL_PATH) and not force:
        logger.info(f"Model already exists at {MODEL_PATH}. Use --force to retrain.")
        return {"status": "skipped", "message": "Model already exists. Use --force to retrain."}
    
    # Load data
    df = load_data()
    if df is None or len(df) < MIN_TRAINING_SAMPLES:
        return {"status": "insufficient_data", "message": f"Need at least {MIN_TRAINING_SAMPLES} data points"}
    
    try:
        # Create features
        logger.info("Creating features from price data")
        df_features = create_features(df)
        
        # Select features for prediction
        # Base features from existing data
        feature_columns = ['prev_close_price', 'open_price', 'high_price', 'low_price']
        # Add derived features
        feature_columns.extend(['price_change', 'price_change_pct', 'high_low_diff', 
                              'close_open_diff', 'ma5', 'ma10', 'volatility5', 'momentum'])
        
        # Target is the current price
        target_column = 'current_price'
        
        # Prepare data for model
        X = df_features[feature_columns].values
        y = df_features[target_column].values
        
        # Scale features
        logger.info("Scaling features")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data into training and testing sets
        logger.info(f"Splitting data with test size {TEST_SIZE}")
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=TEST_SIZE, shuffle=False
        )
        
        # Create sequences for LSTM
        logger.info(f"Creating sequences with length {actual_sequence_length}")
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, actual_sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, actual_sequence_length)
        
        # Build and train LSTM model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_lstm_model(input_shape)
        
        # Define callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
            ModelCheckpoint(
                filepath=os.path.join(MODEL_DIR, 'checkpoints', 'lstm_checkpoint.h5'),
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Create checkpoint directory
        os.makedirs(os.path.join(MODEL_DIR, 'checkpoints'), exist_ok=True)
        
        logger.info(f"Training LSTM model with {len(X_train_seq)} sequences, {actual_epochs} epochs, {actual_batch_size} batch size")
        history = model.fit(
            X_train_seq, y_train_seq,
            epochs=actual_epochs,
            batch_size=actual_batch_size,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        logger.info("Evaluating model")
        train_predictions = model.predict(X_train_seq)
        test_predictions = model.predict(X_test_seq)
        
        train_mse = mean_squared_error(y_train_seq, train_predictions)
        test_mse = mean_squared_error(y_test_seq, test_predictions)
        
        train_r2 = r2_score(y_train_seq, train_predictions)
        test_r2 = r2_score(y_test_seq, test_predictions)
        
        # Calculate direction accuracy
        train_direction = np.sign(np.diff(y_train_seq.reshape(-1)))
        train_pred_direction = np.sign(np.diff(train_predictions.reshape(-1)))
        train_direction_accuracy = np.mean(train_direction == train_pred_direction)
        
        test_direction = np.sign(np.diff(y_test_seq.reshape(-1)))
        test_pred_direction = np.sign(np.diff(test_predictions.reshape(-1)))
        test_direction_accuracy = np.mean(test_direction == test_pred_direction)
        
        # Log metrics
        logger.info(f"Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}, Direction Accuracy: {train_direction_accuracy:.4f}")
        logger.info(f"Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}, Direction Accuracy: {test_direction_accuracy:.4f}")
        
        # Save model and scaler
        save_model_and_scaler(model, scaler)
        
        # Save metrics to file
        metrics_file = os.path.join(DATA_DIR, 'processed', 'model_metrics.csv')
        metrics_df = pd.DataFrame({
            'timestamp': [datetime.now().isoformat()],
            'model_type': ['lstm'],
            'train_mse': [train_mse],
            'test_mse': [test_mse],
            'train_r2': [train_r2],
            'test_r2': [test_r2],
            'train_direction_accuracy': [train_direction_accuracy],
            'test_direction_accuracy': [test_direction_accuracy],
            'samples': [len(df_features)],
            'features': [','.join(feature_columns)],
            'sequence_length': [actual_sequence_length],
            'epochs': [len(history.history['loss'])]
        })
        
        # Append to existing metrics if file exists
        if os.path.exists(metrics_file):
            existing_metrics = pd.read_csv(metrics_file)
            metrics_df = pd.concat([existing_metrics, metrics_df], ignore_index=True)
        
        # Save metrics
        metrics_df.to_csv(metrics_file, index=False)
        logger.info(f"Saved metrics to {metrics_file}")
        
        return {
            "status": "success",
            "samples": len(df_features),
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "train_direction_accuracy": train_direction_accuracy,
            "test_direction_accuracy": test_direction_accuracy,
            "epochs": len(history.history['loss'])
        }
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Training failed: {str(e)}"}


def load_or_create_model(force=False, sequence_length=None):
    """
    Load an existing model or create a new one if it doesn't exist
    
    Args:
        force (bool): Force creating a new model even if one exists
        sequence_length (int): Sequence length for LSTM model
        
    Returns:
        tuple: (model, scaler, is_new) where is_new is True if a new model was created
    """
    seq_len = sequence_length if sequence_length is not None else SEQUENCE_LENGTH
    
    # Define file paths
    model_keras_path = os.path.join(MODEL_DIR, f"{STOCK_SYMBOL.lower()}_lstm_model.h5")
    model_scaler_path = os.path.join(MODEL_DIR, f"{STOCK_SYMBOL.lower()}_scaler.joblib")
    
    # Check if model exists
    if os.path.exists(model_keras_path) and os.path.exists(model_scaler_path) and not force:
        try:
            # Load existing model and scaler
            model = load_model(model_keras_path)
            scaler = joblib.load(model_scaler_path)
            logger.info(f"Loaded existing model from {model_keras_path}")
            return model, scaler, False
        except Exception as e:
            logger.error(f"Failed to load existing model: {e}. Will create a new one.")
    
    # Create a new model
    # Note: We need some dummy data to initialize the model shape
    # In a real implementation, we'd determine the number of features from the data
    # Base features plus derived features
    n_features = 12  # 4 base features + 8 derived features
    input_shape = (seq_len, n_features)
    model = build_lstm_model(input_shape)
    scaler = StandardScaler()
    logger.info("Created new model")
    
    return model, scaler, True


def update_model_with_new_data(model, scaler, min_samples=MIN_NEW_SAMPLES, epochs=EPOCHS, 
                              batch_size=BATCH_SIZE, sequence_length=SEQUENCE_LENGTH):
    """
    Update an existing model with new data
    
    Args:
        model: Existing Keras model
        scaler: Existing scaler
        min_samples (int): Minimum number of samples needed to update
        epochs (int): Number of epochs for training
        batch_size (int): Batch size for training
        sequence_length (int): Sequence length for LSTM
        
    Returns:
        tuple: (updated_model, updated_scaler, metrics) or (model, scaler, None) if update wasn't performed
    """
    # Get newly available data
    new_data_files = glob.glob(os.path.join(RAW_DATA_DIR, f"{STOCK_SYMBOL.lower()}_*.csv"))
    if not new_data_files:
        logger.info("No new data files found")
        return model, scaler, None
    
    # Load and combine new data
    dfs = []
    for file in new_data_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
            logger.info(f"Loaded {len(df)} rows from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
    
    if not dfs:
        logger.warning("No valid data files found")
        return model, scaler, None
        
    # Combine all data
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Check if we have enough new data
    if len(combined_df) < min_samples:
        logger.info(f"Not enough new data: {len(combined_df)} samples, need at least {min_samples}")
        return model, scaler, None
    
    # Create features
    df_features = create_features(combined_df)
    
    # Select features for prediction
    feature_columns = ['prev_close_price', 'open_price', 'high_price', 'low_price']
    feature_columns.extend(['price_change', 'price_change_pct', 'high_low_diff', 
                          'close_open_diff', 'ma5', 'ma10', 'volatility5', 'momentum'])
    
    # Target is the current price
    target_column = 'current_price'
    
    # Prepare data for model
    X = df_features[feature_columns].values
    y = df_features[target_column].values
    
    # Scale features using the existing scaler
    X_scaled = scaler.transform(X)
    
    # Create sequences
    X_seq, y_seq = create_sequences(X_scaled, y, sequence_length)
    
    # If we have too few sequences after creation, don't proceed
    if len(X_seq) < min_samples:
        logger.info(f"Not enough sequences after processing: {len(X_seq)}, need {min_samples}")
        return model, scaler, None
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=TEST_SIZE, shuffle=False
    )
    
    # Define callbacks for training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    ]
    
    # Update the model with the new data
    logger.info(f"Updating model with {len(X_train)} sequences")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate updated model
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    train_mse = mean_squared_error(y_train, train_predictions)
    test_mse = mean_squared_error(y_test, test_predictions)
    
    train_r2 = r2_score(y_train, train_predictions)
    test_r2 = r2_score(y_test, test_predictions)
    
    # Calculate direction accuracy
    train_direction = np.sign(np.diff(y_train.reshape(-1)))
    train_pred_direction = np.sign(np.diff(train_predictions.reshape(-1)))
    train_direction_accuracy = np.mean(train_direction == train_pred_direction)
    
    test_direction = np.sign(np.diff(y_test.reshape(-1)))
    test_pred_direction = np.sign(np.diff(test_predictions.reshape(-1)))
    test_direction_accuracy = np.mean(test_direction == test_pred_direction)
    
    # Log metrics
    logger.info(f"Updated model - Training MSE: {train_mse:.4f}, R²: {train_r2:.4f}, Direction Accuracy: {train_direction_accuracy:.4f}")
    logger.info(f"Updated model - Testing MSE: {test_mse:.4f}, R²: {test_r2:.4f}, Direction Accuracy: {test_direction_accuracy:.4f}")
    
    # Save updated model and metrics
    save_model_and_scaler(model, scaler)
    
    # Return metrics
    metrics = {
        "status": "success",
        "samples": len(df_features),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_direction_accuracy": train_direction_accuracy,
        "test_direction_accuracy": test_direction_accuracy,
        "epochs": len(history.history['loss'])
    }
    
    return model, scaler, metrics


def get_new_data_files():
    """
    Get a list of new data files that haven't been processed yet
    
    Returns:
        list: List of new data file paths
    """
    # Get all CSV files in the raw data directory for the current stock symbol
    data_files = glob.glob(os.path.join(RAW_DATA_DIR, f"{STOCK_SYMBOL.lower()}_*.csv"))
    
    # Get the processed files marker path
    processed_marker = os.path.join(DATA_DIR, 'processed', 'processed_files.txt')
    
    # Get list of already processed files
    processed_files = set()
    if os.path.exists(processed_marker):
        with open(processed_marker, 'r') as f:
            processed_files = set(f.read().splitlines())
    
    # Filter out already processed files
    new_files = [f for f in data_files if f not in processed_files]
    
    return new_files


def mark_files_as_processed(file_paths):
    """
    Mark a list of files as processed
    
    Args:
        file_paths (list): List of file paths to mark as processed
    """
    # Ensure the processed directory exists
    os.makedirs(os.path.join(DATA_DIR, 'processed'), exist_ok=True)
    
    # Get the processed files marker path
    processed_marker = os.path.join(DATA_DIR, 'processed', 'processed_files.txt')
    
    # Append new processed files
    with open(processed_marker, 'a') as f:
        for file_path in file_paths:
            f.write(f"{file_path}\n")
    
    logger.info(f"Marked {len(file_paths)} files as processed")


class StockDataEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler for monitoring new stock data files
    """
    def __init__(self, callback):
        """
        Initialize the event handler
        
        Args:
            callback (callable): Callback function to call when new files are detected
        """
        self.callback = callback
        self.last_processed_time = time.time()
        # Set to keep track of files we've already seen
        self.seen_files = set()
    
    def on_created(self, event):
        """
        Handler for file creation events
        
        Args:
            event: Watchdog event object
        """
        if event.is_directory:
            return
            
        # Check if the file is a CSV file for our stock symbol
        if event.src_path.endswith('.csv') and STOCK_SYMBOL.lower() in os.path.basename(event.src_path):
            # Avoid processing the same file multiple times
            if event.src_path in self.seen_files:
                return
                
            self.seen_files.add(event.src_path)
            
            # Don't process too frequently
            current_time = time.time()
            if current_time - self.last_processed_time >= WATCH_INTERVAL:
                logger.info(f"New file detected: {event.src_path}")
                self.last_processed_time = current_time
                
                # Call the callback function
                self.callback()


def continuous_training_loop():
    """
    Main loop for continuous training mode
    """
    logger.info("Starting continuous training mode")
    
    # Load or create the initial model
    model, scaler, is_new = load_or_create_model()
    
    # If it's a new model, train it with available data
    if is_new:
        logger.info("Training initial model")
        result = train(force=True)
        if result["status"] != "success":
            logger.warning(f"Initial training failed: {result.get('message', 'Unknown error')}")
    
    # Define callback for when new data is detected
    def on_new_data():
        nonlocal model, scaler
        logger.info("New data detected, updating model")
        new_files = get_new_data_files()
        
        if new_files:
            logger.info(f"Found {len(new_files)} new data files")
            model, scaler, metrics = update_model_with_new_data(model, scaler)
            
            if metrics is not None:
                logger.info(f"Model updated successfully with metrics: {metrics}")
                # Mark files as processed
                mark_files_as_processed(new_files)
            else:
                logger.info("Model update skipped (insufficient data or other reason)")
    
    # Set up watchdog observer
    event_handler = StockDataEventHandler(on_new_data)
    observer = Observer()
    observer.schedule(event_handler, RAW_DATA_DIR, recursive=False)
    observer.start()
    
    try:
        logger.info(f"Monitoring {RAW_DATA_DIR} for new data files...")
        
        # Initial check for new files
        on_new_data()
        
        # Main loop
        while True:
            time.sleep(WATCH_INTERVAL)
            # Check for new files periodically in addition to the file watcher
            on_new_data()
            
    except KeyboardInterrupt:
        logger.info("Stopping continuous training loop")
        observer.stop()
    finally:
        observer.join()


def batch_training_mode(force=False, epochs=None, batch_size=None, sequence_length=None):
    """
    Run a single batch training session
    """
    logger.info("Running batch training mode")
    result = train(
        force=force,
        epochs=epochs,
        batch_size=batch_size,
        sequence_length=sequence_length
    )
    
    logger.info(f"Batch training completed with status: {result['status']}")
    return result


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stock Price Prediction Model Training')
    
    # Mode selection
    parser.add_argument('--mode', choices=['batch', 'continuous'], default='batch',
                       help='Training mode: batch (default) or continuous')
    
    # Force retraining even if model exists
    parser.add_argument('--force', action='store_true',
                       help='Force retraining even if model exists')
    
    # Model hyperparameters
    parser.add_argument('--epochs', type=int, default=None,
                       help=f'Number of training epochs (default: {EPOCHS})')
    parser.add_argument('--batch-size', type=int, default=None,
                       help=f'Batch size for training (default: {BATCH_SIZE})')
    parser.add_argument('--sequence-length', type=int, default=None,
                       help=f'Sequence length for LSTM (default: {SEQUENCE_LENGTH})')
    
    # Continuous mode specific parameters
    parser.add_argument('--watch-interval', type=int, default=None,
                       help=f'Interval (seconds) between checking for new data (default: {WATCH_INTERVAL})')
    parser.add_argument('--min-new-samples', type=int, default=None,
                       help=f'Minimum new samples required for retraining (default: {MIN_NEW_SAMPLES})')
    
    args = parser.parse_args()
    
    # Update globals if overridden by command line
    if args.watch_interval is not None:
        WATCH_INTERVAL = args.watch_interval
    if args.min_new_samples is not None:
        MIN_NEW_SAMPLES = args.min_new_samples
    
    # Run in specified mode
    if args.mode == 'continuous':
        continuous_training_loop()
    else:
        batch_training_mode(
            force=args.force, 
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )

