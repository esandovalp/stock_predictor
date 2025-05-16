#!/usr/bin/env python3
"""
Test Data Generator for Stock Price Prediction System

This script generates synthetic stock data with realistic patterns for testing
the stock prediction model. It supports two modes:

1. Bulk Generation: Creates a large dataset for initial model training
2. Continuous Updates: Simulates real-time data generation with configurable frequency

Usage:
    # Generate bulk data (default):
    python test_data_generator.py --mode bulk --samples 200

    # Generate continuous updates:
    python test_data_generator.py --mode continuous --interval 60 --samples-per-update 5

The data will be saved to data/raw/ directory with appropriate timestamps.
"""

import pandas as pd
import numpy as np
import os
import argparse
import time
import logging
import random
from datetime import datetime, timedelta
import signal
import sys

# Import configurations from our config file
from app.config import (
    STOCK_SYMBOL as CONFIG_STOCK_SYMBOL,
    DATA_DIR,
    RAW_DATA_DIR,
    MODEL_FEATURES,
    MODEL_TARGET,
    LOG_DIR
)

# Global variables
global STOCK_SYMBOL
STOCK_SYMBOL = CONFIG_STOCK_SYMBOL

# Configure logging
log_file = os.path.join(LOG_DIR, 'test_data_generator.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file)
    ]
)
logger = logging.getLogger(__name__)

# Global variables to track the current price state
current_price = None
last_timestamp = None
price_trend = 0  # -1 = downward, 0 = sideways, 1 = upward
trend_days_left = 0
volatility_factor = 0.015


def initialize_price_state(start_price=None):
    """
    Initialize or reset the global price state variables
    
    Args:
        start_price (float): Optional starting price, random if None
    """
    global current_price, last_timestamp, price_trend, trend_days_left, volatility_factor
    
    # Set starting price
    if start_price is None:
        # Random starting price between 150 and 200
        current_price = 150 + random.random() * 50
    else:
        current_price = start_price
    
    # Set last timestamp to now (will be overridden if reading existing data)
    last_timestamp = datetime.now()
    
    # Randomly select trend and volatility
    price_trend = random.choice([-1, 0, 1])
    trend_days_left = random.randint(5, 20)  # Trend lasts for 5-20 days
    volatility_factor = 0.01 + random.random() * 0.02  # 1-3% volatility
    
    logger.info(f"Initialized price state: price=${current_price:.2f}, "
                f"trend={price_trend}, volatility={volatility_factor:.4f}")


def load_latest_state():
    """
    Load the latest price state from existing data files
    
    Returns:
        bool: True if state was loaded successfully, False otherwise
    """
    global current_price, last_timestamp, price_trend, volatility_factor
    
    try:
        # Find all existing data files for this stock
        files = [f for f in os.listdir(RAW_DATA_DIR) 
                if f.startswith(f"{STOCK_SYMBOL.lower()}_") and f.endswith(".csv")]
        
        if not files:
            logger.info("No existing data files found, using default state")
            return False
        
        # Find the most recent file (sorted by name which includes timestamp)
        latest_file = sorted(files)[-1]
        file_path = os.path.join(RAW_DATA_DIR, latest_file)
        
        # Load the latest file
        df = pd.read_csv(file_path)
        if len(df) == 0:
            logger.warning(f"Latest file {latest_file} is empty, using default state")
            return False
        
        # Sort by timestamp and get the last row
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        last_row = df.iloc[-1]
        
        # Update the global state
        current_price = last_row['current_price']
        last_timestamp = df['timestamp'].iloc[-1].to_pydatetime()
        
        # Calculate a new volatility based on recent data
        if len(df) >= 10:
            recent_data = df.tail(10)
            volatility_factor = np.std(recent_data['current_price'].pct_change().dropna()) * 0.8
            # Clamp volatility to reasonable range
            volatility_factor = max(0.005, min(0.03, volatility_factor))
        
        # Set a random new trend
        price_trend = random.choice([-1, 0, 1])
        trend_days_left = random.randint(5, 20)
        
        logger.info(f"Loaded state from {latest_file}: price=${current_price:.2f}, "
                   f"last_timestamp={last_timestamp}, volatility={volatility_factor:.4f}")
        return True
    
    except Exception as e:
        logger.error(f"Error loading latest state: {e}")
        logger.info("Using default state instead")
        return False


def update_trend():
    """
    Update the price trend periodically to simulate market cycles
    """
    global price_trend, trend_days_left, volatility_factor
    
    # Decrease days left in the current trend
    trend_days_left -= 1
    
    # Change trend if the current one has ended
    if trend_days_left <= 0:
        # Weighted random choice for next trend
        if price_trend == 1:  # If it was going up
            weights = [0.4, 0.4, 0.2]  # Higher chance of reversal or sideways
        elif price_trend == -1:  # If it was going down
            weights = [0.2, 0.4, 0.4]  # Higher chance of reversal or sideways
        else:  # If it was sideways
            weights = [0.3, 0.4, 0.3]  # Equal chance of any direction
        
        # Choose new trend: -1 (down), 0 (sideways), 1 (up)
        price_trend = random.choices([-1, 0, 1], weights=weights)[0]
        
        # Set new trend duration (5-20 days)
        trend_days_left = random.randint(5, 20)
        
        # Occasionally change volatility
        if random.random() < 0.3:
            # Either increase or decrease volatility
            if random.random() < 0.5:
                volatility_factor *= 1.2  # Increase
            else:
                volatility_factor *= 0.8  # Decrease
            
            # Clamp to reasonable range (0.5% to 3%)
            volatility_factor = max(0.005, min(0.03, volatility_factor))
        
        logger.info(f"Trend changed: direction={price_trend}, days_left={trend_days_left}, "
                    f"volatility={volatility_factor:.4f}")


def generate_stock_data(num_samples=50, continuation=False, start_date=None):
    """
    Generate synthetic stock price data with realistic patterns.
    
    Args:
        num_samples (int): Number of data points to generate
        continuation (bool): Whether this is continuing from previous data
        start_date (datetime): Starting date for data generation
        
    Returns:
        pd.DataFrame: DataFrame with generated stock data
    """
    global current_price, last_timestamp, price_trend, trend_days_left
    
    # Initialize the price state if this is not a continuation
    if not continuation or current_price is None:
        initialize_price_state()
    
    # Determine the start date for data generation
    if start_date is None:
        if continuation and last_timestamp is not None:
            # If continuing, start from the day after last timestamp
            start_date = last_timestamp + timedelta(days=1)
        else:
            # Otherwise start a certain number of days back from today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=num_samples)
    
    # Initialize empty lists for data
    timestamps = []
    prev_close_prices = []
    open_prices = []
    high_prices = []
    low_prices = []
    close_prices = []
    volumes = []
    
    # Keep track of the last generated date
    current_date = start_date
    
    # Generate data with realistic patterns
    for i in range(num_samples):
        # Skip weekends (0 = Monday, 6 = Sunday)
        while current_date.weekday() >= 5:  # Saturday or Sunday
            current_date += timedelta(days=1)
        
        # Format timestamp
        timestamp = current_date.replace(hour=16, minute=0, second=0)  # End of trading day
        
        # Update trend if this is a new day
        update_trend()
        
        # Determine daily price movement simulation
        # Base volatility plus trend influence
        trend_influence = price_trend * 0.003  # Trend adds/subtracts up to 0.3% per day
        daily_movement = np.random.normal(trend_influence, volatility_factor)
        
        # Add some mean-reversion tendency if price gets too extreme
        mean_price = 180.0
        if current_price > mean_price * 1.5:  # 50% above mean
            # More likely to go down
            daily_movement -= volatility_factor * 0.5
        elif current_price < mean_price * 0.7:  # 30% below mean
            # More likely to go up
            daily_movement += volatility_factor * 0.5
        
        # Previous close
        if i == 0 and not continuation:
            # First data point without continuation - set previous close to similar to current price
            prev_close = current_price * (1 + (random.random() - 0.5) * volatility_factor * 0.5)
        else:
            # Use current price as previous close
            prev_close = current_price
        
        # Open price has overnight gap from previous close
        overnight_gap = np.random.normal(0, volatility_factor * 0.7)  # Overnight gaps can be larger
        open_price = prev_close * (1 + overnight_gap)
        
        # Calculate current day's close price
        close_price = open_price * (1 + daily_movement)
        
        # High and low prices
        price_range = abs(close_price - open_price) + (close_price * volatility_factor)
        high_price = max(close_price, open_price) + (random.random() * price_range * 0.5)
        low_price = min(close_price, open_price) - (random.random() * price_range * 0.5)
        
        # Ensure low is actually lower than high
        if low_price >= high_price:
            low_price = high_price * 0.99
        
        # Generate volume (higher on volatile days)
        base_volume = 1_000_000  # Base volume for the stock
        volume_volatility = abs(close_price - open_price) / open_price  # More volume on volatile days
        daily_volume = int(base_volume * (1 + volume_volatility * 5) * (0.7 + 0.6 * random.random()))
        
        # Add some randomness to volume
        # More volume on trend change days and high volatility days
        if trend_days_left > 18 or volume_volatility > 0.02:
            daily_volume *= 1.5
        
        # Update the current price for next iteration
        current_price = close_price
        
        # Update the last timestamp
        last_timestamp = timestamp
        
        # Append to lists
        timestamps.append(timestamp)
        prev_close_prices.append(round(prev_close, 2))
        open_prices.append(round(open_price, 2))
        high_prices.append(round(high_price, 2))
        low_prices.append(round(low_price, 2))
        close_prices.append(round(close_price, 2))
        volumes.append(daily_volume)
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prev_close_price': prev_close_prices,
        'open_price': open_prices,
        'high_price': high_prices,
        'low_price': low_prices,
        'current_price': close_prices,
        'volume': volumes
    })
    
    return df


def save_data(df, mode='bulk', update_id=None):
    """
    Save generated data to CSV file(s)
    
    Args:
        df (pd.DataFrame): DataFrame with generated data
        mode (str): 'bulk' or 'continuous' - affects file naming
        update_id (int): Identifier for the update in continuous mode
        
    Returns:
        str: Path to the saved file
    """
    # Ensure the data directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Format timestamp for filename
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create filename based on mode
    if mode == 'bulk':
        filename = f"{STOCK_SYMBOL.lower()}_stock_data.csv"
    else:
        # For continuous mode, include update ID and timestamp
        filename = f"{STOCK_SYMBOL.lower()}_update_{update_id}_{timestamp_str}.csv"
    
    # Full file path
    file_path = os.path.join(RAW_DATA_DIR, filename)
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    
    # Log statistics
    logger.info(f"Saved {len(df)} rows to {file_path}")
    logger.info(f"Price range: ${df['current_price'].min():.2f} - ${df['current_price'].max():.2f}")
    logger.info(f"Average volume: {df['volume'].mean():.0f}")
    
    # Verify the file exists
    if os.path.exists(file_path):
        logger.info(f"File successfully created at {file_path}")
        return file_path
    else:
        logger.error(f"Failed to create file at {file_path}")
        return None


def bulk_generation(samples, start_date=None):
    """
    Generate a bulk dataset for initial model training
    
    Args:
        samples (int): Number of samples to generate
        start_date (datetime): Optional start date
        
    Returns:
        bool: Success status
    """
    logger.info(f"Starting bulk generation of {samples} samples")
    
    # Initialize fresh state
    initialize_price_state()
    
    # Generate the data
    df = generate_stock_data(
        num_samples=samples,
        continuation=False,
        start_date=start_date
    )
    
    # Save the data to CSV
    result = save_data(df, mode='bulk')
    
    if result:
        logger.info(f"Bulk generation completed successfully with {samples} samples")
        return True
    else:
        logger.error("Bulk generation failed")
        return False


def continuous_generation(interval, samples_per_update, max_updates=None):
    """
    Generate data continuously to simulate real-time updates
    
    Args:
        interval (int): Time between updates in seconds
        samples_per_update (int): Number of samples to generate per update
        max_updates (int): Maximum number of updates to generate, None for unlimited
        
    Returns:
        bool: Success status
    """
    logger.info(f"Starting continuous generation mode: interval={interval}s, "
               f"samples_per_update={samples_per_update}")
    
    # Try to load existing state
    loaded = load_latest_state()
    
    if not loaded:
        # Initialize fresh state if no existing state found
        initialize_price_state()
    
    # Track update count
    update_count = 0
    
    # Define function to handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received shutdown signal, stopping continuous generation")
        logger.info(f"Generated {update_count} updates in this session")
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        while True:
            # Check if we've reached the maximum number of updates
            if max_updates is not None and update_count >= max_updates:
                logger.info(f"Reached maximum updates ({max_updates}). Stopping.")
                break
            
            # Generate update
            update_count += 1
            logger.info(f"Generating update #{update_count}")
            
            # Generate data
            df = generate_stock_data(
                num_samples=samples_per_update,
                continuation=True
            )
            
            # Save the data
            result = save_data(df, mode='continuous', update_id=update_count)
            
            if not result:
                logger.error(f"Failed to save update #{update_count}")
                return False
            
            # Wait for the next interval, if not the last update
            if max_updates is None or update_count < max_updates:
                logger.info(f"Waiting {interval} seconds until next update...")
                time.sleep(interval)
        
        logger.info("Continuous generation completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error in continuous generation: {e}")
        return False


def main():
    """
    Main execution function with argument parsing
    """
    # Create directories if they don't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate synthetic stock data for testing and simulation"
    )
    
    # Add arguments
    parser.add_argument(
        "--mode",
        choices=["bulk", "continuous"],
        default="bulk",
        help="Generation mode: bulk or continuous"
    )
    
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples to generate in bulk mode"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Seconds between updates in continuous mode"
    )
    
    parser.add_argument(
        "--samples-per-update",
        type=int,
        default=5,
        help="Number of samples per update in continuous mode"
    )
    
    parser.add_argument(
        "--max-updates",
        type=int,
        default=None,
        help="Maximum number of updates in continuous mode (default: unlimited)"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default=STOCK_SYMBOL,
        help=f"Stock symbol to use (default: {STOCK_SYMBOL})"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Log startup information
    logger.info("=" * 40)
    logger.info(f"Test Data Generator starting in {args.mode} mode")
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Data directory: {RAW_DATA_DIR}")
    logger.info("=" * 40)
    
    # Override stock symbol if provided
    if args.symbol != STOCK_SYMBOL:
        STOCK_SYMBOL = args.symbol
        logger.info(f"Using custom stock symbol: {STOCK_SYMBOL}")
    
    # Execute based on mode
    if args.mode == "bulk":
        logger.info(f"Running bulk generation with {args.samples} samples")
        result = bulk_generation(samples=args.samples)
    else:
        logger.info(f"Running continuous generation with {args.interval}s interval")
        result = continuous_generation(
            interval=args.interval,
            samples_per_update=args.samples_per_update,
            max_updates=args.max_updates
        )
    
    # Log result
    if result:
        logger.info("Test data generation completed successfully")
        return 0
    else:
        logger.error("Test data generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
