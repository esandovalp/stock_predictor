#!/usr/bin/env python3
"""
Test Data Generator for Stock Price Prediction System

This script generates synthetic stock data with realistic patterns for testing
the stock prediction model. It creates a CSV file with 50 rows of data containing
stock price information and volume.

Usage:
    python generate_test_data.py

The data will be saved to app/data/raw/{symbol}_stock_data.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import random
import logging

# Import configurations from our config file
from app.config import (
    STOCK_SYMBOL,
    DATA_DIR,
    RAW_DATA_DIR,
    STOCK_DATA_FILE,
    MODEL_FEATURES,
    MODEL_TARGET
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_stock_data(symbol, num_samples=50, start_price=180.0, volatility=0.015):
    """
    Generate synthetic stock price data with realistic patterns.
    
    Args:
        symbol (str): Stock symbol
        num_samples (int): Number of data points to generate
        start_price (float): Starting price for the simulation
        volatility (float): Volatility factor for price movements
        
    Returns:
        pd.DataFrame: DataFrame with generated stock data
    """
    # Calculate start date (working backwards from today)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_samples)
    
    # Generate dates (market days only - skip weekends)
    all_dates = []
    current_date = start_date
    while len(all_dates) < num_samples:
        # Skip weekends (0 = Monday, 6 = Sunday)
        if current_date.weekday() < 5:  # Weekday
            all_dates.append(current_date)
        current_date += timedelta(days=1)
    
    # Initialize empty price lists
    timestamps = []
    current_prices = []
    prev_close_prices = []
    open_prices = []
    high_prices = []
    low_prices = []
    volumes = []
    
    # Set initial price
    current_price = start_price
    
    # Generate data with realistic patterns
    for i, date in enumerate(all_dates):
        # Format timestamp
        timestamp = date.isoformat()
        timestamps.append(timestamp)
        
        # First day has special handling
        if i == 0:
            # On first day, we need to set previous close
            prev_close = current_price * (1 + (random.random() - 0.5) * volatility)
            # Open price is based on previous close with some overnight movement
            open_price = prev_close * (1 + (random.random() - 0.5) * volatility * 0.5)
        else:
            # Previous day's close is the previous day's current price
            prev_close = current_prices[-1]
            # Open price is based on previous close with some overnight movement
            open_price = prev_close * (1 + (random.random() - 0.5) * volatility * 0.5)
        
        # Determine daily price movement simulation
        daily_movement = np.random.normal(0, volatility)
        
        # Add some mean-reversion tendency
        mean_price = 180.0
        if current_price > mean_price * 1.1:  # 10% above mean
            # More likely to go down
            daily_movement -= volatility * 0.3
        elif current_price < mean_price * 0.9:  # 10% below mean
            # More likely to go up
            daily_movement += volatility * 0.3
        
        # Calculate current day's price
        current_price = open_price * (1 + daily_movement)
        
        # Ensure current price is positive
        current_price = max(current_price, 0.01)
        
        # High and low prices
        price_range = abs(current_price - open_price) + (current_price * volatility)
        high_price = max(current_price, open_price) + (random.random() * price_range * 0.5)
        low_price = min(current_price, open_price) - (random.random() * price_range * 0.5)
        
        # Ensure low is actually lower than high
        if low_price >= high_price:
            low_price = high_price * 0.99
        
        # Generate volume (higher on volatile days)
        base_volume = 1_000_000  # Base volume for the stock
        volume_volatility = abs(current_price - open_price) / open_price  # More volume on volatile days
        daily_volume = int(base_volume * (1 + volume_volatility * 5) * (0.7 + 0.6 * random.random()))
        
        # Append to lists
        prev_close_prices.append(round(prev_close, 2))
        open_prices.append(round(open_price, 2))
        current_prices.append(round(current_price, 2))
        high_prices.append(round(high_price, 2))
        low_prices.append(round(low_price, 2))
        volumes.append(daily_volume)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'prev_close_price': prev_close_prices,
        'open_price': open_prices,
        'high_price': high_prices,
        'low_price': low_prices,
        'current_price': current_prices,
        'volume': volumes
    })
    
    return df

def main():
    """Generate and save test data"""
    logger.info(f"Generating test data for {STOCK_SYMBOL}")
    
    # Ensure the data directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    # Generate data
    df = generate_stock_data(
        symbol=STOCK_SYMBOL,
        num_samples=50,
        start_price=180.0,  # Approximate AAPL price
        volatility=0.015    # Typical daily volatility
    )
    
    # Save to CSV
    logger.info(f"Saving test data to {STOCK_DATA_FILE}")
    df.to_csv(STOCK_DATA_FILE, index=False)
    
    # Log statistics
    logger.info(f"Generated {len(df)} rows of test data")
    logger.info(f"Price range: ${df['current_price'].min():.2f} - ${df['current_price'].max():.2f}")
    logger.info(f"Average volume: {df['volume'].mean():.0f}")
    
    # Verify the file exists
    if os.path.exists(STOCK_DATA_FILE):
        logger.info(f"File successfully created at {STOCK_DATA_FILE}")
        logger.info(f"Data is ready for testing the prediction model")
    else:
        logger.error(f"Failed to create file at {STOCK_DATA_FILE}")

if __name__ == "__main__":
    main()

