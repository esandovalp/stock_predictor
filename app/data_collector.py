"""
Stock data collector that fetches real-time data from Finnhub API
"""

import os
import time
import logging
import pandas as pd
from datetime import datetime, timedelta
import finnhub
from kafka import KafkaProducer
import json

from app.config import (
    FINNHUB_API_KEY, KAFKA_BOOTSTRAP_SERVERS, KAFKA_TOPIC,
    RAW_DATA_DIR, LOG_LEVEL, PREDICTION_INTERVAL,
    AVAILABLE_STOCKS
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockDataCollector:
    def __init__(self):
        try:
            self.finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
            self.producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
                value_serializer=lambda x: json.dumps(x).encode('utf-8')
            )
            
            # Create data directory if it doesn't exist
            os.makedirs(RAW_DATA_DIR, exist_ok=True)
            
            # Initialize data storage for each stock
            self.stock_data = {symbol: pd.DataFrame() for symbol in AVAILABLE_STOCKS.keys()}
            
            # Load existing data for each stock
            for symbol in AVAILABLE_STOCKS.keys():
                self.load_existing_data(symbol)
                
            logger.info(f"Initialized data collector for {len(AVAILABLE_STOCKS)} stocks")
        except Exception as e:
            logger.error(f"Failed to initialize data collector: {e}")
            raise
    
    def load_existing_data(self, symbol):
        """Load existing data for a stock symbol from CSV"""
        file_path = os.path.join(RAW_DATA_DIR, f'{symbol.lower()}_stock_data.csv')
        try:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    self.stock_data[symbol] = df
                    logger.info(f"Loaded {len(df)} existing data points for {symbol}")
        except Exception as e:
            logger.error(f"Error loading existing data for {symbol}: {e}")
    
    def collect_stock_data(self, symbol):
        """Collect real-time stock data for a given symbol"""
        try:
            # Get quote data
            quote = self.finnhub_client.quote(symbol)
            
            # Prepare data point with default values
            data_point = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'current_price': quote.get('c', 0.0),
                'open': quote.get('o', 0.0),
                'high': quote.get('h', 0.0),
                'low': quote.get('l', 0.0),
                'volume': quote.get('v', 0.0)
            }
            
            # Save to CSV
            df = pd.DataFrame([data_point])
            csv_file = os.path.join(RAW_DATA_DIR, f'{symbol.lower()}_stock_data.csv')
            
            # If file exists, append; otherwise create with headers
            if os.path.exists(csv_file):
                # Read existing file to check columns
                try:
                    existing_df = pd.read_csv(csv_file)
                    if list(existing_df.columns) != list(df.columns):
                        # If columns don't match, rewrite the file
                        df.to_csv(csv_file, mode='w', header=True, index=False)
                    else:
                        df.to_csv(csv_file, mode='a', header=False, index=False)
                except Exception as e:
                    logger.error(f"Error reading existing CSV file: {str(e)}")
                    # If there's an error reading the file, rewrite it
                    df.to_csv(csv_file, mode='w', header=True, index=False)
            else:
                df.to_csv(csv_file, mode='w', header=True, index=False)
            
            # Send to Kafka
            self.producer.send(KAFKA_TOPIC, data_point)
            
            logger.info(f"Collected data for {symbol}")
            logger.info(f"Collected data for {symbol}: ${data_point['current_price']:.2f}")
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error collecting data for {symbol}: {str(e)}")
            return None
    
    def run(self):
        """Main loop to continuously collect data"""
        logger.info("Starting stock data collection")
        
        while True:
            try:
                for symbol in AVAILABLE_STOCKS.keys():
                    try:
                        data = self.collect_stock_data(symbol)
                        if data:
                            logger.info(f"Collected data for {symbol}: ${data['current_price']:.2f}")
                        time.sleep(1)  # Small delay between API calls for different stocks
                    except Exception as e:
                        logger.error(f"Error in collection loop for {symbol}: {e}")
                        continue
                
                # Wait for next interval
                time.sleep(PREDICTION_INTERVAL)
            except Exception as e:
                logger.error(f"Error in main collection loop: {e}")
                time.sleep(10)  # Wait before retrying

if __name__ == "__main__":
    try:
        collector = StockDataCollector()
        collector.run()
    except Exception as e:
        logger.critical(f"Fatal error in data collector: {e}")
        raise
