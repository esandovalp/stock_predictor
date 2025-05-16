"""
Configuration module for the Stock Predictor application.

This module provides configuration parameters for all components of the application:
- API access
- Data collection
- Kafka streaming
- Spark processing
- ML model settings
- Data storage
- Logging
- Streamlit dashboard

Configuration values can be overridden using environment variables or a .env file.
"""

import os
import logging
import socket
from datetime import timedelta
from pathlib import Path
from dotenv import load_dotenv

# Determine if we're running in Docker
IN_DOCKER = os.path.exists('/.dockerenv')

# Determine environment (development/production)
ENV = os.getenv('ENV', 'development')
IS_PROD = ENV.lower() == 'production'

# Base directory of the project
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from .env file
load_dotenv(BASE_DIR / '.env')

#-----------------------------------------------------------------------
# Application Settings
#-----------------------------------------------------------------------

# Enable/disable debug mode
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_LEVEL_OBJ = getattr(logging, LOG_LEVEL)

# Default host and port
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8501'))

# App name and version
APP_NAME = os.getenv('APP_NAME', 'Stock Price Predictor')
APP_VERSION = os.getenv('APP_VERSION', '1.0.0')

#-----------------------------------------------------------------------
# Finnhub API Configuration
#-----------------------------------------------------------------------

# Finnhub API Key (required for accessing stock data)
# Default is provided for development, but should be overridden in production
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY', 'd0ik8mpr01qrfsagmplgd0ik8mpr01qrfsagmpm0')

# Stock symbol to track
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'AAPL')

# API endpoint base URL
FINNHUB_API_URL = os.getenv('FINNHUB_API_URL', 'https://finnhub.io/api/v1')

# API request timeout in seconds
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '10'))

# Maximum API retries
API_MAX_RETRIES = int(os.getenv('API_MAX_RETRIES', '3'))

#-----------------------------------------------------------------------
# Data Collection Parameters
#-----------------------------------------------------------------------

# Interval between API calls in seconds
PREDICTION_INTERVAL = int(os.getenv('PREDICTION_INTERVAL', '60'))

# Number of days of historical data to fetch on startup
HISTORICAL_DAYS = int(os.getenv('HISTORICAL_DAYS', '7'))

# Webhook detection of significant price changes
PRICE_CHANGE_THRESHOLD = float(os.getenv('PRICE_CHANGE_THRESHOLD', '0.5'))  # percentage

#-----------------------------------------------------------------------
# Kafka Configuration
#-----------------------------------------------------------------------

# Kafka topic for stock data
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'stock_data')

# Kafka broker address
KAFKA_BROKER = os.getenv('KAFKA_BROKER', 'kafka:29092' if IN_DOCKER else 'localhost:9092')

# Kafka producer configuration
KAFKA_PRODUCER_CONFIG = {
    'bootstrap_servers': [KAFKA_BROKER],
    'acks': os.getenv('KAFKA_ACKS', 'all'),
    'retries': int(os.getenv('KAFKA_RETRIES', '3')),
    'batch_size': int(os.getenv('KAFKA_BATCH_SIZE', '16384')),
    'linger_ms': int(os.getenv('KAFKA_LINGER_MS', '5')),
    'request_timeout_ms': int(os.getenv('KAFKA_REQUEST_TIMEOUT_MS', '30000')),
    'security_protocol': os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT'),
}

# Kafka consumer configuration
KAFKA_CONSUMER_CONFIG = {
    'bootstrap_servers': [KAFKA_BROKER],
    'group_id': os.getenv('KAFKA_CONSUMER_GROUP', 'stock-predictor'),
    'auto_offset_reset': os.getenv('KAFKA_OFFSET_RESET', 'latest'),
    'enable_auto_commit': os.getenv('KAFKA_AUTO_COMMIT', 'True').lower() == 'true',
    'security_protocol': os.getenv('KAFKA_SECURITY_PROTOCOL', 'PLAINTEXT'),
}

#-----------------------------------------------------------------------
# Spark Configuration
#-----------------------------------------------------------------------

# Spark master URL
SPARK_MASTER = os.getenv('SPARK_MASTER', 'spark://spark-master:7077' if IN_DOCKER else 'local[*]')

# Application name in Spark UI
SPARK_APP_NAME = os.getenv('SPARK_APP_NAME', 'StockPriceAnalytics')

# Spark executor configuration
SPARK_EXECUTOR_MEMORY = os.getenv('SPARK_EXECUTOR_MEMORY', '1g')
SPARK_EXECUTOR_CORES = int(os.getenv('SPARK_EXECUTOR_CORES', '1'))

# Spark driver configuration
SPARK_DRIVER_MEMORY = os.getenv('SPARK_DRIVER_MEMORY', '1g')

# Spark session configurations
SPARK_CONFIG = {
    'spark.executor.memory': SPARK_EXECUTOR_MEMORY,
    'spark.executor.cores': str(SPARK_EXECUTOR_CORES),
    'spark.driver.memory': SPARK_DRIVER_MEMORY,
    'spark.sql.streaming.checkpointLocation': 'checkpoint',
    'spark.speculation': 'false',
}

#-----------------------------------------------------------------------
# Machine Learning Model Configuration
#-----------------------------------------------------------------------

# Minimum number of data points required for training
MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '30'))

# Period to retrain the model in minutes
MODEL_RETRAIN_PERIOD = int(os.getenv('MODEL_RETRAIN_PERIOD', '120')) 

# Confidence threshold for buy/sell recommendations
RECOMMENDATION_THRESHOLD = float(os.getenv('RECOMMENDATION_THRESHOLD', '0.5'))

# Feature columns used for prediction
MODEL_FEATURES = ['prev_close_price', 'open_price', 'high_price', 'low_price']

# Target column for prediction
MODEL_TARGET = 'current_price'

# Test set size (percentage)
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))

#-----------------------------------------------------------------------
# Data Storage Configuration
#-----------------------------------------------------------------------

# Base data directory
DATA_DIR = os.getenv('DATA_DIR', str(BASE_DIR / 'data'))

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Model storage
MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Model file path
MODEL_PATH = os.path.join(MODEL_DIR, f'{STOCK_SYMBOL.lower()}_price_model.joblib')

# Raw data storage
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

# Processed data storage
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

# Log directory
LOG_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Data files
STOCK_DATA_FILE = os.path.join(RAW_DATA_DIR, f'{STOCK_SYMBOL.lower()}_stock_data.csv')
MODEL_METRICS_FILE = os.path.join(PROCESSED_DATA_DIR, 'model_metrics.csv')
PREDICTIONS_FILE = os.path.join(PROCESSED_DATA_DIR, 'predictions.csv')

#-----------------------------------------------------------------------
# Streamlit Dashboard Configuration
#-----------------------------------------------------------------------

# Dashboard title
DASHBOARD_TITLE = os.getenv('DASHBOARD_TITLE', f'{STOCK_SYMBOL} Stock Predictor')

# Dashboard theme (light or dark)
DASHBOARD_THEME = os.getenv('DASHBOARD_THEME', 'light')

# Charts configuration
CHART_HEIGHT = int(os.getenv('CHART_HEIGHT', '400'))
CHART_WIDTH = int(os.getenv('CHART_WIDTH', '800'))

# Default time window for dashboard display (in hours)
DISPLAY_TIMEFRAME = int(os.getenv('DISPLAY_TIMEFRAME', '24'))

# UI refresh rate in seconds
UI_REFRESH_RATE = int(os.getenv('UI_REFRESH_RATE', '5'))

# Whether to use live data or static data
USE_LIVE_DATA = os.getenv('USE_LIVE_DATA', 'True').lower() == 'true'

# Number of predictions to show on dashboard
NUM_PREDICTIONS_TO_SHOW = int(os.getenv('NUM_PREDICTIONS_TO_SHOW', '10'))

#-----------------------------------------------------------------------
# Environment-specific overrides
#-----------------------------------------------------------------------

if IS_PROD:
    # Production settings
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    LOG_LEVEL_OBJ = logging.WARNING
    # Production should always use secure credentials
    if not os.getenv('FINNHUB_API_KEY'):
        raise ValueError("FINNHUB_API_KEY must be set in production environment")
else:
    # Development settings
    DEBUG = True  # Override any env setting in development
    # Development defaults can remain


# Helper function to get hostname for debugging
def get_hostname():
    """Return the hostname of the current machine"""
    return socket.gethostname()


# Export version info
def get_version_info():
    """Return app version information as a dict"""
    return {
        'app_name': APP_NAME,
        'version': APP_VERSION,
        'environment': ENV,
        'debug': DEBUG,
        'hostname': get_hostname(),
    }


# Print configuration summary if this module is run directly
if __name__ == "__main__":
    print(f"Running with configuration:")
    print(f"- Environment: {ENV}")
    print(f"- Debug mode: {DEBUG}")
    print(f"- Logging level: {LOG_LEVEL}")
    print(f"- Stock symbol: {STOCK_SYMBOL}")
    print(f"- Prediction interval: {PREDICTION_INTERVAL} seconds")
    print(f"- Data directory: {DATA_DIR}")
    print(f"- Kafka broker: {KAFKA_BROKER}")
    print(f"- Spark master: {SPARK_MASTER}")
    print(f"- Running in Docker: {IN_DOCKER}")

# Available stocks for prediction
AVAILABLE_STOCKS = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'AMZN': 'Amazon.com Inc.',
    'GOOGL': 'Alphabet Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'V': 'Visa Inc.',
    'WMT': 'Walmart Inc.'
}

# Default stock symbol
STOCK_SYMBOL = os.getenv('STOCK_SYMBOL', 'AAPL')

# Number of predictions to show in dashboard
NUM_PREDICTIONS_TO_SHOW = int(os.getenv('NUM_PREDICTIONS_TO_SHOW', '10'))

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:29092')
KAFKA_TOPIC = os.getenv('KAFKA_TOPIC', 'stock_data')

# Model configuration
MODEL_TARGET = 'current_price'
TEST_SIZE = float(os.getenv('TEST_SIZE', '0.2'))

# Data directories
DATA_DIR = os.getenv('DATA_DIR', str(BASE_DIR / 'data'))
os.makedirs(DATA_DIR, exist_ok=True)

MODEL_DIR = os.path.join(DATA_DIR, 'models')
os.makedirs(MODEL_DIR, exist_ok=True)

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

LOG_DIR = os.path.join(DATA_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Model training configuration
MIN_TRAINING_SAMPLES = int(os.getenv('MIN_TRAINING_SAMPLES', '30'))
MODEL_RETRAIN_PERIOD = int(os.getenv('MODEL_RETRAIN_PERIOD', '120'))  # minutes

# Dashboard configuration
DASHBOARD_TITLE = os.getenv('DASHBOARD_TITLE', 'Stock Price Predictor')
DASHBOARD_THEME = os.getenv('DASHBOARD_THEME', 'light')
CHART_HEIGHT = int(os.getenv('CHART_HEIGHT', '400'))
CHART_WIDTH = int(os.getenv('CHART_WIDTH', '800'))
DISPLAY_TIMEFRAME = int(os.getenv('DISPLAY_TIMEFRAME', '24'))
UI_REFRESH_RATE = int(os.getenv('UI_REFRESH_RATE', '5'))

# System configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
ENV = os.getenv('ENV', 'development')

def get_version_info():
    """Get version information for the application"""
    return {
        'app_name': APP_NAME,
        'version': APP_VERSION,
        'environment': ENV,
        'hostname': os.uname().nodename
    }
