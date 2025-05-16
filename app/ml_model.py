import logging
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from app.config import MODEL_PATH, DATA_DIR, STOCK_SYMBOL, STOCK_DATA_FILE, MODEL_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StockPricePredictor:
    def __init__(self, stock_symbol='AAPL', model_path=None):
        """Initialize the stock price predictor
        
        Args:
            stock_symbol (str): Stock symbol to predict (default: 'AAPL')
            model_path (str): Optional path to load a specific model
        """
        self.stock_symbol = stock_symbol if stock_symbol else 'AAPL'
        self.model = None
        self.scaler = None
        
        # Use provided model path or construct from stock symbol
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = os.path.join(MODEL_DIR, f'{self.stock_symbol.lower()}_price_model.joblib')
            
        # Construct data file path based on stock symbol
        self.data_file = os.path.join(RAW_DATA_DIR, f'{self.stock_symbol.lower()}_stock_data.csv')
        self.metrics_file = os.path.join(PROCESSED_DATA_DIR, f'{self.stock_symbol.lower()}_model_metrics.csv')
        self.min_training_samples = 30  # Minimum number of samples required for training
        
        # Create directories if they don't exist
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        
        # Initialize metrics tracking
        self.initialize_metrics_tracking()
        
        # Try to load existing model and scaler
        self.load_model()

    def initialize_metrics_tracking(self):
        """Initialize or load the model metrics tracking file"""
        try:
            if os.path.exists(self.metrics_file):
                self.metrics_df = pd.read_csv(self.metrics_file)
                logger.info(f"Loaded existing metrics file with {len(self.metrics_df)} records")
            else:
                # Create new dataframe if file doesn't exist
                self.metrics_df = pd.DataFrame(columns=[
                    'timestamp', 'mse', 'r2', 'samples', 'prediction_accuracy'
                ])
                logger.info("Created new metrics tracking file")
                self.metrics_df.to_csv(self.metrics_file, index=False)
        except Exception as e:
            logger.warning(f"Could not load metrics file: {e}")
            self.metrics_df = pd.DataFrame(columns=[
                'timestamp', 'mse', 'r2', 'samples', 'prediction_accuracy'
            ])

    def load_model(self):
        """Load the trained model and scaler from disk"""
        try:
            if os.path.exists(self.model_path):
                loaded_data = joblib.load(self.model_path)
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    self.model, self.scaler = loaded_data
                    logger.info("Loaded pre-trained model successfully")
                    return True
                else:
                    logger.warning("Invalid model file format")
                    self.model = None
                    self.scaler = None
            else:
                logger.info("No pre-trained model found")
                self.model = None
                self.scaler = None
        except Exception as e:
            logger.warning(f"Could not load model: {e}")
            self.model = None
            self.scaler = None
        return False

    def load_data(self):
        """Load stock data from CSV file"""
        try:
            if os.path.exists(self.data_file):
                df = pd.read_csv(self.data_file)
                
                # Ensure required columns exist
                required_columns = ['timestamp', 'symbol', 'current_price', 'open', 'high', 'low', 'volume']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"Missing required columns: {missing_columns}")
                    return pd.DataFrame()
                
                # Convert timestamp to datetime
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # Sort by timestamp
                df = df.sort_values('timestamp')
                
                logger.info(f"Loaded {len(df)} data points")
                return df
            else:
                logger.warning(f"Data file not found: {self.data_file}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()

    def save_model(self):
        """Save the current model to disk"""
        if self.model is not None:
            try:
                # Save both the model and scaler
                joblib.dump((self.model, self.scaler), self.model_path)
                logger.info(f"Model saved to {self.model_path}")
                return True
            except Exception as e:
                logger.error(f"Error saving model: {e}")
                return False
        else:
            logger.warning("No model to save")
            return False

    def preprocess_data(self, data):
        """Preprocess the data for training or prediction"""
        try:
            # Convert input to DataFrame if it's a string (file path) or Series
            if isinstance(data, str):
                if os.path.exists(data):
                    data = pd.read_csv(data)
                else:
                    raise ValueError(f"File not found: {data}")
            elif isinstance(data, pd.Series):
                data = data.to_frame().T
            elif not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a DataFrame, Series, or path to a CSV file")
            
            # Create a copy to avoid SettingWithCopyWarning
            data = data.copy()
            
            # Ensure all required features exist
            features = ['current_price', 'open', 'high', 'low', 'volume']
            for feature in features:
                if feature not in data.columns:
                    # Map column names if they exist with slightly different names
                    if feature == 'current_price' and 'close' in data.columns:
                        data['current_price'] = data['close']
                    elif feature == 'high' and 'high_price' in data.columns:
                        data['high'] = data['high_price']
                    elif feature == 'low' and 'low_price' in data.columns:
                        data['low'] = data['low_price']
                    else:
                        data[feature] = 0.0
            
            # Convert numeric columns to float
            for feature in features:
                data[feature] = pd.to_numeric(data[feature], errors='coerce')
            
            # Extract features in correct order
            X = data[features].values
            
            # Initialize scaler if needed
            if self.scaler is None:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            else:
                X = self.scaler.transform(X)
                
            return X
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise

    def train(self, data=None):
        """Train the model on the provided data"""
        try:
            if data is None:
                data = self.load_data()
                
            if len(data) < self.min_training_samples:
                return {'status': 'error', 'message': f'Not enough training data. Minimum required: {self.min_training_samples}'}
                
            # Prepare features and target
            X = self.preprocess_data(data)
            y = data['current_price'].values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            
            # Calculate metrics
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Save model and scaler
            joblib.dump((self.model, self.scaler), self.model_path)
            
            # Save metrics
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'train_score': train_score,
                'test_score': test_score,
                'num_samples': len(data)
            }
            
            self.update_metrics(metrics)
            
            return {'status': 'success', 'metrics': metrics}
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            return {'status': 'error', 'message': f'Training failed: {str(e)}'}

    def predict(self, data):
        """Make predictions using the trained model"""
        try:
            if self.model is None:
                return {'status': 'error', 'message': 'Model not trained'}
                
            # Convert data to DataFrame if it's a Series
            if isinstance(data, pd.Series):
                data = data.to_frame().T
            elif not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a DataFrame or Series")
            
            # Preprocess data
            X = self.preprocess_data(data)
            
            # Make prediction
            prediction = self.model.predict(X)
            
            # Calculate confidence from model metrics
            confidence = 0.0
            if not self.metrics_df.empty:
                recent_metrics = self.metrics_df.iloc[-1]
                r2 = recent_metrics.get('r2', 0)
                direction_accuracy = recent_metrics.get('prediction_accuracy', 0)
                confidence = min(100, max(0, (r2 * 50 + direction_accuracy * 50))) / 100
            
            return {
                'status': 'success',
                'prediction': prediction[0] if isinstance(prediction, np.ndarray) else prediction,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'status': 'error', 'message': f'Prediction failed: {str(e)}'}

    def update_metrics(self, metrics):
        """Update and save performance metrics
        
        Args:
            metrics: Dictionary with current metrics
        """
        # Create new metrics entry
        new_metrics = {
            'timestamp': datetime.now().isoformat(),
            'mse': metrics.get('mse', None),
            'r2': metrics.get('r2', None),
            'samples': metrics.get('samples', 0),
            'prediction_accuracy': metrics.get('direction_accuracy', None)
        }
        
        # Add to metrics dataframe
        new_row = pd.DataFrame([new_metrics])
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        
        # Save updated metrics
        self.metrics_df.to_csv(self.metrics_file, index=False)
        logger.debug("Updated model metrics")

    def get_buy_sell_recommendation(self, predicted_price, data):
        """Generate buy/sell/hold recommendation based on prediction
        
        Args:
            predicted_price: The predicted stock price
            data: Input data used for prediction
            
        Returns:
            String with recommendation and confidence
        """
        # Extract current price
        if isinstance(data, pd.DataFrame) and 'current_price' in data.columns:
            current_price = data.iloc[-1]['current_price']
        elif isinstance(data, dict) and 'current_price' in data:
            current_price = data['current_price']
        else:
            return "HOLD - Insufficient data"
        
        # Calculate prediction metrics
        price_diff = predicted_price - current_price
        pct_change = (price_diff / current_price) * 100
        
        # Get model confidence from recent metrics
        if not self.metrics_df.empty:
            recent_metrics = self.metrics_df.iloc[-1]
            r2 = recent_metrics.get('r2', 0)
            direction_accuracy = recent_metrics.get('prediction_accuracy', 0)
            
            # Calculate confidence score (0-100)
            confidence = min(100, max(0, (r2 * 50 + direction_accuracy * 50)))
        else:
            confidence = 0
        
        # Determine threshold based on confidence
        threshold = 0.5 + (1.5 * (1 - confidence/100))  # Higher threshold when confidence is low
        
        # Generate recommendation
        if pct_change > threshold:
            strength = "STRONG" if pct_change > threshold * 2 else "MODERATE"
            return f"BUY - {strength} ({pct_change:.2f}% upside predicted, {confidence:.1f}% confidence)"
        elif pct_change < -threshold:
            strength = "STRONG" if pct_change < -threshold * 2 else "MODERATE"
            return f"SELL - {strength} ({-pct_change:.2f}% downside predicted, {confidence:.1f}% confidence)"
        else:
            return f"HOLD - Price expected to remain stable ({pct_change:.2f}%, {confidence:.1f}% confidence)"


if __name__ == "__main__":
    # Simple test
    predictor = StockPricePredictor()
    
    # Check if we have enough data to train
    try:
        df = pd.read_csv(f"{DATA_DIR}/{STOCK_SYMBOL}_stock_data.csv")
        if len(df) >= predictor.min_training_samples:
            metrics = predictor.train(df)
            print(f"Training metrics: {metrics}")
            
            # Test prediction on most recent data point
            prediction = predictor.predict(df.iloc[-1])
            print(f"Prediction: {prediction}")
        else:
            print(f"Not enough data for training. Have {len(df)} samples, need {predictor.min_training_samples}.")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("No data file found. Please run data collection first.")

