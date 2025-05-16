import pandas as pd
import os
from app.ml_model import StockPricePredictor
from app.config import STOCK_DATA_FILE, DATA_DIR, STOCK_SYMBOL

def test_model_setup():
    """Test model initialization and training"""
    print("=== Testing Model Setup ===")
    
    # Check if data file exists
    data_path = STOCK_DATA_FILE
    print(f"Looking for data at: {data_path}")
    
    # Load data directly first
    try:
        df = pd.read_csv(data_path)
        print(f"Successfully loaded {len(df)} data points")
        print("\nFirst 3 rows:")
        print(df.head(3))
        print("\nData types:")
        print(df.dtypes)
        
        # Convert numeric columns to float
        numeric_columns = ['current_price', 'open', 'high', 'low', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print("\nData types after conversion:")
        print(df.dtypes)
        
    except Exception as e:
        print(f"Error loading data: {e}")
        
        # Create test data if file not found
        if isinstance(e, FileNotFoundError):
            print("\nCreating sample test data...")
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            
            # Create sample data with correct column names
            sample_data = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=50),
                'symbol': [STOCK_SYMBOL] * 50,
                'current_price': [101 + i*0.5 for i in range(50)],
                'open': [101 + i*0.4 for i in range(50)],
                'high': [102 + i*0.6 for i in range(50)],
                'low': [99 + i*0.3 for i in range(50)],
                'volume': [1000000 + i*10000 for i in range(50)]
            })
            
            # Save sample data
            sample_data.to_csv(data_path, index=False)
            print(f"Created sample data with {len(sample_data)} rows at {data_path}")
            df = sample_data
        else:
            return

    # Initialize predictor
    print("\nInitializing StockPricePredictor...")
    predictor = StockPricePredictor()
    
    # Try training
    print("\nAttempting to train model...")
    try:
        # Pass the DataFrame directly to train
        metrics = predictor.train(df)
        print("\nTraining metrics:")
        print(metrics)
        
        if metrics.get("status") == "success":
            print("\nModel training successful!")
        else:
            print(f"\nModel training status: {metrics.get('status')}")
            
        # Try making a prediction with the latest data point
        if len(df) > 0:
            print("\nTesting prediction with latest data point...")
            latest_data = df.iloc[-1:]  # Get last row as DataFrame
            print("\nLatest data:")
            print(latest_data)
            print("\nLatest data types:")
            print(latest_data.dtypes)
            prediction = predictor.predict(latest_data)
            print("Prediction result:")
            for key, value in prediction.items():
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    test_model_setup()

