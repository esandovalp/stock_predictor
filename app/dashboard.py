"""
Streamlit dashboard for the Stock Price Predictor application.

This module provides a web-based dashboard to visualize:
- Real-time stock price data
- Predictive model outputs and accuracy
- Buy/sell recommendations
- Historical performance metrics

The dashboard auto-refreshes to show the latest data and predictions.
"""

import os
import time
import datetime
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz

# Import from our application modules
from app.config import (
    STOCK_SYMBOL as DEFAULT_STOCK_SYMBOL, DASHBOARD_TITLE, DASHBOARD_THEME, CHART_HEIGHT, CHART_WIDTH,
    DISPLAY_TIMEFRAME, UI_REFRESH_RATE, NUM_PREDICTIONS_TO_SHOW, 
    STOCK_DATA_FILE as DEFAULT_STOCK_DATA_FILE, MODEL_METRICS_FILE, PREDICTIONS_FILE, MODEL_PATH as DEFAULT_MODEL_PATH,
    DATA_DIR, USE_LIVE_DATA, APP_NAME, APP_VERSION, get_version_info, RAW_DATA_DIR, MODEL_DIR, PROCESSED_DATA_DIR
)
from app.ml_model import StockPricePredictor

# Configure the Streamlit page
st.set_page_config(
    page_title=DASHBOARD_TITLE,
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/stock_predictor',
        'Report a bug': 'https://github.com/yourusername/stock_predictor/issues',
        'About': f"{APP_NAME} v{APP_VERSION} - Real-time stock price prediction dashboard"
    }
)

# Apply theme
if DASHBOARD_THEME == 'dark':
    st.markdown("""
        <style>
        .reportview-container {
            background-color: #0e1117;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# List of available stock symbols
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

# Function to get the stock data file path for a specific symbol
def get_stock_data_file(symbol):
    """Get the stock data file path for the specified symbol"""
    return os.path.join(RAW_DATA_DIR, f'{symbol.lower()}_stock_data.csv')

# Function to get the model path for a specific symbol
def get_model_path(symbol):
    """Get the model file path for the specified symbol"""
    return os.path.join(MODEL_DIR, f'{symbol.lower()}_price_model.joblib')

# Function to load stock data
@st.cache_data(ttl=UI_REFRESH_RATE)
def load_stock_data(symbol):
    """Load stock data for the given symbol"""
    try:
        file_path = os.path.join(RAW_DATA_DIR, f'{symbol.lower()}_stock_data.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()

# Function to load model metrics
@st.cache_data(ttl=UI_REFRESH_RATE)
def load_model_metrics(symbol):
    """Load model performance metrics from CSV"""
    try:
        if os.path.exists(MODEL_METRICS_FILE):
            df = pd.read_csv(MODEL_METRICS_FILE)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading model metrics: {e}")
        return pd.DataFrame()

# Function to get predictions using the model
def get_predictions(stock_data, symbol):
    """Get predictions for the stock data"""
    try:
        predictor = StockPricePredictor(stock_symbol=symbol)
        
        if stock_data.empty:
            return pd.DataFrame()
            
        predictions = []
        for _, row in stock_data.iterrows():
            pred = predictor.predict(row)
            if pred['status'] == 'success':
                predictions.append({
                    'timestamp': row['timestamp'],
                    'actual_price': row['current_price'],
                    'predicted_price': pred['prediction'],
                    'confidence': pred.get('confidence', 0.0)  # Default to 0.0 if confidence is not available
                })
                
        return pd.DataFrame(predictions)
    except Exception as e:
        st.error(f"Error getting predictions: {str(e)}")
        return pd.DataFrame()

# Function to create real-time price chart
def create_price_chart(stock_data, symbol):
    """Create interactive price chart with Plotly"""
    if stock_data.empty:
        return None
    
    # Filter to the display timeframe
    cutoff_time = datetime.now() - timedelta(hours=DISPLAY_TIMEFRAME)
    filtered_data = stock_data[stock_data['timestamp'] > cutoff_time]
    
    if filtered_data.empty:
        filtered_data = stock_data.tail(30)  # Show at least some data
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add price line
    fig.add_trace(
        go.Scatter(
            x=filtered_data['timestamp'],
            y=filtered_data['current_price'],
            name=f"{symbol} Price",
            line=dict(color='#2962FF', width=2),
            hovertemplate='%{y:.2f} USD<extra></extra>',
        ),
        secondary_y=False
    )
    
    # Add volume as bar chart
    if 'volume' in filtered_data.columns:
        fig.add_trace(
            go.Bar(
                x=filtered_data['timestamp'],
                y=filtered_data['volume'],
                name="Volume",
                marker_color='rgba(58, 71, 80, 0.3)',
                hovertemplate='%{y:,.0f}<extra></extra>'
            ),
            secondary_y=True
        )
    
    # Add day's range indicators
    fig.add_trace(
        go.Scatter(
            x=filtered_data['timestamp'],
            y=filtered_data['high_price'],
            name="High",
            line=dict(color='rgba(0, 255, 0, 0.2)', width=1, dash='dot'),
            hovertemplate='%{y:.2f} USD<extra></extra>',
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=filtered_data['timestamp'],
            y=filtered_data['low_price'],
            name="Low",
            line=dict(color='rgba(255, 0, 0, 0.2)', width=1, dash='dot'),
            hovertemplate='%{y:.2f} USD<extra></extra>',
            fill='tonexty',
            fillcolor='rgba(0, 100, 80, 0.05)'
        ),
        secondary_y=False
    )
    
    # Set chart layout
    fig.update_layout(
        title=f"{symbol} Real-Time Price",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.3)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.3)',
        secondary_y=False
    )
    
    fig.update_yaxes(
        showgrid=False,
        title_text="Volume",
        secondary_y=True
    )
    
    return fig

# Function to create prediction chart
def create_prediction_chart(stock_data, predictions_df):
    """Create chart showing actual vs predicted prices"""
    if stock_data.empty or predictions_df.empty:
        return None
    
    # Merge predictions with actual data
    merged_data = pd.merge(
        stock_data, 
        predictions_df[['timestamp', 'predicted_price']], 
        on='timestamp', 
        how='left'
    )
    
    # Filter to the recent data points
    recent_data = merged_data.tail(NUM_PREDICTIONS_TO_SHOW).copy()
    
    # Create figure
    fig = go.Figure()
    
    # Add actual price line
    fig.add_trace(
        go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['current_price'],
            name="Actual Price",
            line=dict(color='#2962FF', width=2),
            hovertemplate='%{y:.2f} USD<extra></extra>'
        )
    )
    
    # Add predicted price line
    fig.add_trace(
        go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['predicted_price'],
            name="Predicted Price",
            line=dict(color='#FF6D00', width=2, dash='dash'),
            hovertemplate='%{y:.2f} USD<extra></extra>'
        )
    )
    
    # Set chart layout
    fig.update_layout(
        title="Actual vs Predicted Prices",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        hovermode="x unified",
        height=CHART_HEIGHT,
        width=CHART_WIDTH,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.3)'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.3)'
    )
    
    return fig

# Function to create performance metrics chart
def create_metrics_chart(metrics_df):
    """Create chart showing model performance over time"""
    if metrics_df.empty:
        return None
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add MSE line
    fig.add_trace(
        go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['mse'],
            name="MSE",
            line=dict(color='#FF6D00', width=2),
            hovertemplate='%{y:.4f}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Add R² line
    fig.add_trace(
        go.Scatter(
            x=metrics_df['timestamp'],
            y=metrics_df['r2'],
            name="R²",
            line=dict(color='#2962FF', width=2),
            hovertemplate='%{y:.4f}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add prediction accuracy line
    if 'prediction_accuracy' in metrics_df.columns:
        fig.add_trace(
            go.Scatter(
                x=metrics_df['timestamp'],
                y=metrics_df['prediction_accuracy'],
                name="Direction Accuracy",
                line=dict(color='#00C853', width=2),
                hovertemplate='%{y:.2%}<extra></extra>'
            ),
            secondary_y=False
        )
    
    # Set chart layout
    fig.update_layout(
        title="Model Performance Metrics",
        xaxis_title="Time",
        hovermode="x unified",
        height=int(CHART_HEIGHT * 0.75),
        width=CHART_WIDTH,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.3)'
    )
    
    fig.update_yaxes(
        title_text="R² / Accuracy",
        range=[0, 1],
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(211, 211, 211, 0.3)',
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="MSE",
        showgrid=False,
        secondary_y=True
    )
    
    return fig

# Function to display key statistics
def display_key_statistics(stock_data):
    """Display key statistics about the stock"""
    if stock_data.empty:
        st.warning("No stock data available")
        return
    
    # Get the most recent data point and previous point for comparison
    latest = stock_data.iloc[-1]
    
    # Market hours check (US Eastern Time)
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
    
    # Standard market hours are 9:30 AM - 4:00 PM ET, Monday-Friday
    is_market_open = (
        0 <= now_et.weekday() <= 4 and  # Monday-Friday
        datetime.time(9, 30) <= now_et.time() <= datetime.time(16, 0)
    )
    
    # Check for price changes
    is_data_changing = False
    time_since_last_change = None
    
    if len(stock_data) > 1:
        # Compare with previous data points to see if prices are changing
        for i in range(len(stock_data)-1, 0, -1):
            if stock_data.iloc[i]['current_price'] != stock_data.iloc[i-1]['current_price']:
                is_data_changing = True
                # Calculate time since last price change
                time_since_last_change = (latest['timestamp'] - stock_data.iloc[i]['timestamp']).total_seconds() / 60
                break
    
    # Calculate daily change
    daily_change = latest['current_price'] - latest['prev_close_price']
    daily_change_pct = (daily_change / latest['prev_close_price']) * 100
    
    # Calculate distance from high/low
    pct_from_high = ((latest['high_price'] - latest['current_price']) / latest['high_price']) * 100
    pct_from_low = ((latest['current_price'] - latest['low_price']) / latest['low_price']) * 100
    
    # Create a container for market status
    market_status_container = st.container()
    
    # Market status indicator
    if is_market_open:
        market_status = "🟢 Market Open"
        market_color = "green"
    else:
        market_status = "🔴 Market Closed"
        market_color = "red"
    
    # Data freshness indicator
    if not is_data_changing and len(stock_data) > 1:
        data_status = "⚠️ Price data not changing"
        data_color = "orange"
    else:
        data_status = "✅ Data updating normally"
        data_color = "green"
    
    # Display market status
    market_status_container.markdown(f"""
    <div style="display: flex; justify-content: space-between; padding: 10px; border-radius: 5px; background-color: rgba(0, 0, 0, 0.05); margin-bottom: 15px;">
        <div>
            <span style="color: {market_color}; font-weight: bold;">{market_status}</span>
            <span style="margin-left: 20px; color: {data_color};">{data_status}</span>
        </div>
        <div>
            Last data point: {latest['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create four columns for statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Price",
            value=f"${latest['current_price']:.2f}",
            delta=f"{daily_change:.2f} ({daily_change_pct:.2f}%)"
        )
        
        st.metric(
            label="Previous Close",
            value=f"${latest['prev_close_price']:.2f}"
        )
    
    with col2:
        st.metric(
            label="Day's Range",
            value=f"${latest['low_price']:.2f} - ${latest['high_price']:.2f}"
        )
        
        st.metric(
            label="% From Day High",
            value=f"{pct_from_high:.2f}%"
        )
    
    with col3:
        st.metric(
            label="Open Price",
            value=f"${latest['open_price']:.2f}"
        )
        
        st.metric(
            label="% From Day Low",
            value=f"{pct_from_low:.2f}%"
        )
    
    with col4:
        # Add time since last change if applicable
        if time_since_last_change is not None:
            if time_since_last_change < 60:
                time_label = f"{time_since_last_change:.1f} mins ago"
            else:
                time_label = f"{time_since_last_change/60:.1f} hours ago"
            
            st.metric(
                label="Last Price Change",
                value=time_label
            )
        else:
            st.metric(
                label="Price Status",
                value="No changes detected"
            )
        
        # Add data points count
        st.metric(
            label="Data Points Collected",
            value=f"{len(stock_data):,}"
        )
    
    # If market is closed, show explanation
    if not is_market_open:
        st.info("""
        📊 **Market Hours Information**
        
        The stock market is currently closed. Data collection continues but price updates 
        may be infrequent or unchanged until the market reopens.
        
        **Regular Trading Hours (ET):**
        - Monday-Friday: 9:30 AM - 4:00 PM
        - Closed on weekends and holidays
        
        Pre-market and after-hours trading may show some price movements outside regular hours.
        """)

# Function to display recommendations
def display_recommendations(predictions_df):
    """Display trading recommendations based on predictions"""
    if predictions_df.empty:
        st.warning("No predictions available for recommendations")
        return
    
    # Get the most recent prediction
    latest_prediction = predictions_df.iloc[-1]
    
    # Extract the recommendation
    recommendation = latest_prediction['recommendation']
    
    # Determine color based on recommendation
    if "BUY" in recommendation:
        color = "green"
        icon = "📈"
    elif "SELL" in recommendation:
        color = "red"
        icon = "📉"
    else:
        color = "orange"
        icon = "📊"
    
    # Display recommendation
    st.markdown(f"""
    <div style="padding: 15px; border-radius: 5px; background-color: rgba(0, 0, 0, 0.05);">
        <h3 style="color: {color};">{icon} Trading Recommendation</h3>
        <p style="font-size: 18px; font-weight: bold; color: {color};">{recommendation}</p>
        <p>Current Price: ${latest_prediction['current_price']:.2f} | Predicted: ${latest_prediction['predicted_price']:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    # Show historical recommendations
    with st.expander("Historical Recommendations"):
        # Create a dataframe with relevant columns
        recs_df = predictions_df[['timestamp', 'current_price', 'predicted_price', 'price_change_pct', 'recommendation']].copy()
        recs_df['timestamp'] = recs_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Display the dataframe
        st.dataframe(recs_df, use_container_width=True)

# Function to display model statistics summary
def display_model_statistics(metrics_df):
    """Display summary statistics about the model performance"""
    if metrics_df.empty:
        st.warning("No model metrics available")
        return
    
    # Get the most recent metrics
    latest_metrics = metrics_df.iloc[-1]
    
    # Create columns for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="R² Score",
            value=f"{latest_metrics['r2']:.4f}"
        )
    
    with col2:
        st.metric(
            label="Mean Squared Error",
            value=f"{latest_metrics['mse']:.4f}"
        )
    
    with col3:
        if 'prediction_accuracy' in latest_metrics:
            st.metric(
                label="Direction Accuracy",
                value=f"{latest_metrics['prediction_accuracy']:.2%}"
            )
        else:
            st.metric(
                label="Samples",
                value=f"{int(latest_metrics['samples'])}"
            )
    
    # Display training info
    st.markdown(f"""
    <div style="font-size: 14px; color: #666;">
        <p>Model last trained: {latest_metrics['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Training samples: {int(latest_metrics['samples'])}</p>
    </div>
    """, unsafe_allow_html=True)

# Main function to run the dashboard
def main():
    st.title('Stock Price Predictor Dashboard')
    
    # Sidebar
    st.sidebar.title('Settings')
    selected_symbol = st.sidebar.selectbox(
        'Select Stock',
        options=list(AVAILABLE_STOCKS.keys()),
        format_func=lambda x: f"{x} - {AVAILABLE_STOCKS[x]}"
    )
    
    # Load data
    stock_data = load_stock_data(selected_symbol)
    
    if not stock_data.empty:
        # Display current stock info
        current_price = stock_data['current_price'].iloc[-1]
        
        # Calculate previous close (last price from previous day)
        today = pd.Timestamp.now().date()
        prev_data = stock_data[stock_data['timestamp'].dt.date < today]
        if not prev_data.empty:
            prev_close = prev_data['current_price'].iloc[-1]
        else:
            # If no previous day data, use the first price of the day
            prev_close = stock_data['current_price'].iloc[0]
            
        price_change = current_price - prev_close
        price_change_pct = (price_change / prev_close) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Current Price", f"${current_price:.2f}", f"{price_change_pct:+.2f}%")
        col2.metric("Volume", f"{stock_data['volume'].iloc[-1]:,.0f}")
        col3.metric("Previous Close", f"${prev_close:.2f}")
        
        # Get and display predictions
        predictions_df = get_predictions(stock_data, selected_symbol)
        
        if not predictions_df.empty:
            st.subheader('Price Predictions')
            
            # Create price chart
            fig = go.Figure()
            
            # Add actual prices
            fig.add_trace(go.Scatter(
                x=predictions_df['timestamp'],
                y=predictions_df['actual_price'],
                name='Actual Price',
                line=dict(color='blue')
            ))
            
            # Add predicted prices
            fig.add_trace(go.Scatter(
                x=predictions_df['timestamp'],
                y=predictions_df['predicted_price'],
                name='Predicted Price',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'{selected_symbol} Price History and Predictions',
                xaxis_title='Time',
                yaxis_title='Price ($)',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig)
            
            # Display prediction metrics
            if 'confidence' in predictions_df.columns:
                avg_confidence = predictions_df['confidence'].mean()
                st.metric("Average Prediction Confidence", f"{avg_confidence:.2%}")
        else:
            st.warning("No predictions available yet. The model may need more data to train.")
    else:
        st.warning(f"No data available for {selected_symbol}. Please wait for data collection.")


if __name__ == "__main__":
    main()
