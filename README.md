# Stock Price Predictor

A real-time stock price prediction system that uses Kafka, machine learning, and real-time data streaming to predict stock prices and provide trading recommendations for multiple stocks simultaneously.

![Dashboard Preview](https://via.placeholder.com/800x400?text=Stock+Predictor+Dashboard)

## Features

- 📈 **Multi-Stock Support**: Track and predict prices for multiple stocks simultaneously
- 🔄 **Real-time Data**: Fetches real-time stock price data from Finnhub API
- 🤖 **Continuous Learning**: Models are trained and updated in real-time with new data
- 📊 **Interactive Dashboard**: Visualizes data and predictions with Streamlit
- 💰 **Trading Signals**: Provides buy/sell recommendations based on predictions
- 🐳 **Containerized**: Fully dockerized for easy deployment and scaling

## Architecture Overview

The system consists of three main components:

1. **Data Collector**: Continuously fetches real-time stock data from Finnhub API for multiple stocks and publishes it to Kafka
2. **Model Trainer**: Consumes data from Kafka, processes it, and maintains separate ML models for each stock
3. **Dashboard**: Visualizes the data, predictions, and model performance metrics with stock selection capability

![Architecture Diagram](https://via.placeholder.com/800x400?text=Architecture+Diagram)

## Supported Stocks

The system currently supports the following stocks:
- AAPL (Apple Inc.)
- MSFT (Microsoft Corporation)
- AMZN (Amazon.com Inc.)
- GOOGL (Alphabet Inc.)
- META (Meta Platforms Inc.)
- TSLA (Tesla Inc.)
- NVDA (NVIDIA Corporation)
- JPM (JPMorgan Chase & Co.)
- V (Visa Inc.)
- WMT (Walmart Inc.)

## Prerequisites

- Docker and Docker Compose
- Finnhub API key (register for free at [finnhub.io](https://finnhub.io/))
- 4GB+ RAM available for Docker
- Internet connection for API access

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock_predictor.git
cd stock_predictor
```

### 2. Configure Environment Variables

Copy the example environment file and edit as needed:

```bash
cp .env.example .env
```

Required configuration:
- `FINNHUB_API_KEY`: Your Finnhub API key

Optional configuration:
- `PREDICTION_INTERVAL`: Time between API calls in seconds (default: 60)
- `MIN_TRAINING_SAMPLES`: Minimum samples required for training (default: 30)
- `MODEL_RETRAIN_PERIOD`: Minutes between model retraining (default: 120)
- `UI_REFRESH_RATE`: Dashboard refresh interval in seconds (default: 5)

### 3. Start the Services

```bash
# Build the images
docker-compose build

# Start all services
docker-compose up -d
```

This will start all required services:
- ZooKeeper
- Kafka
- Data Collector (collecting data for all supported stocks)
- Model Trainer (training separate models for each stock)
- Streamlit Dashboard

### 4. Access the Dashboard

Open your browser and navigate to:

```
http://localhost:8501
```

The dashboard allows you to:
- Select different stocks to view
- See real-time price updates
- View predictions and trading recommendations
- Monitor model performance metrics
- Manually trigger model retraining

## Model Training

The system uses a sophisticated approach to maintain accurate predictions:

1. **Initial Training**: When first launched, the system will collect a minimum number of data points (default: 30) before training the initial model for each stock.

2. **Continuous Updates**: As new data arrives:
   - Data is collected for all supported stocks
   - Each stock's model is updated periodically with new data
   - Performance metrics are tracked separately for each stock
   - Trading recommendations are generated based on the latest predictions

3. **Performance Monitoring**: The dashboard shows:
   - Model accuracy metrics (MSE, R², Direction Accuracy)
   - Prediction confidence levels
   - Historical performance charts
   - Real-time vs predicted price comparisons

## Trading Recommendations

The system provides trading recommendations based on:
- Price movement predictions
- Model confidence levels
- Historical accuracy
- Market volatility

Recommendations include:
- BUY/SELL signals with strength indicators (STRONG/MODERATE)
- Predicted price movements with confidence levels
- Risk assessments based on model performance

## Troubleshooting

### Common Issues

#### 1. No Data Available
- Check your Finnhub API key in `.env` file
- Verify internet connectivity
- Check data collector logs: `docker logs data-collector`
- Ensure the Finnhub API rate limits haven't been exceeded

#### 2. Model Training Issues
- Ensure enough data points are collected (minimum 30 per stock)
- Check model trainer logs: `docker logs model-trainer`
- Verify system resources (memory/CPU)
- Check if model files are being created in the data/models directory

#### 3. Dashboard Not Updating
- Check if data collection is running
- Verify Kafka connectivity
- Check dashboard logs: `docker logs dashboard`
- Try adjusting the UI refresh rate in `.env`

#### 4. Service Startup Issues
If services fail to start:
1. Stop all services:
```bash
docker-compose down
```

2. Clean up volumes:
```bash
docker-compose down -v
```

3. Rebuild and restart:
```bash
docker-compose build --no-cache
docker-compose up -d
```

### Viewing Logs

```bash
# View logs for specific services
docker logs data-collector
docker logs model-trainer
docker logs dashboard

# Follow logs in real-time
docker logs -f dashboard

# View all service logs
docker-compose logs -f
```

### Data Persistence

Data is stored in the following locations:
- Raw stock data: `data/raw/`
- Trained models: `data/models/`
- Performance metrics: `data/processed/`

To reset all data:
```bash
rm -rf data/*
docker-compose down -v
docker-compose up -d
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Finnhub API](https://finnhub.io/) for providing stock market data
- [Apache Kafka](https://kafka.apache.org/) for streaming infrastructure
- [Streamlit](https://streamlit.io/) for the interactive dashboard framework
- [scikit-learn](https://scikit-learn.org/) for machine learning tools
