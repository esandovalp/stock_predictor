# Stock Predictor Demo Script

## Pre-Demo Checklist

1. Verify Docker installation:
```bash
docker --version
docker-compose --version
```

2. Start the system:
```bash
./setup.sh start
```

3. Check service status:
```bash
docker-compose ps
```

## Demo Flow (20-25 minutes)

### 1. System Architecture (5 min)
- Show project structure:
```bash
tree -L 2
```
- Explain the key components:
  - Data Collector (Finnhub API)
  - Kafka Message Broker
  - Model Trainer
  - Streamlit Dashboard

### 2. Data Collection (5 min)
- Show real-time data collection:
```bash
docker-compose logs --tail=20 data-collector
```
- Display collected stock data:
```bash
docker-compose exec data-collector head -n 10 /app/data/raw/AAPL_stock_data.csv
```

### 3. Model Training (5 min)
- View model training process:
```bash
docker-compose logs --tail=20 model-trainer
```
- Check model performance:
```bash
docker-compose exec model-trainer cat /app/data/processed/model_metrics.csv
```

### 4. Dashboard Demo (7 min)
1. Open dashboard: http://localhost:8501
2. Key features to demonstrate:
   - Real-time price updates
   - Price predictions
   - Trading recommendations
   - Performance metrics

### 5. System Resilience (3 min)
- Demonstrate system recovery:
```bash
# Restart a component
docker-compose restart data-collector

# Show recovery logs
docker-compose logs --follow data-collector
```

## Architecture Deep Dive

### System Components Architecture
```
┌─────────────────┐     ┌───────────────┐     ┌────────────────┐     ┌───────────────┐
│                 │     │               │     │                │     │               │
│  Finnhub API    │────▶│ Data Collector│────▶│  Kafka Broker  │────▶│ Model Trainer │
│                 │     │               │     │                │     │               │
└─────────────────┘     └───────────────┘     └────────────────┘     └───────┬───────┘
                                                                             │
                                                                             │
                                                                             ▼
                                                        ┌───────────────────────────────┐
                                                        │                               │
                                                        │  Streamlit Dashboard          │
                                                        │                               │
                                                        └───────────────────────────────┘
```

### Real-Time Data Flow Details

1. **Data Collection Layer**
   - Finnhub API Connection (60-second intervals)
   - Data points collected per stock:
     ```json
     {
       "c": "Current price",
       "h": "High price of the day",
       "l": "Low price of the day",
       "o": "Open price",
       "pc": "Previous close price",
       "t": "Timestamp"
     }
     ```
   - Error handling with exponential backoff
   - Rate limiting: 60 API calls/minute

2. **Message Broker Layer (Kafka)**
   - Topics structure:
     - `stock-raw-data`: Raw incoming stock data
     - `stock-processed-data`: Cleaned and normalized data
     - `stock-predictions`: Model predictions
   - Partition strategy: By stock symbol
   - Message retention: 7 days
   - Replication factor: 3

3. **Model Training Layer**
   - Training triggers:
     - Every 30 new data points
     - When accuracy drops below 95%
   - Feature engineering:
     ```python
     features = [
       'price_change',
       'volume_weighted_price',
       'rolling_mean_5min',
       'rolling_std_5min',
       'rsi_14',
       'macd'
     ]
     ```
   - Model metrics tracked:
     - Mean Squared Error (MSE)
     - R-squared (R²)
     - Directional Accuracy

4. **Dashboard Data Flow**
   - Real-time updates:
     - Price data: Every 60 seconds
     - Predictions: Every 5 minutes
     - Model metrics: Every 30 minutes
   - Caching strategy:
     - Short-term cache (5 min): Current prices
     - Medium-term cache (1 hour): Historical data
     - Long-term cache (1 day): Model performance

### Performance Metrics

1. **Latency Targets**
   - Data collection to storage: < 100ms
   - Model prediction generation: < 500ms
   - Dashboard update: < 1s

2. **System Scalability**
   - Current capacity: 10 stocks
   - Maximum capacity per node:
     - Data Collector: 100 stocks
     - Kafka Broker: 10,000 messages/second
     - Model Trainer: 50 concurrent models

3. **Reliability Measures**
   - Service redundancy
   - Automatic failover
   - Data persistence
   - Message queue backpressure handling

## Troubleshooting Commands

If issues arise during the demo:

1. Reset all services:
```bash
docker-compose down
docker-compose up -d
```

2. Check logs for specific service:
```bash
docker-compose logs [service-name]
```

3. Verify Kafka topics:
```bash
docker-compose exec kafka kafka-topics.sh --list --bootstrap-server localhost:9092
```

4. Monitor resource usage:
```bash
docker stats
```

## Key Points to Emphasize

1. Real-time data processing capabilities
2. Model accuracy and performance
3. System resilience and error handling
4. Scalability of the architecture
5. Interactive visualization features

Remember to:
- Speak clearly and maintain eye contact
- Encourage questions throughout the demo
- Have backup data ready in case of API issues
- Monitor system resources during the presentation 