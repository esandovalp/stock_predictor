# Stock Price Prediction System - Testing Checklist

This document provides a structured approach to testing the Stock Price Prediction system and preparing for the presentation. Follow these steps to ensure all components work correctly and the demonstration goes smoothly.

## 1. Pre-testing Requirements

### Environment Setup Verification

- [ ] Docker and Docker Compose are installed and running:
  ```bash
  docker --version
  docker-compose --version
  ```

- [ ] Python 3.9 or higher is installed:
  ```bash
  python --version
  ```

- [ ] Sufficient disk space is available (at least 2GB free):
  ```bash
  docker system df
  ```

- [ ] Required ports are available (2181, 9092, 8501):
  ```bash
  lsof -i :2181
  lsof -i :9092
  lsof -i :8501
  ```

- [ ] Network connectivity is available (for API access):
  ```bash
  ping -c 3 finnhub.io
  ```

### Data Directory Preparation

- [ ] Create necessary data directories:
  ```bash
  mkdir -p data/raw data/processed data/models
  ```

- [ ] Set correct permissions:
  ```bash
  chmod -R 755 data
  ```

- [ ] Clean old test data (if exists):
  ```bash
  rm -f data/raw/*.csv data/processed/*.csv
  ```

- [ ] Verify data directory structure:
  ```bash
  find data -type d | sort
  ```

### API Key Validation Steps

- [ ] Confirm Finnhub API key is set in the .env file:
  ```bash
  grep FINNHUB_API_KEY .env
  ```

- [ ] Verify API key is valid with a simple test:
  ```bash
  ./scripts/test_api_key.sh
  ```

- [ ] Check API rate limits and quotas in Finnhub dashboard

- [ ] Ensure the API key is not expired or revoked

## 2. Testing Procedure

### Component Startup Sequence

- [ ] Start the Zookeeper and Kafka broker first:
  ```bash
  docker-compose up -d zookeeper kafka
  ```

- [ ] Wait for Kafka to be ready (usually 30-60 seconds)
  ```bash
  docker-compose logs --tail=20 kafka | grep "started"
  ```

- [ ] Start the data collector service:
  ```bash
  docker-compose up -d data-collector
  ```

- [ ] Start the model trainer service:
  ```bash
  docker-compose up -d model-trainer
  ```

- [ ] Start the dashboard service last:
  ```bash
  docker-compose up -d dashboard
  ```

- [ ] Verify all services are running:
  ```bash
  docker-compose ps
  ```

### Verification Steps for Each Component

- [ ] **Zookeeper & Kafka:**
  - Verify Zookeeper is running:
    ```bash
    docker-compose logs zookeeper | grep "binding to port"
    ```
  - Verify Kafka is connected to Zookeeper:
    ```bash
    docker-compose logs kafka | grep "connected to zookeeper"
    ```
  - Verify Kafka topics are created:
    ```bash
    docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092
    ```

- [ ] **Data Collector:**
  - Check logs for successful API connections:
    ```bash
    docker-compose logs data-collector | grep "Data collected successfully"
    ```
  - Verify no API errors or rate limit issues:
    ```bash
    docker-compose logs data-collector | grep -i "error"
    ```
  - Check Kafka producer is working:
    ```bash
    docker-compose logs data-collector | grep "Message delivered"
    ```

- [ ] **Model Trainer:**
  - Verify it's waiting for data:
    ```bash
    docker-compose logs model-trainer | grep "Waiting for sufficient data"
    ```
  - Check Kafka consumer is working:
    ```bash
    docker-compose logs model-trainer | grep "Consumed message"
    ```

- [ ] **Dashboard:**
  - Verify Streamlit is running:
    ```bash
    docker-compose logs dashboard | grep "Streamlit app started"
    ```
  - Check if it can access the data:
    ```bash
    docker-compose logs dashboard | grep "Loading data"
    ```
  - Open dashboard in browser: http://localhost:8501

### Data Collection Verification

- [ ] Verify data is being collected:
  ```bash
  docker-compose exec data-collector ls -la /app/data/raw/
  ```

- [ ] Check CSV file format is correct:
  ```bash
  docker-compose exec data-collector head -n 3 /app/data/raw/AAPL_stock_data.csv
  ```

- [ ] Confirm data collection frequency:
  ```bash
  docker-compose exec data-collector cat /app/logs/data_collector.log | grep "Data collected" | wc -l
  ```

- [ ] Monitor data points count:
  ```bash
  docker-compose exec data-collector wc -l /app/data/raw/AAPL_stock_data.csv
  ```

### Model Training Validation

- [ ] Wait for 30+ data points to be collected:
  ```bash
  docker-compose exec data-collector wc -l /app/data/raw/AAPL_stock_data.csv
  ```

- [ ] Verify training starts automatically:
  ```bash
  docker-compose logs --follow model-trainer | grep "Training model"
  ```

- [ ] Check model metrics CSV creation:
  ```bash
  docker-compose exec model-trainer ls -la /app/data/processed/model_metrics.csv
  ```

- [ ] Verify model file is saved:
  ```bash
  docker-compose exec model-trainer ls -la /app/data/models/
  ```

- [ ] Check model performance metrics:
  ```bash
  docker-compose exec model-trainer tail -n 1 /app/data/processed/model_metrics.csv
  ```

## 3. Pre-presentation Setup

### System Requirements

- [ ] Free up system resources (close other applications)

- [ ] Restart Docker service for clean environment:
  ```bash
  docker system prune -f
  docker-compose down -v
  ```

- [ ] Set system to prevent sleep during presentation:
  ```bash
  # On macOS
  caffeinate -d &
  # On Linux
  systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
  ```

- [ ] Turn off notifications and updates

- [ ] Connect to power source (not battery)

- [ ] Use wired internet connection if possible

### Data Preparation

- [ ] Option 1: Start fresh for presentation:
  ```bash
  rm -rf data/*
  mkdir -p data/raw data/processed data/models
  ```

- [ ] Option 2: Pre-populate with sample data:
  ```bash
  cp sample_data/* data/raw/
  ```

- [ ] Consider creating a pre-trained model for immediate demonstration:
  ```bash
  ./scripts/train_initial_model.sh
  ```

- [ ] Prepare at least 30-60 minutes of pre-collected data:
  ```bash
  ./scripts/simulate_data_collection.sh
  ```

### Timing Considerations

- [ ] Start services at least 30-45 minutes before presentation:
  ```bash
  ./setup.sh start
  ```

- [ ] Allow time for data collection (minimum 30 points needed):
  - Collection rate: 1 point per minute
  - Required minimum: 30 minutes
  - Safe buffer: 45 minutes

- [ ] Schedule time for model training (typically 1-2 minutes)

- [ ] Practice the presentation flow at least once with real data

### Backup Procedures

- [ ] Create a backup of configuration files:
  ```bash
  mkdir -p backup
  cp docker-compose.yml .env requirements.txt Dockerfile backup/
  ```

- [ ] Prepare a backup dataset:
  ```bash
  ./scripts/backup_data.sh
  ```

- [ ] Save a screenshot of a working dashboard:
  ```bash
  # Take screenshot manually or use:
  ./scripts/capture_dashboard.sh
  ```

- [ ] Have a backup API key ready

- [ ] Prepare offline demo capability:
  ```bash
  ./scripts/prepare_offline_demo.sh
  ```

- [ ] Create a backup Docker image:
  ```bash
  docker-compose build
  docker save -o backup/stock_predictor_images.tar stock_predictor-data-collector stock_predictor-model-trainer stock_predictor-dashboard
  ```

## Final Checks Before Presentation

- [ ] All services are running and healthy:
  ```bash
  docker-compose ps
  ```

- [ ] Dashboard is accessible: http://localhost:8501

- [ ] Data is being collected and displayed correctly

- [ ] Model has been trained at least once

- [ ] All demonstration steps from PRESENTATION.md have been tested

- [ ] Computer is connected to power and internet

- [ ] Presentation materials are ready and accessible

---

Good luck with your presentation! This system demonstrates a complete end-to-end data pipeline with real-time prediction capabilities - a comprehensive showcase of software engineering, data science, and ML deployment skills.

