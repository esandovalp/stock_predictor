# Stock Price Prediction System - Architecture Report

## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Data Structure and Flow](#2-data-structure-and-flow)
3. [Data Capture Methods](#3-data-capture-methods)
4. [Statistical Analysis](#4-statistical-analysis)
5. [Fault Tolerance](#5-fault-tolerance)
6. [Execution Environment Comparison](#6-execution-environment-comparison)
7. [Conclusions](#7-conclusions)

## 1. System Architecture Overview

The stock prediction system is built on a distributed, microservices-based architecture with the following key components:

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

### Core Components:
1. **Data Collector**: Fetches real-time stock data from Finnhub API
2. **Kafka Message Broker**: Handles data streaming between components
3. **Model Trainer**: Processes data and maintains ML models
4. **Streamlit Dashboard**: Visualizes data and predictions
5. **ZooKeeper**: Manages the Kafka cluster configuration

## 2. Data Structure and Flow

### Data Storage Structure
```
/data
├── raw/                  # Raw stock data from API
├── processed/            # Processed data for model training
├── models/              # Trained model files
└── logs/                # Application logs
```

### Data Flow Process:
1. Data Collector fetches stock data from Finnhub API
2. Raw data is simultaneously:
   - Published to Kafka topics
   - Saved to CSV files for persistence
3. Model Trainer consumes data from Kafka
4. Processed data and models are stored in respective directories
5. Dashboard reads from both raw and processed data

## 3. Data Capture Methods

The system implements two types of data capture:

### 1. Real-time API Streaming
- Direct connection to Finnhub API
- Configurable intervals (default: 60 seconds)
- Captures current price, high, low, open, and previous close
- Implements rate limiting and error handling
- Uses exponential backoff for API failures

### 2. Batch Processing
- Periodic collection of historical data
- CSV file storage for persistence
- Kafka topics for real-time processing
- Data validation and cleaning
- Automatic data partitioning by stock symbol

## 4. Statistical Analysis

The system calculates and monitors several statistical metrics:

### Price Statistics
- Rolling mean (5-minute window)
- Rolling standard deviation
- Volume-weighted average price
- Price momentum indicators

### Model Metrics
- Mean Squared Error (MSE)
- R-squared (R²)
- Directional Accuracy
- Prediction Confidence Intervals

## 5. Fault Tolerance

### Slave Node Failure Handling
1. **Detection**
   - Health checks every 30 seconds
   - ZooKeeper monitors node status
   - Automatic failure detection

2. **Recovery Process**
   - Automatic service restart (unless-stopped policy)
   - Data replication across nodes
   - Message queue persistence
   - Automatic consumer group rebalancing

3. **Data Consistency**
   - Transaction logs in Kafka
   - Data checkpointing
   - CSV backup storage
   - Automatic data recovery

## 6. Execution Environment Comparison

### Local Environment
- **Advantages**:
  - Quick development iterations
  - Easy debugging
  - Minimal resource requirements
  - Direct file system access
- **Limitations**:
  - Limited processing power
  - No high availability
  - Single point of failure

### Cluster Environment
- **Advantages**:
  - Horizontal scalability
  - High availability
  - Load balancing
  - Fault tolerance
- **Limitations**:
  - Complex setup
  - Higher operational overhead
  - Network latency
  - Resource coordination needed

### Databricks Environment
- **Advantages**:
  - Managed infrastructure
  - Built-in monitoring
  - Automatic scaling
  - Notebook integration
  - Optimized Spark performance
- **Limitations**:
  - Platform dependency
  - Higher cost
  - Less control over infrastructure
  - Additional configuration needed

## 7. Conclusions

### Architecture Strengths
1. **Scalability**: The microservices architecture allows independent scaling of components
2. **Reliability**: Multiple fault tolerance mechanisms ensure system stability
3. **Flexibility**: Support for different execution environments
4. **Real-time Processing**: Efficient data streaming and processing pipeline

### Key Differences in Environments
1. **Performance**:
   - Local: Suitable for development and testing
   - Cluster: Optimal for production workloads
   - Databricks: Best for large-scale data processing

2. **Maintenance**:
   - Local: Minimal maintenance required
   - Cluster: Requires dedicated DevOps
   - Databricks: Managed service reduces operational overhead

3. **Cost Efficiency**:
   - Local: Most cost-effective for development
   - Cluster: Balance of cost and performance
   - Databricks: Higher cost but reduced management overhead

### Recommendations
1. Use local environment for development and testing
2. Deploy to cluster for production workloads
3. Consider Databricks for large-scale deployments or when managed service is preferred
4. Implement proper monitoring and alerting in all environments
5. Maintain data backup and recovery procedures 