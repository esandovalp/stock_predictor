# Stock Price Predictor - Development Guide

This document provides information for developers who want to contribute to the Stock Price Predictor project.

## Table of Contents

1. [Development Setup](#development-setup)
2. [Project Structure](#project-structure)
3. [Contributing Guidelines](#contributing-guidelines)
4. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

## Development Setup

### Prerequisites

- **Git**: For version control
- **Docker**: Version 20.10.0 or higher
- **Docker Compose**: Version 2.0.0 or higher
- **Python**: Version 3.9 or higher
- **Make**: (Optional) For using the Makefile shortcuts

#### Key Python Packages

- Kafka-Python
- PySpark
- Streamlit
- Scikit-learn
- Pandas/Numpy
- Plotly
- Requests

### Local Development Environment Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/stock_predictor.git
   cd stock_predictor
   ```

2. **Setup the project**:
   There are two ways to set up the project:

   **Using Make**:
   ```bash
   make setup
   ```

   **Using the setup script**:
   ```bash
   ./setup.sh setup
   ```

   This will:
   - Check your system for required dependencies
   - Create necessary directories
   - Set up the .env file
   - Build Docker images

3. **Using a Python virtual environment** (for local development without Docker):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Environment configuration**:
   - Copy `.env.example` to `.env` if not already done
   - Obtain a Finnhub API key from [finnhub.io](https://finnhub.io/register)
   - Configure settings in the `.env` file

### IDE Configuration

#### Visual Studio Code

1. **Recommended Extensions**:
   - Python (Microsoft)
   - Docker (Microsoft)
   - Remote - Containers (Microsoft)
   - Pylance (Microsoft)
   - autoDocstring (Nils Werner)
   - Git Graph (mhutchie)

2. **Settings configuration** (`.vscode/settings.json`):
   ```json
   {
     "python.linting.enabled": true,
     "python.linting.flake8Enabled": true,
     "python.formatting.provider": "black",
     "editor.formatOnSave": true,
     "python.testing.pytestEnabled": true,
     "python.testing.nosetestsEnabled": false,
     "python.testing.unittestEnabled": false,
     "python.testing.pytestArgs": [
       "tests"
     ]
   }
   ```

#### PyCharm

1. **Recommended Plugins**:
   - Docker integration
   - Streamlit
   - Save Actions

2. **Setup Project Interpreter**:
   - Settings → Project → Python Interpreter
   - Add new interpreter → Choose Docker Compose
   - Select docker-compose.yml and service 'stock-app'

### Running Tests

Testing can be performed in several ways:

**Using Make**:
```bash
make test                        # Run all tests
make test TEST_ARGS="--api"      # Run specific tests
```

**Using the setup script**:
```bash
./setup.sh test                  # Run all tests
./setup.sh test --api --kafka    # Run specific API and Kafka tests
```

**Directly using Python**:
```bash
python -m pytest tests/          # Run all tests
python -m pytest tests/test_api.py  # Run specific test file
```

## Project Structure

### Directory Layout

```
stock_predictor/
├── app/                    # Main application code
│   ├── __init__.py         # Package initialization
│   ├── config.py           # Configuration loading
│   ├── data_collector.py   # Finnhub API data collection
│   ├── ml_model.py         # Machine learning models
│   └── dashboard.py        # Streamlit dashboard
├── data/                   # Data storage (created at runtime)
│   ├── raw/                # Raw data from API
│   ├── processed/          # Processed datasets
│   ├── models/             # Trained model files
│   └── logs/               # Application logs
├── tests/                  # Test files
├── .env.example            # Example environment variables
├── .gitignore              # Git ignore file
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── Makefile                # Make targets
├── requirements.txt        # Python dependencies
├── setup.sh                # Setup and management script
├── test_setup.py           # Setup testing script
├── README.md               # Project documentation
└── DEVELOPMENT.md          # This file
```

### Component Descriptions

1. **Data Collector**: Connects to Finnhub API to fetch real-time stock data and streams it to Kafka. It also stores the data to CSV for persistence.

2. **Model Trainer**: Consumes data from Kafka, processes it, and trains a linear regression model to predict stock prices.

3. **Streamlit Dashboard**: Visualizes the real-time data, displays model predictions, and provides trading recommendations.

4. **Kafka**: Message broker that enables asynchronous communication between components.

5. **ZooKeeper**: Required for Kafka, manages the Kafka cluster.

### Data Flow

```
Finnhub API → Data Collector → Kafka → Model Trainer
                    ↓                       ↓
                CSV Storage              Model Storage
                    ↓                       ↓
                    └────→ Dashboard ←──────┘
                              ↓
                        User Interface
```

1. The Data Collector fetches stock data from Finnhub API at regular intervals.
2. Data is published to a Kafka topic and saved to CSV files.
3. The Model Trainer consumes data from Kafka, trains/updates the model, and saves it.
4. The Dashboard reads from CSV files and the trained model to display visualizations.
5. Users interact with the dashboard to view data, predictions, and recommendations.

### Key Files

- **config.py**: Central configuration file that loads environment variables.
- **data_collector.py**: Implements the FinnhubDataCollector class for API integration.
- **ml_model.py**: Implements the StockPricePredictor class for machine learning.
- **dashboard.py**: Implements the Streamlit dashboard interface.
- **docker-compose.yml**: Defines all services and their relationships.
- **setup.sh**: Management script for development operations.

## Contributing Guidelines

### Code Style

This project follows [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines for Python code.

- **Formatting**: We use Black with a line length of 88 characters.
- **Linting**: We use Flake8 for code linting.
- **Docstrings**: Follow Google-style docstrings format.
- **Import Order**: Standard library, third-party, local application.

Example of proper code style:

```python
"""Module docstring explaining the purpose of the module."""

import os
import logging
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from app.config import STOCK_SYMBOL


def function_name(param1, param2):
    """Summary of function purpose.
    
    Args:
        param1 (type): Description of param1.
        param2 (type): Description of param2.
        
    Returns:
        return_type: Description of return value.
        
    Raises:
        ExceptionType: When and why this exception is raised.
    """
    # Function implementation
    return result
```

### Git Workflow

We follow a simplified Git Flow workflow:

1. **main**: Production-ready code
2. **develop**: Latest development code
3. **feature/xxx**: Feature branches

#### Branching Strategy:

1. Create a new feature branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes, commit them with clear messages:
   ```bash
   git add .
   git commit -m "Add feature X" -m "Detailed description of the changes"
   ```

3. Push your branch and create a pull request to `develop`:
   ```bash
   git push origin feature/your-feature-name
   ```

### Pull Request Process

1. **Create your PR**: Target the `develop` branch
2. **Describe your changes**: Provide a clear description of what you've changed
3. **Link related issues**: Reference any related issues with `#issue_number`
4. **Wait for CI**: Ensure all CI checks pass
5. **Request review**: Assign at least one reviewer
6. **Address feedback**: Make requested changes and push again
7. **Merge**: Once approved, the PR can be merged

### Testing Requirements

All new code should include appropriate tests:

1. **Unit Tests**: For individual functions/methods
2. **Integration Tests**: For component interactions
3. **Functional Tests**: For end-to-end workflows

Test coverage should be at least 80% for new code. Run tests regularly:

```bash
make test
```

## Debugging and Troubleshooting

### Common Issues

#### 1. Docker Connection Issues

**Symptom**: Cannot connect to Docker daemon.
**Solution**: 
- Ensure Docker is running
- Verify your user is in the `docker` group
- Try restarting Docker

#### 2. Kafka Connection Problems

**Symptom**: Services can't connect to Kafka.
**Solution**:
- Check if Kafka container is running: `docker ps | grep kafka`
- Verify network settings in docker-compose.yml
- Check Kafka logs: `make logs SERVICE=kafka`

#### 3. Finnhub API Issues

**Symptom**: No data is being collected.
**Solution**:
- Verify your API key in the .env file
- Check API rate limits
- Test API connection manually: `curl -X GET "https://finnhub.io/api/v1/quote?symbol=AAPL&token=YOUR_API_KEY"`

#### 4. Model Training Problems

**Symptom**: Model fails to train.
**Solution**:
- Ensure enough data points are collected (at least 30)
- Check model-trainer logs: `make logs SERVICE=model-trainer`
- Verify data quality in CSV files

### Debugging Tools

1. **Docker Logs**: 
   ```bash
   make logs SERVICE=service-name
   # or
   docker-compose logs service-name
   ```

2. **Interactive Python Debugging**:
   ```bash
   docker-compose exec service-name python -m pdb /path/to/script.py
   ```

3. **Inspecting Data**:
   ```bash
   docker-compose exec service-name python
   >>> import pandas as pd
   >>> df = pd.read_csv('/app/data/raw/aapl_stock_data.csv')
   >>> df.head()
   ```

4. **Network Inspection**:
   ```bash
   docker network inspect stock-net
   ```

### Logging

All services use Python's logging module with configurable levels:

- **DEBUG**: Detailed information, typically for diagnosis
- **INFO**: Confirmation that things are working as expected
- **WARNING**: An indication something unexpected happened
- **ERROR**: Due to a more serious problem, software hasn't performed some function
- **CRITICAL**: A serious error indicating the program may be unable to continue

You can adjust the log level in the `.env` file:
```
LOG_LEVEL=DEBUG
```

### Performance Monitoring

1. **Docker Stats**:
   ```bash
   docker stats
   ```

2. **CPU and Memory Usage**:
   ```bash
   docker-compose exec service-name top
   ```

3. **Disk Usage**:
   ```bash
   docker system df -v
   ```

4. **Network Traffic**:
   ```bash
   docker-compose exec service-name netstat -tulpn
   ```

## Additional Resources

- [Finnhub API Documentation](https://finnhub.io/docs/api)
- [Kafka Documentation](https://kafka.apache.org/documentation/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Docker Documentation](https://docs.docker.com/)

