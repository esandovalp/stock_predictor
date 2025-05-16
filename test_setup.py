#!/usr/bin/env python3
"""
Test Setup Script for Stock Predictor

This script verifies the installation and setup of the Stock Predictor application.
It checks all major components and provides feedback on issues found.

Usage:
    python test_setup.py [--all] [--env] [--api] [--kafka] [--docker] [--data] [--model] [--dashboard]

Options:
    --all         Run all tests (default)
    --env         Test environment variables
    --api         Test Finnhub API connection
    --kafka       Test Kafka connectivity
    --docker      Test Docker container status
    --data        Test data persistence
    --model       Test model training
    --dashboard   Test dashboard accessibility
"""

import os
import sys
import time
import json
import socket
import argparse
import subprocess
import requests
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
import docker
import dotenv

# Constants
SUCCESS = "\033[92m✓\033[0m"  # Green checkmark
WARNING = "\033[93m⚠\033[0m"  # Yellow warning
ERROR = "\033[91m✗\033[0m"    # Red x
INFO = "\033[94mℹ\033[0m"     # Blue info

# Load environment variables
dotenv.load_dotenv()

class TestSetup:
    def __init__(self):
        self.results = {
            "env": {"status": "not_run", "issues": []},
            "api": {"status": "not_run", "issues": []},
            "kafka": {"status": "not_run", "issues": []},
            "docker": {"status": "not_run", "issues": []},
            "data": {"status": "not_run", "issues": []},
            "model": {"status": "not_run", "issues": []},
            "dashboard": {"status": "not_run", "issues": []}
        }
        self.docker_client = None
        try:
            self.docker_client = docker.from_env()
        except Exception as e:
            print(f"{ERROR} Could not connect to Docker: {e}")

    def test_environment_variables(self):
        """Test if all required environment variables are set"""
        print("\n🔍 Testing environment variables...")
        self.results["env"]["status"] = "running"
        
        # Required variables
        required_vars = [
            "FINNHUB_API_KEY"
        ]
        
        # Optional variables with defaults
        optional_vars = [
            "STOCK_SYMBOL", 
            "PREDICTION_INTERVAL", 
            "KAFKA_TOPIC", 
            "KAFKA_BROKER",
            "ENV", 
            "LOG_LEVEL"
        ]
        
        # Check required variables
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
                self.results["env"]["issues"].append(f"Missing required environment variable: {var}")
        
        if missing_vars:
            print(f"{ERROR} Missing required environment variables: {', '.join(missing_vars)}")
            print(f"{INFO} Create a .env file based on .env.example with your own values")
            self.results["env"]["status"] = "failed"
        else:
            print(f"{SUCCESS} All required environment variables are set")
            
            # Check optional variables
            for var in optional_vars:
                if not os.getenv(var):
                    print(f"{WARNING} Optional environment variable not set: {var}")
                    self.results["env"]["issues"].append(f"Optional environment variable not set: {var}")
            
            self.results["env"]["status"] = "passed"
        return self.results["env"]["status"] == "passed"

    def test_finnhub_api(self):
        """Test connection to Finnhub API"""
        print("\n🔍 Testing Finnhub API connection...")
        self.results["api"]["status"] = "running"
        
        api_key = os.getenv("FINNHUB_API_KEY")
        if not api_key:
            print(f"{ERROR} No Finnhub API key found in environment variables")
            self.results["api"]["issues"].append("No Finnhub API key found")
            self.results["api"]["status"] = "failed"
            return False
        
        stock_symbol = os.getenv("STOCK_SYMBOL", "AAPL")
        
        try:
            url = f"https://finnhub.io/api/v1/quote?symbol={stock_symbol}&token={api_key}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'c' in data:
                    print(f"{SUCCESS} Successfully connected to Finnhub API")
                    print(f"{INFO} Current price of {stock_symbol}: ${data['c']}")
                    self.results["api"]["status"] = "passed"
                    return True
                else:
                    print(f"{ERROR} Invalid response from Finnhub API")
                    self.results["api"]["issues"].append("Invalid API response format")
            elif response.status_code == 401:
                print(f"{ERROR} Authentication failed. Check your API key")
                self.results["api"]["issues"].append("API authentication failed")
            elif response.status_code == 429:
                print(f"{ERROR} Rate limit exceeded for Finnhub API")
                self.results["api"]["issues"].append("API rate limit exceeded")
            else:
                print(f"{ERROR} API request failed with status code: {response.status_code}")
                self.results["api"]["issues"].append(f"API request failed with status code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"{ERROR} Failed to connect to Finnhub API: {e}")
            self.results["api"]["issues"].append(f"Connection error: {str(e)}")
        
        self.results["api"]["status"] = "failed"
        return False

    def test_kafka_connectivity(self):
        """Test connection to Kafka"""
        print("\n🔍 Testing Kafka connectivity...")
        self.results["kafka"]["status"] = "running"
        
        kafka_broker = os.getenv("KAFKA_BROKER", "kafka:29092")
        
        # Parse the broker address
        if ":" in kafka_broker:
            host, port = kafka_broker.split(":")
            port = int(port)
        else:
            host = kafka_broker
            port = 9092
        
        # If we're using Docker, adjust host for local testing
        if host == "kafka" and not self._is_running_in_docker():
            print(f"{INFO} Adjusting Kafka host for local testing")
            host = "localhost"
        
        # Try to connect to Kafka
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"{SUCCESS} Successfully connected to Kafka at {host}:{port}")
                self.results["kafka"]["status"] = "passed"
                return True
            else:
                print(f"{ERROR} Could not connect to Kafka at {host}:{port}")
                self.results["kafka"]["issues"].append(f"Could not connect to Kafka at {host}:{port}")
        except Exception as e:
            print(f"{ERROR} Error connecting to Kafka: {e}")
            self.results["kafka"]["issues"].append(f"Connection error: {str(e)}")
        
        # If Docker is available, check if Kafka container is running
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                kafka_containers = [c for c in containers if "kafka" in c.name.lower()]
                if kafka_containers:
                    container = kafka_containers[0]
                    print(f"{INFO} Found Kafka container: {container.name} ({container.status})")
                    if container.status != "running":
                        print(f"{WARNING} Kafka container is not running")
                        self.results["kafka"]["issues"].append("Kafka container is not running")
                        print(f"{INFO} Try starting it with: docker-compose up -d kafka")
                else:
                    print(f"{WARNING} No Kafka container found")
                    self.results["kafka"]["issues"].append("No Kafka container found")
                    print(f"{INFO} Try starting it with: docker-compose up -d kafka")
            except Exception as e:
                print(f"{ERROR} Error checking Kafka container: {e}")
                self.results["kafka"]["issues"].append(f"Docker error: {str(e)}")
        
        self.results["kafka"]["status"] = "failed"
        return False

    def test_docker_containers(self):
        """Test Docker container status"""
        print("\n🔍 Testing Docker container status...")
        self.results["docker"]["status"] = "running"
        
        if not self.docker_client:
            print(f"{ERROR} Docker is not available")
            self.results["docker"]["issues"].append("Docker is not available")
            self.results["docker"]["status"] = "failed"
            return False
        
        expected_containers = [
            "zookeeper",
            "kafka",
            "data-collector",
            "model-trainer",
            "dashboard"
        ]
        
        running_containers = []
        non_running_containers = []
        missing_containers = []
        
        try:
            containers = self.docker_client.containers.list(all=True)
            container_names = [c.name for c in containers]
            
            for expected in expected_containers:
                matching = [c for c in containers if expected in c.name]
                if matching:
                    if matching[0].status == "running":
                        running_containers.append(matching[0].name)
                    else:
                        non_running_containers.append((matching[0].name, matching[0].status))
                else:
                    missing_containers.append(expected)
            
            if running_containers:
                print(f"{SUCCESS} Running containers: {', '.join(running_containers)}")
            
            if non_running_containers:
                print(f"{WARNING} Non-running containers: {', '.join([f'{name} ({status})' for name, status in non_running_containers])}")
                for name, status in non_running_containers:
                    self.results["docker"]["issues"].append(f"Container {name} is {status}")
                print(f"{INFO} Start all containers with: docker-compose up -d")
            
            if missing_containers:
                print(f"{ERROR} Missing containers: {', '.join(missing_containers)}")
                for name in missing_containers:
                    self.results["docker"]["issues"].append(f"Container {name} is missing")
                print(f"{INFO} Create all containers with: docker-compose up -d")
            
            if non_running_containers or missing_containers:
                self.results["docker"]["status"] = "failed"
                return False
            
            self.results["docker"]["status"] = "passed"
            return True
            
        except Exception as e:
            print(f"{ERROR} Error checking Docker containers: {e}")
            self.results["docker"]["issues"].append(f"Docker error: {str(e)}")
            self.results["docker"]["status"] = "failed"
            return False

    def test_data_persistence(self):
        """Test data persistence"""
        print("\n🔍 Testing data persistence...")
        self.results["data"]["status"] = "running"
        
        # Check if data directory exists
        data_dir = Path("data")
        if not data_dir.exists():
            print(f"{WARNING} Data directory not found at {data_dir.absolute()}")
            print(f"{INFO} This may be normal if you're using Docker volumes")
            
            # Try to check inside Docker container
            if self.docker_client:
                try:
                    containers = self.docker_client.containers.list()
                    data_container = None
                    for c in containers:
                        if "data-collector" in c.name:
                            data_container = c
                            break
                    
                    if data_container:
                        print(f"{INFO} Checking data directory in container {data_container.name}")
                        exit_code, output = data_container.exec_run("ls -la /app/data")
                        if exit_code == 0:
                            print(f"{SUCCESS} Data directory found in container")
                            print(output.decode())
                            self.results["data"]["status"] = "passed"
                            return True
                        else:
                            print(f"{ERROR} Error checking data directory in container: {output.decode()}")
                            self.results["data"]["issues"].append("Data directory not accessible in container")
                except Exception as e:
                    print(f"{ERROR} Error checking data in Docker container: {e}")
                    self.results["data"]["issues"].append(f"Docker error: {str(e)}")
        else:
            # Check for data files
            raw_dir = data_dir / "raw"
            if not raw_dir.exists():
                print(f"{WARNING} Raw data directory not found")
                self.results["data"]["issues"].append("Raw data directory not found")
            else:
                csv_files = list(raw_dir.glob("*.csv"))
                if csv_files:
                    print(f"{SUCCESS} Found {len(csv_files)} CSV files in raw data directory")
                    for csv_file in csv_files:
                        print(f"{INFO} {csv_file.name}: {csv_file.stat().st_size} bytes")
                    self.results["data"]["status"] = "passed"
                    return True
                else:
                    print(f"{WARNING} No CSV files found in raw data directory")
                    self.results["data"]["issues"].append("No CSV files found")
                    print(f"{INFO} This may be normal if data collection hasn't started yet")
        
        self.results["data"]["status"] = "warning"
        return True

    def test_model_training(self):
        """Test model training capabilities"""
        print("\n🔍 Testing model training capabilities...")
        self.results["model"]["status"] = "running"
        
        # Check if model file exists
        model_dir = Path("data/models") if Path("data/models").exists() else Path("data")
        model_files = list(model_dir.glob("*_price_model.joblib"))
        
        if model_files:
            print(f"{SUCCESS} Found model file: {model_files[0].name}")
            print(f"{INFO} Last modified: {datetime.fromtimestamp(model_files[0].stat().st_mtime)}")
            
            # Check if metrics file exists
            metrics_file = Path("data/processed/model_metrics.csv")
            if metrics_file.exists():
                print(f"{SUCCESS} Found model metrics file")
                self.results["model"]["status"] = "passed"
                return True
            else:
                print(f"{WARNING} Model metrics file not found")
                self.results["model"]["issues"].append("Model metrics file not found")
        else:
            print(f"{WARNING} No model file found")
            self.results["model"]["issues"].append("No model file found")
            print(f"{INFO} This may be normal if model training hasn't started yet")
            
            # Try to check inside Docker container
            if self.docker_client:
                try:
                    containers = self.docker_client.containers.list()
                    model_container = None
                    for c in containers:
                        if "model-trainer" in c.name:
                            model_container = c
                            break
                    
                    if model_container:
                        print(f"{INFO} Found model-trainer container: {model_container.name} ({model_container.status})")
                        if model_container.status == "running":
                            print(f"{INFO} Model trainer is running, check logs with: docker logs {model_container.name}")
                        else:
                            print(f"{WARNING} Model trainer container is not running")
                            self.results["model"]["issues"].append("Model trainer container is not running")
                except Exception as e:
                    print(f"{ERROR} Error checking model container: {e}")
                    self.results["model"]["issues"].append(f"Docker error: {str(e)}")
        
        self.results["model"]["status"] = "warning"
        return True

    def test_dashboard_accessibility(self):
        """Test dashboard accessibility"""
        print("\n🔍 Testing dashboard accessibility...")
        self.results["dashboard"]["status"] = "running"
        
        dashboard_port = int(os.getenv("DASHBOARD_PORT", "8501"))
        host = "localhost"  # Assume dashboard is running locally or port-forwarded
        
        # Try to connect to the dashboard
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, dashboard_port))
            sock.close()
            
            if result == 0:
                print(f"{SUCCESS} Dashboard is accessible at http://{host}:{dashboard_port}")
                
                # Try to fetch the dashboard content
                try:
                    response = requests.get(f"http://{host}:{dashboard_port}", timeout=10)
                    if response.status_code == 200:
                        print(f"{SUCCESS} Dashboard responded with status code 200")
                        self.results["dashboard"]["status"] = "passed"
                        return True
                    else:
                        print(f"{WARNING} Dashboard responded with status code {response.status_code}")
                        self.results["dashboard"]["issues"].append(f"Dashboard responded with status code {response.status_code}")
                except requests.exceptions.RequestException as e:
                    print(f"{WARNING} Error fetching dashboard content: {e}")
                    self.results["dashboard"]["issues"].append(f"Request error: {str(e)}")
            else:
                print(f"{ERROR} Dashboard is not accessible at http://{host}:{dashboard_port}")
                self.results["dashboard"]["issues"].append(f"Cannot connect to {host}:{dashboard_port}")
        except Exception as e:
            print(f"{ERROR} Error checking dashboard accessibility: {e}")
            self.results["dashboard"]["issues"].append(f"Connection error: {str(e)}")
        
        # If Docker is available, check if dashboard container is running
        if self.docker_client:
            try:
                containers = self.docker_client.containers.list()
                dashboard_containers = [c for c in containers if "dashboard" in c.name.lower()]
                if dashboard_containers:
                    container = dashboard_containers[0]
                    print(f"{INFO} Found dashboard container: {container.name} ({container.status})")
                    if container.status != "running":
                        print(f"{WARNING} Dashboard container is not running")
                        self.results["dashboard"]["issues"].append("Dashboard container is not running")
                        print(f"{INFO} Try starting it with: docker-compose up -d dashboard")
                    else:
                        # Check logs for potential issues
                        logs = container.logs(tail=20).decode()
                        if "Error" in logs or "Exception" in logs:
                            print(f"{WARNING} Found potential errors in dashboard logs")
                            self.results["dashboard"]["issues"].append("Found potential errors in dashboard logs")
                            print(f"{INFO} Check logs with: docker logs {container.name}")
                else:
                    print(f"{WARNING} No dashboard container found")
                    self.results["dashboard"]["issues"].append("No dashboard container found")
                    print(f"{INFO} Try starting it with: docker-compose up -d dashboard")
            except Exception as e:
                print(f"{ERROR} Error checking dashboard container: {e}")
                self.results["dashboard"]["issues"].append(f"Docker error: {str(e)}")
        
        self.results["dashboard"]["status"] = "failed"
        return False

    def _is_running_in_docker(self):
        """Check if we're running inside a Docker container"""
        return os.path.exists('/.dockerenv')

    def print_summary(self):
        """Print a summary of all test results"""
        print("\n" + "="*80)
        print("SUMMARY OF TEST RESULTS")
        print("="*80)
        
        status_icons = {
            "passed": f"{SUCCESS} PASSED",
            "warning": f"{WARNING} WARNING",
            "failed": f"{ERROR} FAILED",
            "not_run": "  NOT RUN"
        }
        
        for test_name, result in self.results.items():
            status = result["status"]
            icon = status_icons.get(status, "  UNKNOWN")
            print(f"{icon} {test_name.upper()}")
            
            if status in ["warning", "failed"] and result["issues"]:
                print("  Issues:")
                for issue in result["issues"]:
                    print(f"    - {issue}")
                
                # Print suggestions for common issues
                if test_name == "env":
                    print(f"  {INFO} Suggestion: Create a .env file based on .env.example with your own values")
                elif test_name == "api":
                    print(f"  {INFO} Suggestion: Verify your Finnhub API key is correct")
                elif test_name == "kafka":
                    print(f"  {INFO} Suggestion: Ensure Kafka is running with 'docker-compose up -d kafka'")
                elif test_name == "docker":
                    print(f"  {INFO} Suggestion: Start all services with 'docker-compose up -d'")
                elif test_name == "dashboard":
                    print(f"  {INFO} Suggestion: Check dashboard logs with 'docker logs dashboard'")
        
        print("\n" + "="*80)
        all_passed = all(r["status"] == "passed" for r in self.results.values())
        all_run = all(r["status"] != "not_run" for r in self.results.values())
        
        if all_passed:
            print(f"{SUCCESS} All tests passed! Your Stock Predictor setup is working correctly.")
        elif all_run:
            print(f"{WARNING} Some tests did not pass. Check the issues and suggestions above.")
        else:
            print(f"{INFO} Some tests were not run. Use --all to run all tests.")
        
        print("="*80 + "\n")
        
        return all_passed


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Test the Stock Predictor setup")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--env", action="store_true", help="Test environment variables")
    parser.add_argument("--api", action="store_true", help="Test Finnhub API connection")
    parser.add_argument("--kafka", action="store_true", help="Test Kafka connectivity")
    parser.add_argument("--docker", action="store_true", help="Test Docker container status")
    parser.add_argument("--data", action="store_true", help="Test data persistence")
    parser.add_argument("--model", action="store_true", help="Test model training")
    parser.add_argument("--dashboard", action="store_true", help="Test dashboard accessibility")
    
    args = parser.parse_args()
    
    # If no specific tests are requested, run all tests
    run_all = args.all or not (args.env or args.api or args.kafka or args.docker or args.data or args.model or args.dashboard)
    
    tester = TestSetup()
    
    print("Stock Predictor Setup Test")
    print("-" * 80)
    print(f"Running tests at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 80)
    
    # Run tests
    if run_all or args.env:
        tester.test_environment_variables()
    
    if run_all or args.api:
        tester.test_finnhub_api()
    
    if run_all or args.kafka:
        tester.test_kafka_connectivity()
    
    if run_all or args.docker:
        tester.test_docker_containers()
    
    if run_all or args.data:
        tester.test_data_persistence()
    
    if run_all or args.model:
        tester.test_model_training()
    
    if run_all or args.dashboard:
        tester.test_dashboard_accessibility()
    
    # Print summary
    success = tester.print_summary()
    
    # Return exit code based on test results
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
