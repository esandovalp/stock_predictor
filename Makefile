# Stock Price Predictor Makefile
# Provides convenient shortcuts for common operations

# Variables
SHELL := /bin/bash
SETUP_SCRIPT := ./setup.sh
PYTHON := python3
PIP := pip3
DOCKER_COMPOSE := docker-compose
SERVICE_NAMES := zookeeper kafka data-collector model-trainer dashboard

# Default service for logs
DEFAULT_LOG_SERVICE := dashboard
# Default number of log lines
DEFAULT_LOG_LINES := it 100

# Help command is the default
.DEFAULT_GOAL := help

# Declare phony targets (targets that don't represent files)
.PHONY: help setup start stop restart clean \
        test logs status lint format \
        start-dashboard start-collector start-model

# Help target - Lists all available targets
help:
	@echo "Stock Price Predictor - Makefile Commands"
	@echo ""
	@echo "Basic Operations:"
	@echo "  make setup              - Run initial setup (check requirements, create directories, build images)"
	@echo "  make start              - Start all services"
	@echo "  make stop               - Stop all services"
	@echo "  make restart            - Restart all services"
	@echo "  make clean              - Clean up project (use CLEAN_MODE=data|containers|all, default: all)"
	@echo ""
	@echo "Development Helpers:"
	@echo "  make test               - Run tests (use TEST_ARGS=--specific-test for specific tests)"
	@echo "  make logs               - View logs (use SERVICE=service-name, LINES=100)"
	@echo "  make status             - Check project status"
	@echo "  make lint               - Run code linting"
	@echo "  make format             - Format code"
	@echo ""
	@echo "Individual Services:"
	@echo "  make start-dashboard    - Start dashboard service"
	@echo "  make start-collector    - Start data collector service"
	@echo "  make start-model        - Start model trainer service"
	@echo ""
	@echo "Examples:"
	@echo "  make logs SERVICE=data-collector LINES=50   - View last 50 lines of data-collector logs"
	@echo "  make clean CLEAN_MODE=data                  - Clean up data files only"
	@echo "  make test TEST_ARGS=\"--api --kafka\"        - Run specific tests"

# Basic Operations

# Setup: Run initial setup
setup:
	@$(SETUP_SCRIPT) setup

# Start: Start all services
start:
	@$(SETUP_SCRIPT) start

# Stop: Stop all services
stop:
	@$(SETUP_SCRIPT) stop

# Restart: Restart all services
restart:
	@$(SETUP_SCRIPT) restart

# Clean: Clean up project
clean:
	@$(SETUP_SCRIPT) clean $(CLEAN_MODE)

# Development Helpers

# Test: Run tests
test:
	@$(SETUP_SCRIPT) test $(TEST_ARGS)

# Logs: View logs
logs:
	@$(SETUP_SCRIPT) logs $(SERVICE) $(LINES)

# Status: Check status
status:
	@$(SETUP_SCRIPT) status

# Lint: Run linting
lint:
	@echo "Running linters..."
	@if command -v flake8 > /dev/null; then \
		echo "Running flake8..."; \
		flake8 app/ || true; \
	else \
		echo "flake8 not installed. Run: pip install flake8"; \
	fi
	@echo "Linting complete"

# Format: Format code
format:
	@echo "Formatting code..."
	@if command -v black > /dev/null; then \
		echo "Running black..."; \
		black app/ || true; \
	else \
		echo "black not installed. Run: pip install black"; \
	fi
	@echo "Formatting complete"

# Individual Service Controls

# Start Dashboard: Start dashboard service
start-dashboard:
	@$(SETUP_SCRIPT) start dashboard

# Start Collector: Start data collector service
start-collector:
	@$(SETUP_SCRIPT) start data-collector

# Start Model: Start model trainer service
start-model:
	@$(SETUP_SCRIPT) start model-trainer

# Install Dev Dependencies: Install development dependencies
install-dev:
	@echo "Installing development dependencies..."
	@$(PIP) install flake8 black pytest docker requests python-dotenv
	@echo "Development dependencies installed"

