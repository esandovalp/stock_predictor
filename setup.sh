#!/bin/bash
#
# Stock Price Predictor Setup Script
# This script helps with setup, management, and maintenance of the Stock Price Predictor project.
#

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Project directories
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${PROJECT_DIR}/data"
LOGS_DIR="${DATA_DIR}/logs"
RAW_DIR="${DATA_DIR}/raw"
PROCESSED_DIR="${DATA_DIR}/processed"
MODELS_DIR="${DATA_DIR}/models"

# Environment file
ENV_FILE="${PROJECT_DIR}/.env"
ENV_EXAMPLE="${PROJECT_DIR}/.env.example"

# Spinner for long-running tasks
spinner() {
    local pid=$1
    local delay=0.1
    local spinstr='|/-\'
    while [ "$(ps a | awk '{print $1}' | grep -w $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
}

# Progress indicator
progress() {
    local message=$1
    echo -ne "${CYAN}${message}...${NC}"
}

# Success message
success() {
    local message=$1
    echo -e " ${GREEN}✓ ${message}${NC}"
}

# Error message
error() {
    local message=$1
    echo -e "${RED}✗ ${message}${NC}"
}

# Warning message
warning() {
    local message=$1
    echo -e "${YELLOW}⚠ ${message}${NC}"
}

# Info message
info() {
    local message=$1
    echo -e "${BLUE}ℹ ${message}${NC}"
}

# Check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check system requirements
check_requirements() {
    echo -e "${BOLD}Checking system requirements...${NC}"
    
    # Check Docker
    progress "Checking Docker"
    if command_exists docker; then
        DOCKER_VERSION=$(docker --version | awk '{print $3}' | sed 's/,//')
        success "Docker $DOCKER_VERSION installed"
    else
        error "Docker not found. Please install Docker: https://docs.docker.com/get-docker/"
        return 1
    fi
    
    # Check Docker Compose
    progress "Checking Docker Compose"
    if command_exists docker-compose; then
        COMPOSE_VERSION=$(docker-compose --version | awk '{print $3}' | sed 's/,//')
        success "Docker Compose $COMPOSE_VERSION installed"
    else
        error "Docker Compose not found. Please install Docker Compose: https://docs.docker.com/compose/install/"
        return 1
    fi
    
    # Check Python
    progress "Checking Python"
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | awk '{print $2}')
        success "Python $PYTHON_VERSION installed"
        
        # Check required Python packages
        progress "Checking required Python packages"
        if ! command_exists pip3; then
            warning "pip3 not found. Some features may not work"
        else
            # Check for key packages
            if python3 -c "import docker, requests, dotenv" 2>/dev/null; then
                success "Required Python packages installed"
            else
                warning "Some required Python packages are missing. Run: pip3 install docker requests python-dotenv"
            fi
        fi
    else
        warning "Python 3 not found. Some features may not work"
    fi
    
    # Check Docker daemon is running
    progress "Checking Docker daemon"
    if docker info >/dev/null 2>&1; then
        success "Docker daemon is running"
    else
        error "Docker daemon is not running. Please start Docker"
        return 1
    fi
    
    echo -e "\n${GREEN}System check complete.${NC}\n"
    return 0
}

# Create necessary directories
create_directories() {
    echo -e "${BOLD}Creating necessary directories...${NC}"
    
    # Create main data directory
    progress "Creating data directory"
    mkdir -p "$DATA_DIR"
    success "Created"
    
    # Create subdirectories
    progress "Creating logs directory"
    mkdir -p "$LOGS_DIR"
    success "Created"
    
    progress "Creating raw data directory"
    mkdir -p "$RAW_DIR"
    success "Created"
    
    progress "Creating processed data directory"
    mkdir -p "$PROCESSED_DIR"
    success "Created"
    
    progress "Creating models directory"
    mkdir -p "$MODELS_DIR"
    success "Created"
    
    echo -e "\n${GREEN}Directories created successfully.${NC}\n"
}

# Set up environment file
setup_env() {
    echo -e "${BOLD}Setting up environment...${NC}"
    
    # Check if .env file already exists
    if [ -f "$ENV_FILE" ]; then
        info "Environment file already exists. Checking configuration..."
        
        # Check for required variables
        source "$ENV_FILE" 2>/dev/null
        if [ -z "$FINNHUB_API_KEY" ] || [ "$FINNHUB_API_KEY" = "d0ik8mpr01qrfsagmplgd0ik8mpr01qrfsagmpm0" ]; then
            warning "Finnhub API key is not set or using default value"
            setup_api_key
        else
            success "Finnhub API key is configured"
        fi
    else
        # Create .env file from example
        progress "Creating environment file from template"
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            success "Created"
            setup_api_key
        else
            error "Example environment file not found (.env.example)"
            return 1
        fi
    fi
    
    echo -e "\n${GREEN}Environment setup complete.${NC}\n"
}

# Set up Finnhub API key
setup_api_key() {
    echo -e "\n${YELLOW}Finnhub API key is required for data collection.${NC}"
    echo -e "${BLUE}You can get a free API key from https://finnhub.io/register${NC}"
    echo -n "Enter your Finnhub API key (or leave empty to skip): "
    read -r API_KEY
    
    if [ -n "$API_KEY" ]; then
        progress "Setting up API key"
        # Replace the API key in the .env file
        if [ "$(uname)" = "Darwin" ]; then  # macOS
            sed -i '' "s/FINNHUB_API_KEY=.*/FINNHUB_API_KEY=$API_KEY/" "$ENV_FILE"
        else  # Linux
            sed -i "s/FINNHUB_API_KEY=.*/FINNHUB_API_KEY=$API_KEY/" "$ENV_FILE"
        fi
        success "API key configured"
    else
        warning "Skipping API key setup. You'll need to edit .env file manually."
    fi
}

# Build Docker images
build_images() {
    echo -e "${BOLD}Building Docker images...${NC}"
    
    progress "Building Docker images"
    docker-compose build > "$LOGS_DIR/docker_build.log" 2>&1 &
    build_pid=$!
    spinner $build_pid
    
    if [ $? -eq 0 ]; then
        success "Docker images built successfully"
    else
        error "Failed to build Docker images. See logs at $LOGS_DIR/docker_build.log"
        return 1
    fi
    
    echo -e "\n${GREEN}Docker setup complete.${NC}\n"
}

# Start services
start_services() {
    local services=$1
    
    if [ -z "$services" ]; then
        echo -e "${BOLD}Starting all services...${NC}"
        progress "Starting Docker containers"
        docker-compose up -d > "$LOGS_DIR/docker_start.log" 2>&1 &
        start_pid=$!
        spinner $start_pid
        
        if [ $? -eq 0 ]; then
            success "All services started"
        else
            error "Failed to start services. See logs at $LOGS_DIR/docker_start.log"
            return 1
        fi
    else
        echo -e "${BOLD}Starting services: $services...${NC}"
        progress "Starting selected Docker containers"
        docker-compose up -d $services > "$LOGS_DIR/docker_start.log" 2>&1 &
        start_pid=$!
        spinner $start_pid
        
        if [ $? -eq 0 ]; then
            success "Services started: $services"
        else
            error "Failed to start services. See logs at $LOGS_DIR/docker_start.log"
            return 1
        fi
    fi
    
    # Display dashboard URL
    if [[ -z "$services" || "$services" == *"dashboard"* ]]; then
        echo -e "\n${GREEN}Dashboard is available at: ${BOLD}http://localhost:8501${NC}\n"
    fi
}

# Stop services
stop_services() {
    local services=$1
    
    if [ -z "$services" ]; then
        echo -e "${BOLD}Stopping all services...${NC}"
        progress "Stopping Docker containers"
        docker-compose down > "$LOGS_DIR/docker_stop.log" 2>&1 &
        stop_pid=$!
        spinner $stop_pid
        
        if [ $? -eq 0 ]; then
            success "All services stopped"
        else
            error "Failed to stop services. See logs at $LOGS_DIR/docker_stop.log"
            return 1
        fi
    else
        echo -e "${BOLD}Stopping services: $services...${NC}"
        progress "Stopping selected Docker containers"
        docker-compose stop $services > "$LOGS_DIR/docker_stop.log" 2>&1 &
        stop_pid=$!
        spinner $stop_pid
        
        if [ $? -eq 0 ]; then
            success "Services stopped: $services"
        else
            error "Failed to stop services. See logs at $LOGS_DIR/docker_stop.log"
            return 1
        fi
    fi
}

# View service logs
view_logs() {
    local service=$1
    local lines=$2
    
    if [ -z "$service" ]; then
        error "Service name is required (e.g., 'dashboard', 'data-collector')"
        return 1
    fi
    
    if [ -z "$lines" ]; then
        lines=50
    fi
    
    echo -e "${BOLD}Viewing logs for $service (last $lines lines)...${NC}"
    echo -e "${CYAN}Press Ctrl+C to exit${NC}\n"
    
    docker-compose logs --tail="$lines" -f "$service"
}

# Run tests
run_tests() {
    local test_args=$1
    
    echo -e "${BOLD}Running tests...${NC}"
    
    # Check if test_setup.py exists
    if [ ! -f "${PROJECT_DIR}/test_setup.py" ]; then
        error "Test script not found: test_setup.py"
        return 1
    fi
    
    # Check if Docker containers are running
    if ! docker-compose ps | grep -q Up; then
        warning "No running Docker containers detected. Some tests may fail."
        echo -n "Do you want to start the services before testing? (y/n): "
        read -r start_services
        
        if [ "$start_services" = "y" ] || [ "$start_services" = "Y" ]; then
            start_services
        fi
    fi
    
    # Run the test script
    if [ -z "$test_args" ]; then
        python3 "${PROJECT_DIR}/test_setup.py" --all
    else
        python3 "${PROJECT_DIR}/test_setup.py" $test_args
    fi
}

# Clean up data and containers
clean_up() {
    local mode=$1
    
    if [ "$mode" = "all" ]; then
        echo -e "${BOLD}Performing complete cleanup...${NC}"
        
        # Stop and remove containers
        progress "Stopping and removing containers"
        docker-compose down -v > "$LOGS_DIR/docker_cleanup.log" 2>&1 &
        cleanup_pid=$!
        spinner $cleanup_pid
        
        if [ $? -eq 0 ]; then
            success "Containers removed"
        else
            error "Failed to remove containers"
        fi
        
        # Remove data files
        progress "Removing data files"
        rm -rf "${DATA_DIR}"/* > /dev/null 2>&1
        success "Data files removed"
        
        # Recreate directories
        create_directories
        
    elif [ "$mode" = "data" ]; then
        echo -e "${BOLD}Cleaning up data only...${NC}"
        
        echo -n "This will delete all collected data. Are you sure? (y/n): "
        read -r confirm
        
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            progress "Removing data files"
            rm -rf "${RAW_DIR}"/* "${PROCESSED_DIR}"/* "${MODELS_DIR}"/* > /dev/null 2>&1
            success "Data files removed"
        else
            info "Operation cancelled"
        fi
        
    elif [ "$mode" = "containers" ]; then
        echo -e "${BOLD}Removing containers only...${NC}"
        
        progress "Stopping and removing containers"
        docker-compose down > "$LOGS_DIR/docker_cleanup.log" 2>&1 &
        cleanup_pid=$!
        spinner $cleanup_pid
        
        if [ $? -eq 0 ]; then
            success "Containers removed"
        else
            error "Failed to remove containers"
        fi
        
    else
        error "Invalid cleanup mode. Use 'all', 'data', or 'containers'"
        return 1
    fi
    
    echo -e "\n${GREEN}Cleanup complete.${NC}\n"
}

# Show project status
show_status() {
    echo -e "${BOLD}Stock Predictor Status${NC}\n"
    
    # Check Docker containers
    echo -e "${CYAN}Docker Containers:${NC}"
    docker-compose ps
    
    # Check data files
    echo -e "\n${CYAN}Data Files:${NC}"
    
    if [ -d "$RAW_DIR" ]; then
        raw_count=$(find "$RAW_DIR" -type f -name "*.csv" | wc -l)
        echo -e "Raw data files: ${BOLD}${raw_count}${NC}"
    fi
    
    if [ -d "$PROCESSED_DIR" ]; then
        processed_count=$(find "$PROCESSED_DIR" -type f -name "*.csv" | wc -l)
        echo -e "Processed data files: ${BOLD}${processed_count}${NC}"
    fi
    
    if [ -d "$MODELS_DIR" ]; then
        model_count=$(find "$MODELS_DIR" -type f -name "*.joblib" | wc -l)
        echo -e "Model files: ${BOLD}${model_count}${NC}"
    fi
    
    # Check model status
    echo -e "\n${CYAN}Model Status:${NC}"
    latest_model=$(find "$MODELS_DIR" -type f -name "*_price_model.joblib" -print0 | xargs -0 ls -t | head -n 1)
    if [ -n "$latest_model" ]; then
        model_time=$(stat -f "%Sm" "$latest_model" 2>/dev/null || stat -c "%y" "$latest_model" 2>/dev/null)
        echo -e "Latest model: ${BOLD}$(basename "$latest_model")${NC}"
        echo -e "Last updated: ${BOLD}${model_time}${NC}"
    else
        echo -e "${YELLOW}No trained models found${NC}"
    fi
    
    # Check API status
    echo -e "\n${CYAN}API Status:${NC}"
    if [ -f "$ENV_FILE" ]; then
        source "$ENV_FILE" 2>/dev/null
        if [ -n "$FINNHUB_API_KEY" ] && [ "$FINNHUB_API_KEY" != "d0ik8mpr01qrfsagmplgd0ik8mpr01qrfsagmpm0" ]; then
            echo -e "Finnhub API key: ${GREEN}Configured${NC}"
            
            # Try to access API (optional)
            if command_exists curl; then
                api_response=$(curl -s "https://finnhub.io/api/v1/quote?symbol=AAPL&token=$FINNHUB_API_KEY")
                if [[ "$api_response" == *"c"* ]]; then
                    echo -e "API connection: ${GREEN}Working${NC}"
                else
                    echo -e "API connection: ${RED}Error - ${api_response}${NC}"
                fi
            fi
        else
            echo -e "Finnhub API key: ${YELLOW}Not configured or using default${NC}"
        fi
    else
        echo -e "Environment file: ${RED}Not found${NC}"
    fi
    
    # System resource usage
    echo -e "\n${CYAN}System Resources:${NC}"
    
    # Check Docker disk usage
    if command_exists docker; then
        echo -e "\nDocker disk usage:"
        docker system df
    fi
    
    # Check system memory
    if command_exists free; then
        echo -e "\nSystem memory:"
        free -h
    elif [ "$(uname)" = "Darwin" ] && command_exists vm_stat; then  # macOS
        echo -e "\nSystem memory (macOS):"
        vm_stat
    fi
    
    echo -e "\n${GREEN}Status check complete.${NC}\n"
}

# Display help menu
help_menu() {
    echo -e "${BOLD}Stock Price Predictor - Help Menu${NC}\n"
    
    echo -e "${CYAN}Usage:${NC} $0 [command] [options]\n"
    
    echo -e "${CYAN}Commands:${NC}"
    echo -e "  ${BOLD}setup${NC}              Set up the project (check requirements, create directories, configure environment)"
    echo -e "  ${BOLD}start${NC}              Start all services or specific services"
    echo -e "  ${BOLD}stop${NC}               Stop all services or specific services"
    echo -e "  ${BOLD}restart${NC}            Restart all services or specific services"
    echo -e "  ${BOLD}logs${NC}               View logs for a specific service"
    echo -e "  ${BOLD}status${NC}             Show the current status of the project"
    echo -e "  ${BOLD}test${NC}               Run tests to verify the setup"
    echo -e "  ${BOLD}clean${NC}              Clean up data, containers, or both"
    echo -e "  ${BOLD}help${NC}               Display this help menu"
    
    echo -e "\n${CYAN}Examples:${NC}"
    echo -e "  $0 setup              # Set up the project environment"
    echo -e "  $0 start              # Start all services"
    echo -e "  $0 start dashboard    # Start only the dashboard service"
    echo -e "  $0 logs data-collector # View logs for the data collector"
    echo -e "  $0 test               # Run all tests"
    echo -e "  $0 clean data         # Clean up data files only"
    echo -e "  $0 status             # Show the current status of the project"
    
    echo -e "\n${CYAN}Service Names:${NC}"
    echo -e "  ${BOLD}zookeeper${NC}          ZooKeeper service (required for Kafka)"
    echo -e "  ${BOLD}kafka${NC}              Kafka message broker"
    echo -e "  ${BOLD}data-collector${NC}     Data collection service"
    echo -e "  ${BOLD}model-trainer${NC}      Model training service"
    echo -e "  ${BOLD}dashboard${NC}          Streamlit dashboard"
    
    echo -e "\n${CYAN}Notes:${NC}"
    echo -e "  - Docker and Docker Compose must be installed and running"
    echo -e "  - Configuration settings can be modified in the .env file"
    echo -e "  - Logs are stored in the ${LOGS_DIR} directory"
    echo -e "  - Run with no arguments to see this help menu"
    
    echo
}

# Main execution
main() {
    # If no arguments provided, show help menu
    if [ $# -eq 0 ]; then
        help_menu
        exit 0
    fi
    
    # Get the command (first argument)
    COMMAND=$1
    shift
    
    case "$COMMAND" in
        setup)
            check_requirements && create_directories && setup_env && build_images
            ;;
        
        start)
            # Check if specific services are specified
            if [ $# -eq 0 ]; then
                start_services
            else
                start_services "$*"
            fi
            ;;
        
        stop)
            # Check if specific services are specified
            if [ $# -eq 0 ]; then
                stop_services
            else
                stop_services "$*"
            fi
            ;;
        
        restart)
            # Check if specific services are specified
            if [ $# -eq 0 ]; then
                stop_services && start_services
            else
                stop_services "$*" && start_services "$*"
            fi
            ;;
        
        logs)
            if [ $# -eq 0 ]; then
                error "Service name required (e.g., 'dashboard', 'data-collector')"
                exit 1
            else
                view_logs "$1" "$2"
            fi
            ;;
        
        status)
            show_status
            ;;
        
        test)
            run_tests "$*"
            ;;
        
        clean)
            # Check if specific clean mode is specified
            if [ $# -eq 0 ]; then
                error "Clean mode required ('all', 'data', or 'containers')"
                exit 1
            else
                clean_up "$1"
            fi
            ;;
        
        help)
            help_menu
            ;;
        
        *)
            error "Unknown command: $COMMAND"
            echo "Run '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Make script executable
chmod +x "$0" 2>/dev/null

# Run main function with all arguments
main "$@"
