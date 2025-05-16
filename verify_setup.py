#!/usr/bin/env python3
"""
Verify Setup Script for Stock Price Predictor

This script checks if the project is properly set up by verifying:
1. Required files exist
2. File permissions are correct
3. Environment configuration is valid
4. Docker is configured correctly
5. Python packages are compatible

Run this script to catch configuration issues early.
"""

import os
import sys
import platform
import subprocess
import importlib.util
import pkg_resources
import re
from pathlib import Path
import json
import yaml
import socket
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Set

# ANSI color codes for terminal output
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"

# Project directories
PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
APP_DIR = PROJECT_DIR / "app"
DATA_DIR = PROJECT_DIR / "data"
LOGS_DIR = DATA_DIR / "logs"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Result tracking
@dataclass
class CheckResult:
    """Result of a verification check"""
    category: str
    check_name: str
    status: bool
    message: str
    fix: Optional[str] = None
    severity: str = "error"  # error, warning, info

# Store check results
results: List[CheckResult] = []

def add_result(category: str, check_name: str, status: bool, message: str, 
               fix: Optional[str] = None, severity: str = "error") -> None:
    """Add a check result to the results list"""
    results.append(CheckResult(
        category=category,
        check_name=check_name,
        status=status,
        message=message,
        fix=fix,
        severity=severity
    ))

def print_header(text: str) -> None:
    """Print a formatted header"""
    print(f"\n{BOLD}{BLUE}{'='*80}{RESET}")
    print(f"{BOLD}{BLUE}{text.center(80)}{RESET}")
    print(f"{BOLD}{BLUE}{'='*80}{RESET}\n")

def print_subheader(text: str) -> None:
    """Print a formatted subheader"""
    print(f"\n{BOLD}{CYAN}{text}{RESET}")
    print(f"{CYAN}{'-' * len(text)}{RESET}")

def run_command(command: List[str]) -> Tuple[int, str, str]:
    """Run a command and return exit code, stdout, and stderr"""
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        return process.returncode, stdout, stderr
    except Exception as e:
        return 1, "", str(e)

# 1. Check required files
def check_required_files() -> None:
    """Check if all required files exist in the project"""
    print_subheader("Checking Required Files")
    
    # App Python files
    app_files = [
        "config.py",
        "data_collector.py",
        "ml_model.py",
        "dashboard.py",
        "__init__.py"
    ]
    
    # Project configuration files
    config_files = [
        "Dockerfile",
        "docker-compose.yml",
        "requirements.txt",
        "setup.sh",
        "Makefile",
        ".env.example"
    ]
    
    # Check app Python files
    for file in app_files:
        file_path = APP_DIR / file
        exists = file_path.exists()
        status = f"{GREEN}Found{RESET}" if exists else f"{RED}Missing{RESET}"
        print(f"  {status} App file: {file}")
        
        if not exists:
            add_result(
                category="Required Files",
                check_name=f"App file: {file}",
                status=False,
                message=f"Required app file {file} is missing",
                fix=f"Create the {file} file in the app directory"
            )
        else:
            add_result(
                category="Required Files",
                check_name=f"App file: {file}",
                status=True,
                message=f"Required app file {file} found"
            )
    
    # Check project configuration files
    for file in config_files:
        file_path = PROJECT_DIR / file
        exists = file_path.exists()
        status = f"{GREEN}Found{RESET}" if exists else f"{RED}Missing{RESET}"
        print(f"  {status} Config file: {file}")
        
        if not exists:
            fix = f"Create the {file} file in the project directory"
            if file == ".env":
                fix = "Copy .env.example to .env and configure it"
                
            add_result(
                category="Required Files",
                check_name=f"Config file: {file}",
                status=False,
                message=f"Required configuration file {file} is missing",
                fix=fix
            )
        else:
            add_result(
                category="Required Files",
                check_name=f"Config file: {file}",
                status=True,
                message=f"Required configuration file {file} found"
            )
    
    # Check for .env file
    env_file = PROJECT_DIR / ".env"
    env_example = PROJECT_DIR / ".env.example"
    
    if not env_file.exists() and env_example.exists():
        print(f"  {YELLOW}Warning{RESET}: .env file not found, but .env.example exists")
        add_result(
            category="Required Files",
            check_name="Environment file",
            status=False,
            message=".env file not found, but .env.example exists",
            fix="Run './setup.sh setup' to create a .env file from the template",
            severity="warning"
        )
    elif not env_file.exists() and not env_example.exists():
        print(f"  {RED}Error{RESET}: Neither .env nor .env.example found")
        add_result(
            category="Required Files",
            check_name="Environment file",
            status=False,
            message="Neither .env nor .env.example file found",
            fix="Create a .env file with required configuration variables"
        )
    else:
        print(f"  {GREEN}Found{RESET} Environment file: .env")
        add_result(
            category="Required Files",
            check_name="Environment file",
            status=True,
            message=".env file found"
        )

# 2. Check file permissions
def check_file_permissions() -> None:
    """Check if file permissions are set correctly"""
    print_subheader("Checking File Permissions")
    
    # Check if setup.sh is executable
    setup_sh = PROJECT_DIR / "setup.sh"
    if setup_sh.exists():
        is_executable = os.access(setup_sh, os.X_OK)
        status = f"{GREEN}Yes{RESET}" if is_executable else f"{RED}No{RESET}"
        print(f"  setup.sh executable: {status}")
        
        if not is_executable:
            add_result(
                category="File Permissions",
                check_name="setup.sh executable",
                status=False,
                message="setup.sh is not executable",
                fix="Run 'chmod +x setup.sh' to make it executable"
            )
        else:
            add_result(
                category="File Permissions",
                check_name="setup.sh executable",
                status=True,
                message="setup.sh is executable"
            )
    
    # Create data directories if they don't exist
    for dir_path in [DATA_DIR, LOGS_DIR, RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
        if not dir_path.exists():
            print(f"  {YELLOW}Creating directory{RESET}: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    # Check if data directories are writable
    for dir_name, dir_path in [
        ("Data directory", DATA_DIR),
        ("Logs directory", LOGS_DIR),
        ("Raw data directory", RAW_DIR),
        ("Processed data directory", PROCESSED_DIR),
        ("Models directory", MODELS_DIR)
    ]:
        if dir_path.exists():
            is_writable = os.access(dir_path, os.W_OK)
            status = f"{GREEN}Yes{RESET}" if is_writable else f"{RED}No{RESET}"
            print(f"  {dir_name} writable: {status}")
            
            if not is_writable:
                add_result(
                    category="File Permissions",
                    check_name=f"{dir_name} writable",
                    status=False,
                    message=f"{dir_name} is not writable",
                    fix=f"Run 'chmod -R u+w {dir_path}' to make it writable"
                )
            else:
                add_result(
                    category="File Permissions",
                    check_name=f"{dir_name} writable",
                    status=True,
                    message=f"{dir_name} is writable"
                )
        else:
            print(f"  {YELLOW}Directory not found{RESET}: {dir_path}")
            add_result(
                category="File Permissions",
                check_name=f"{dir_name} exists",
                status=False,
                message=f"{dir_name} does not exist",
                fix=f"Run './setup.sh setup' to create the directory",
                severity="warning"
            )

# 3. Validate environment configuration
def check_environment_config() -> None:
    """Check if environment variables are properly configured"""
    print_subheader("Checking Environment Configuration")
    
    # Required environment variables
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
    
    # Load environment variables from .env file
    env_vars = {}
    env_file = PROJECT_DIR / ".env"
    
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
    
    # Check required variables
    for var in required_vars:
        if var in env_vars and env_vars[var] and not env_vars[var].startswith('${'):
            if var == "FINNHUB_API_KEY" and env_vars[var] == "d0ik8mpr01qrfsagmplgd0ik8mpr01qrfsagmpm0":
                print(f"  {YELLOW}Warning{RESET}: {var} is using default example value")
                add_result(
                    category="Environment Configuration",
                    check_name=f"{var} configuration",
                    status=False,
                    message=f"{var} is using the default example value",
                    fix="Update the .env file with your actual Finnhub API key",
                    severity="warning"
                )
            else:
                print(f"  {GREEN}Found{RESET}: {var}")
                add_result(
                    category="Environment Configuration",
                    check_name=f"{var} configuration",
                    status=True,
                    message=f"{var} is properly configured"
                )
        else:
            print(f"  {RED}Missing{RESET}: {var}")
            add_result(
                category="Environment Configuration",
                check_name=f"{var} configuration",
                status=False,
                message=f"Required environment variable {var} is missing or empty",
                fix=f"Add {var}=<value> to your .env file"
            )
    
    # Check optional variables
    for var in optional_vars:
        if var in env_vars and env_vars[var] and not env_vars[var].startswith('${'):
            print(f"  {GREEN}Found{RESET}: {var} = {env_vars[var]}")
            add_result(
                category="Environment Configuration",
                check_name=f"{var} configuration",
                status=True,
                message=f"Optional variable {var} is configured",
                severity="info"
            )
        else:
            print(f"  {YELLOW}Not set{RESET}: {var} (will use default)")
            add_result(
                category="Environment Configuration",
                check_name=f"{var} configuration",
                status=True,
                message=f"Optional variable {var} not set, will use default",
                severity="info"
            )

# 4. Check Docker configuration
def check_docker_config() -> None:
    """Check if Docker is properly configured"""
    print_subheader("Checking Docker Configuration")
    
    # Check if Docker is installed
    docker_installed = False
    returncode, stdout, stderr = run_command(["docker", "--version"])
    if returncode == 0:
        docker_installed = True
        docker_version = stdout.strip()
        print(f"  {GREEN}Docker installed{RESET}: {docker_version}")
        add_result(
            category="Docker Configuration",
            check_name="Docker installation",
            status=True,
            message=f"Docker is installed: {docker_version}"
        )
    else:
        print(f"  {RED}Docker not installed or not in PATH{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Docker installation",
            status=False,
            message="Docker is not installed or not in PATH",
            fix="Install Docker from https://docs.docker.com/get-docker/"
        )
    
    # Check if Docker Compose is installed
    docker_compose_installed = False
    returncode, stdout, stderr = run_command(["docker-compose", "--version"])
    if returncode == 0:
        docker_compose_installed = True
        compose_version = stdout.strip()
        print(f"  {GREEN}Docker Compose installed{RESET}: {compose_version}")
        add_result(
            category="Docker Configuration",
            check_name="Docker Compose installation",
            status=True,
            message=f"Docker Compose is installed: {compose_version}"
        )
    else:
        print(f"  {RED}Docker Compose not installed or not in PATH{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Docker Compose installation",
            status=False,
            message="Docker Compose is not installed or not in PATH",
            fix="Install Docker Compose from https://docs.docker.com/compose/install/"
        )
    
    # Only continue with Docker checks if Docker and Docker Compose are installed
    if not (docker_installed and docker_compose_installed):
        print(f"  {YELLOW}Skipping further Docker checks{RESET}")
        return
    
    # Check if Docker daemon is running
    returncode, stdout, stderr = run_command(["docker", "info"])
    if returncode == 0:
        print(f"  {GREEN}Docker daemon is running{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Docker daemon",
            status=True,
            message="Docker daemon is running"
        )
    else:
        print(f"  {RED}Docker daemon is not running{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Docker daemon",
            status=False,
            message="Docker daemon is not running",
            fix="Start the Docker daemon service"
        )
        return  # Skip remaining Docker checks if daemon is not running
    
    # Validate docker-compose.yml
    docker_compose_file = PROJECT_DIR / "docker-compose.yml"
    if docker_compose_file.exists():
        try:
            with open(docker_compose_file, 'r') as f:
                docker_compose_yaml = yaml.safe_load(f)
            
            # Check for required services
            required_services = ["zookeeper", "kafka", "data-collector", "model-trainer", "dashboard"]
            missing_services = []
            
            for service in required_services:
                if 'services' in docker_compose_yaml and service not in docker_compose_yaml['services']:
                    missing_services.append(service)
            
            if missing_services:
                services_str = ", ".join(missing_services)
                print(f"  {RED}docker-compose.yml is missing services: {services_str}{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Docker Compose services",
                    status=False,
                    message=f"docker-compose.yml is missing required services: {services_str}",
                    fix="Add the missing services to docker-compose.yml"
                )
            else:
                print(f"  {GREEN}docker-compose.yml contains all required services{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Docker Compose services",
                    status=True,
                    message="docker-compose.yml contains all required services"
                )
                
            # Check network configuration
            if 'networks' in docker_compose_yaml:
                print(f"  {GREEN}docker-compose.yml has network configuration{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Docker Compose networks",
                    status=True,
                    message="docker-compose.yml has network configuration"
                )
            else:
                print(f"  {YELLOW}docker-compose.yml is missing network configuration{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Docker Compose networks",
                    status=False,
                    message="docker-compose.yml is missing network configuration",
                    fix="Add a network configuration section to docker-compose.yml",
                    severity="warning"
                )
                
            # Check volume configuration
            if 'volumes' in docker_compose_yaml:
                print(f"  {GREEN}docker-compose.yml has volume configuration{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Docker Compose volumes",
                    status=True,
                    message="docker-compose.yml has volume configuration"
                )
            else:
                print(f"  {YELLOW}docker-compose.yml is missing volume configuration{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Docker Compose volumes",
                    status=False,
                    message="docker-compose.yml is missing volume configuration",
                    fix="Add a volume configuration section to docker-compose.yml",
                    severity="warning"
                )
                
        except Exception as e:
            print(f"  {RED}Error parsing docker-compose.yml: {e}{RESET}")
            add_result(
                category="Docker Configuration",
                check_name="Docker Compose file",
                status=False,
                message=f"Error parsing docker-compose.yml: {e}",
                fix="Fix the syntax in docker-compose.yml"
            )
    else:
        print(f"  {RED}docker-compose.yml not found{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Docker Compose file",
            status=False,
            message="docker-compose.yml not found",
            fix="Create a docker-compose.yml file in the project directory"
        )

    # Check Dockerfile
    dockerfile = PROJECT_DIR / "Dockerfile"
    if dockerfile.exists():
        print(f"  {GREEN}Dockerfile found{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Dockerfile",
            status=True,
            message="Dockerfile found"
        )
        
        # Quick validation of Dockerfile content
        try:
            with open(dockerfile, 'r') as f:
                dockerfile_content = f.read()
            
            if "FROM" not in dockerfile_content:
                print(f"  {RED}Dockerfile is missing FROM directive{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Dockerfile content",
                    status=False,
                    message="Dockerfile is missing FROM directive",
                    fix="Add a FROM directive to the Dockerfile"
                )
            else:
                print(f"  {GREEN}Dockerfile contains basic directives{RESET}")
                add_result(
                    category="Docker Configuration",
                    check_name="Dockerfile content",
                    status=True,
                    message="Dockerfile contains basic directives"
                )
        except Exception as e:
            print(f"  {RED}Error reading Dockerfile: {e}{RESET}")
    else:
        print(f"  {RED}Dockerfile not found{RESET}")
        add_result(
            category="Docker Configuration",
            check_name="Dockerfile",
            status=False,
            message="Dockerfile not found",
            fix="Create a Dockerfile in the project directory"
        )

# 5. Check Python package compatibility
def check_python_packages() -> None:
    """Check Python package compatibility"""
    print_subheader("Checking Python Package Compatibility")
    
    # Check Python version
    python_version = platform.python_version()
    print(f"  Python version: {python_version}")
    
    # Parse requirements.txt
    requirements_file = PROJECT_DIR / "requirements.txt"
    if not requirements_file.exists():
        print(f"  {RED}requirements.txt not found{RESET}")
        add_result(
            category="Python Packages",
            check_name="Requirements file",
            status=False,
            message="requirements.txt not found",
            fix="Create a requirements.txt file with required packages"
        )
        return
    
    # Parse requirements
    requirements = []
    try:
        with open(requirements_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    # Extract package name and version
                    if '==' in line:
                        package_name, package_version = line.split('==', 1)
                        requirements.append((package_name.strip(), package_version.strip()))
                    else:
                        requirements.append((line.strip(), None))
        
        print(f"  {GREEN}Found {len(requirements)} packages in requirements.txt{RESET}")
        add_result(
            category="Python Packages",
            check_name="Requirements file",
            status=True,
            message=f"Found {len(requirements)} packages in requirements.txt"
        )
    except Exception as e:
        print(f"  {RED}Error parsing requirements.txt: {e}{RESET}")
        add_result(
            category="Python Packages",
            check_name="Requirements file",
            status=False,
            message=f"Error parsing requirements.txt: {e}",
            fix="Fix the syntax in requirements.txt"
        )
        return
    
    # Check essential packages
    essential_packages = [
        "requests", "kafka-python", "pyspark", "streamlit", 
        "scikit-learn", "pandas", "numpy", "plotly"
    ]
    
    found_packages = [pkg_name for pkg_name, _ in requirements]
    missing_essential = [pkg for pkg in essential_packages if pkg not in found_packages]
    
    if missing_essential:
        missing_str = ", ".join(missing_essential)
        print(f"  {RED}Missing essential packages: {missing_str}{RESET}")
        add_result(
            category="Python Packages",
            check_name="Essential packages",
            status=False,
            message=f"Missing essential packages: {missing_str}",
            fix=f"Add the missing packages to requirements.txt"
        )
    else:
        print(f"  {GREEN}All essential packages found in requirements.txt{RESET}")
        add_result(
            category="Python Packages",
            check_name="Essential packages",
            status=True,
            message="All essential packages found in requirements.txt"
        )
    
    # Check for package conflicts (basic check)
    print(f"  {BLUE}Checking for potential package conflicts...{RESET}")
    conflicts = []
    
    # Known incompatible versions (simplified)
    incompatible_pairs = [
        ("pandas", "2.0.0", "numpy", "1.20.0"),  # Example: pandas 2.0.0 requires numpy >= 1.20.0
    ]
    
    # Check for incompatible pairs
    for pkg1_name, pkg1_min_ver, pkg2_name, pkg2_min_ver in incompatible_pairs:
        pkg1_info = next(((name, ver) for name, ver in requirements if name == pkg1_name), None)
        pkg2_info = next(((name, ver) for name, ver in requirements if name == pkg2_name), None)
        
        if pkg1_info and pkg2_info and pkg1_info[1] and pkg2_info[1]:
            pkg1_ver = pkg1_info[1]
            pkg2_ver = pkg2_info[1]
            
            # Very basic version comparison (would need a proper version comparison library for real use)
            if pkg1_ver >= pkg1_min_ver and pkg2_ver < pkg2_min_ver:
                conflicts.append(f"{pkg1_name}=={pkg1_ver} requires {pkg2_name}>={pkg2_min_ver}, but found {pkg2_ver}")
    
    if conflicts:
        conflicts_str = "\n    - ".join(conflicts)
        print(f"  {YELLOW}Potential package conflicts:{RESET}\n    - {conflicts_str}")
        add_result(
            category="Python Packages",
            check_name="Package conflicts",
            status=False,
            message=f"Potential package conflicts detected",
            fix="Update package versions to resolve conflicts",
            severity="warning"
        )
    else:
        print(f"  {GREEN}No obvious package conflicts detected{RESET}")
        add_result(
            category="Python Packages",
            check_name="Package conflicts",
            status=True,
            message="No obvious package conflicts detected"
        )
    
    # Try to import key modules (optional, only if they're installed)
    print(f"  {BLUE}Checking if key packages can be imported...{RESET}")
    key_packages = ["pandas", "numpy", "requests"]
    
    for package in key_packages:
        if importlib.util.find_spec(package):
            print(f"  {GREEN}Successfully imported {package}{RESET}")
        else:
            print(f"  {YELLOW}Could not import {package} (not installed in current environment){RESET}")

# Generate summary report
def generate_report() -> bool:
    """Generate a summary report of all check results
    
    Returns:
        bool: True if all checks passed, False otherwise
    """
    print_header("SUMMARY REPORT")
    
    # Group results by category
    categories = {}
    for result in results:
        if result.category not in categories:
            categories[result.category] = []
        categories[result.category].append(result)
    
    # Count totals
    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.status)
    warning_checks = sum(1 for r in results if not r.status and r.severity == "warning")
    failed_checks = sum(1 for r in results if not r.status and r.severity == "error")
    
    # Print category summaries
    for category, category_results in categories.items():
        print(f"\n{BOLD}{category}:{RESET}")
        for result in category_results:
            if result.status:
                status_color = GREEN
                status_text = "PASS"
            elif result.severity == "warning":
                status_color = YELLOW
                status_text = "WARN"
            else:
                status_color = RED
                status_text = "FAIL"
            
            print(f"  [{status_color}{status_text}{RESET}] {result.check_name}")
            if not result.status and result.fix:
                print(f"       {BLUE}Fix:{RESET} {result.fix}")
    
    # Print summary statistics
    print(f"\n{BOLD}Summary:{RESET}")
    print(f"Total checks: {total_checks}")
    print(f"{GREEN}Passed: {passed_checks}{RESET}")
    print(f"{YELLOW}Warnings: {warning_checks}{RESET}")
    print(f"{RED}Failed: {failed_checks}{RESET}\n")
    
    # Return overall status
    all_passed = failed_checks == 0
    return all_passed

# Main execution
def main():
    """Main execution function"""
    print_header("Stock Price Predictor - Setup Verification")
    
    check_required_files()
    check_file_permissions()
    check_environment_config()
    check_docker_config()
    check_python_packages()
    
    success = generate_report()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
