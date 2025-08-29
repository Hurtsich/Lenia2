# Makefile for the Lenia GPU Project

# Define the virtual environment directory and python executable
VENV_DIR = venv
VENV_PYTHON = $(VENV_DIR)/bin/python

# Phony targets are not real files
.PHONY: all install test run clean

# Default target when running `make`
all: run

# Target: install - Creates a virtual environment and installs dependencies.
# This target is idempotent. It will only create the venv if it doesn't exist,
# and pip will handle the rest.
install:
	@echo "---> Setting up virtual environment and installing dependencies..."
	test -d $(VENV_DIR) || python3 -m venv $(VENV_DIR)
	$(VENV_PYTHON) -m pip install -r requirements.txt
	@echo "\nInstallation complete."

# Target: test - Runs the pytest test suite.
test:
	@echo "---> Running test suite..."
	$(VENV_PYTHON) -m pytest

# Target: run - Runs the main Lenia application.
run:
	@echo "---> Launching Lenia application..."
	$(VENV_PYTHON) main.py

# Target: run-spatial - Runs the main Lenia application with spatial convolution.
run-spatial:
	@echo "---> Launching Lenia application (spatial convolution)..."
	$(VENV_PYTHON) main.py --spatial

# Target: clean - Removes the virtual environment and other generated files.
clean:
	@echo "---> Cleaning up project..."
	rm -rf $(VENV_DIR)
	rm -f tests/golden_master.npy
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	@echo "Cleanup complete."


