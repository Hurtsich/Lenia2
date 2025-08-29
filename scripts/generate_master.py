# This script generates the 'golden master' file for the regression test.
import os
import sys

# Add project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import cupy
import numpy
from lenia_core import Lenia

# Ensure the tests directory exists
if not os.path.exists('tests'):
    os.makedirs('tests')

GOLDEN_MASTER_PATH = os.path.join('tests', 'golden_master.npy')

def generate_golden_master():
    """Runs the Lenia simulation for a fixed number of steps with a fixed seed and saves the result."""
    print(f"Generating new golden master file at: {GOLDEN_MASTER_PATH}")
    
    # Set fixed seeds for reproducibility
    numpy.random.seed(42)
    cupy.random.seed(42)

    # Initialize Lenia. It will use the default config values.
    lenia = Lenia()

    # Run for a fixed number of steps
    for _ in range(10):
        lenia.update()

    # Get the final world state and save it
    result = lenia.get_world().get()
    numpy.save(GOLDEN_MASTER_PATH, result)
    print("Golden master file generated successfully.")

if __name__ == "__main__":
    generate_golden_master()
