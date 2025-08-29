import pytest
import cupy
import numpy
import os
from lenia_core import Lenia

# The path to the golden master file, relative to this test file.
GOLDEN_MASTER_PATH = os.path.join(os.path.dirname(__file__), 'golden_master.npy')

# Check if the golden master file exists. If not, skip this test.
if not os.path.exists(GOLDEN_MASTER_PATH):
    pytest.skip("Golden master file not found. Run `venv/bin/python scripts/generate_master.py` to create it.", allow_module_level=True)

def run_simulation_for_test():
    """Runs the Lenia simulation for a fixed number of steps with a fixed seed."""
    # Set fixed seeds for reproducibility
    numpy.random.seed(42)
    cupy.random.seed(42)

    # Initialize Lenia.
    lenia = Lenia()

    # Run for a fixed number of steps
    for _ in range(10):
        lenia.update()

    # Return the final world state on the CPU
    return lenia.get_world().get()

def test_lenia_algorithm_correctness():
    """Compares the current simulation output against the golden master file."""
    # Load the trusted, correct result
    golden_master = numpy.load(GOLDEN_MASTER_PATH)

    # Generate a fresh result with the current code
    current_result = run_simulation_for_test()

    # Compare them. This will fail if the algorithm has changed.
    numpy.testing.assert_allclose(current_result, golden_master, rtol=1e-5, atol=1e-5)