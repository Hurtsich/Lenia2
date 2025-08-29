import pytest
import cupy
import numpy as np
from lenia_core import Lenia
import config

def test_kernel_creation():
    """Tests the creation of the kernel."""
    lenia = Lenia()
    # Test for gaussian kernel
    lenia.set_kernel_shape("gaussian")
    kernel_fft = lenia._create_kernel_fft(lenia.kernel_radius, lenia.kernel_shape)
    assert isinstance(kernel_fft, cupy.ndarray)
    # Test for ring kernel
    lenia.set_kernel_shape("ring")
    kernel_fft = lenia._create_kernel_fft(lenia.kernel_radius, lenia.kernel_shape)
    assert isinstance(kernel_fft, cupy.ndarray)
    # Test for square kernel
    lenia.set_kernel_shape("square")
    kernel_fft = lenia._create_kernel_fft(lenia.kernel_radius, lenia.kernel_shape)
    assert isinstance(kernel_fft, cupy.ndarray)

def test_growth_function():
    """Tests the growth function."""
    lenia = Lenia()
    # Test with mu
    assert np.isclose(lenia._growth(lenia.mu), 1.0)
    # Test with value far from mu
    assert lenia._growth(100) < 0

def test_update_predictable():
    """Tests the update function with a predictable scenario."""
    # Use a small grid for predictability
    config.GRID_SIZE = 5
    config.KERNEL_RADIUS = 2
    lenia = Lenia()
    # Start with a blank world
    lenia.world = cupy.zeros((5, 5), dtype=cupy.float32)
    # Add a single pixel in the center
    lenia.world[2, 2] = 1.0

    initial_world = lenia.world.copy()
    lenia.update()
    updated_world = lenia.world

    # The center pixel should have changed
    assert initial_world[2, 2] != updated_world[2, 2]
    # The corners should remain unchanged
    assert initial_world[0, 0] == updated_world[0, 0]

def test_world_is_cupy_array():
    """Checks that the Lenia world is a CuPy array."""
    lenia = Lenia()
    assert isinstance(lenia.get_world(), cupy.ndarray)

def test_world_has_correct_shape():
    """Checks that the Lenia world has the expected dimensions."""
    lenia = Lenia()
    assert lenia.get_world().shape == (config.GRID_SIZE, config.GRID_SIZE)

def test_world_values_are_in_range():
    """Checks that all values in the Lenia world are between 0 and 1."""
    lenia = Lenia()
    world = lenia.get_world()
    assert cupy.all(world >= 0)
    assert cupy.all(world <= 1)

def test_update_changes_world():
    """Checks that the update function modifies the world state."""
    lenia = Lenia()
    initial_world = lenia.get_world().copy()
    lenia.update()
    updated_world = lenia.get_world()
    assert not cupy.all(initial_world == updated_world)

def test_set_kernel_radius():
    """Checks that the kernel radius can be set correctly."""
    lenia = Lenia()
    lenia.set_kernel_radius(10)
    assert lenia.kernel_radius == 10

def test_set_timestep():
    """Checks that the timestep can be set correctly."""
    lenia = Lenia()
    lenia.set_timestep(0.05)
    assert lenia.timestep == 0.05

def test_set_kernel_shape():
    """Checks that the kernel shape can be set correctly."""
    lenia = Lenia()
    lenia.set_kernel_shape("ring")
    assert lenia.kernel_shape == "ring"

def test_set_mu():
    """Checks that mu can be set correctly."""
    lenia = Lenia()
    lenia.set_mu(0.2)
    assert lenia.mu == 0.2

def test_set_sigma():
    """Checks that sigma can be set correctly."""
    lenia = Lenia()
    lenia.set_sigma(0.02)
    assert lenia.sigma == 0.02

def test_randomize_world():
    """Checks that the world is randomized."""
    lenia = Lenia()
    initial_world = lenia.get_world().copy()
    lenia.randomize_world()
    randomized_world = lenia.get_world()
    assert not cupy.all(initial_world == randomized_world)