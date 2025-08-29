import pytest
import config

@pytest.fixture(autouse=True)
def restore_config():
    original_grid_size = config.GRID_SIZE
    original_kernel_radius = config.KERNEL_RADIUS
    yield
    config.GRID_SIZE = original_grid_size
    config.KERNEL_RADIUS = original_kernel_radius