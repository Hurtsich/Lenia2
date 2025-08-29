import cupy as np
from cupyx.scipy.signal import convolve2d
import config

class LeniaSpatial:
    def __init__(self):
        self.kernel_radius = config.KERNEL_RADIUS
        self.timestep = config.TIMESTEP
        self.mu = 0.15
        self.sigma = 0.015
        self.kernel_shape = "gaussian" # Default to a better kernel
        self.kernel_shapes = ["ring", "gaussian", "square"]
        self.world = np.random.rand(config.GRID_SIZE, config.GRID_SIZE, dtype=np.float32)
        self.kernel = self._create_kernel(self.kernel_radius, self.kernel_shape)

    def _create_kernel(self, radius, shape):
        radius = int(radius)
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        distance = np.sqrt(x**2 + y**2)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

        if shape == "ring":
            kernel[(distance > radius * 0.38) & (distance < radius * 0.62)] = 1
        elif shape == "gaussian":
            # A smoother Gaussian kernel is generally better
            kernel = np.exp(-(distance**2) / (2 * (radius / 3)**2))
        elif shape == "square":
            kernel[(-radius <= x) & (x <= radius) & (-radius <= y) & (y <= radius)] = 1

        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)
        
        return kernel

    def _growth(self, x):
        # Canonical Lenia growth function
        return np.exp(-((x - self.mu)**2) / (2 * self.sigma**2)) * 2 - 1

    def update(self):
        potential = convolve2d(self.world, self.kernel, mode='same', boundary='wrap')

        growth_val = self._growth(potential)
        self.world = np.clip(self.world + self.timestep * growth_val, 0, 1)

    def get_world(self):
        return self.world

    def set_kernel_radius(self, radius):
        self.kernel_radius = max(1, radius)
        self.kernel = self._create_kernel(self.kernel_radius, self.kernel_shape)

    def set_timestep(self, timestep):
        self.timestep = max(0.01, timestep)

    def set_kernel_shape(self, shape):
        if shape in self.kernel_shapes:
            self.kernel_shape = shape
            self.kernel = self._create_kernel(self.kernel_radius, self.kernel_shape)

    def set_mu(self, mu):
        self.mu = mu

    def set_sigma(self, sigma):
        self.sigma = sigma

    def randomize_world(self):
        self.world = np.random.rand(config.GRID_SIZE, config.GRID_SIZE, dtype=np.float32)
