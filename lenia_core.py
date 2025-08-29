import cupy as np
import config

class Lenia:
    def __init__(self):
        self.kernel_radius = config.KERNEL_RADIUS
        self.timestep = config.TIMESTEP
        self.mu = 0.15
        self.sigma = 0.015
        self.kernel_shape = "gaussian" # Default to a better kernel
        self.kernel_shapes = ["ring", "gaussian", "square"]
        self.world = np.random.rand(config.GRID_SIZE, config.GRID_SIZE, dtype=np.float32)
        self.kernel_fft = self._create_kernel_fft(self.kernel_radius, self.kernel_shape)

    def _create_kernel_fft(self, radius, shape):
        radius = int(radius)
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        distance = np.sqrt(x**2 + y**2)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)

        if shape == "ring":
            kernel[distance > radius * 0.38] = 1
            kernel[distance > radius * 0.62] = 0
        elif shape == "gaussian":
            # A smoother Gaussian kernel is generally better
            kernel = np.exp(-(distance**2) / (2 * (radius / 3)**2))
        elif shape == "square":
            kernel[(-radius <= x) & (x <= radius) & (-radius <= y) & (y <= radius)] = 1

        if np.sum(kernel) > 0:
            kernel = kernel / np.sum(kernel)

        # Pad kernel to world size and shift for FFT
        padded_kernel = np.zeros((config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32)
        pad_x, pad_y = (config.GRID_SIZE - kernel.shape[0]) // 2, (config.GRID_SIZE - kernel.shape[1]) // 2
        padded_kernel[pad_x:pad_x+kernel.shape[0], pad_y:pad_y+kernel.shape[1]] = kernel
        padded_kernel = np.roll(padded_kernel, (-radius, -radius), axis=(0, 1))
        
        return np.fft.rfft2(padded_kernel)

    def _growth(self, x):
        # Canonical Lenia growth function
        return np.exp(-((x - self.mu)**2) / (2 * self.sigma**2)) * 2 - 1

    def update(self):
        world_fft = np.fft.rfft2(self.world)
        potential_fft = world_fft * self.kernel_fft
        potential = np.fft.irfft2(potential_fft)

        growth_val = self._growth(potential)
        self.world = np.clip(self.world + self.timestep * growth_val, 0, 1)

    def get_world(self):
        return self.world

    def set_kernel_radius(self, radius):
        self.kernel_radius = max(1, radius)
        self.kernel_fft = self._create_kernel_fft(self.kernel_radius, self.kernel_shape)

    def set_timestep(self, timestep):
        self.timestep = max(0.01, timestep)

    def set_kernel_shape(self, shape):
        if shape in self.kernel_shapes:
            self.kernel_shape = shape
            self.kernel_fft = self._create_kernel_fft(self.kernel_radius, self.kernel_shape)

    def set_mu(self, mu):
        self.mu = mu

    def set_sigma(self, sigma):
        self.sigma = sigma

    def randomize_world(self):
        self.world = np.random.rand(config.GRID_SIZE, config.GRID_SIZE, dtype=np.float32)
