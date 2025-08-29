import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
GRID_SIZE = 256
TIMESTEP = 0.1
KERNEL_RADIUS = 13

# --- 1. The World ---
# Initialize the world grid with random values
world = np.random.rand(GRID_SIZE, GRID_SIZE).astype(np.float32)

# --- 2. The Kernel ---
# The kernel defines the neighborhood influence.
# We will create a simple Gaussian-like kernel.
X, Y = np.ogrid[-KERNEL_RADIUS:KERNEL_RADIUS+1, -KERNEL_RADIUS:KERNEL_RADIUS+1]
distance = np.sqrt(X**2 + Y**2)
kernel = np.zeros((2 * KERNEL_RADIUS + 1, 2 * KERNEL_RADIUS + 1), dtype=np.float32)
# A simple ring-like kernel
kernel[distance > 5] = 1
kernel[distance > 8] = 0

# Normalize the kernel
kernel = kernel / np.sum(kernel)

# Pad kernel to world size and shift for FFT
padded_kernel = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
pad_x, pad_y = (GRID_SIZE - kernel.shape[0]) // 2, (GRID_SIZE - kernel.shape[1]) // 2
padded_kernel[pad_x:pad_x+kernel.shape[0], pad_y:pad_y+kernel.shape[1]] = kernel
padded_kernel = np.roll(padded_kernel, (-KERNEL_RADIUS, -KERNEL_RADIUS), axis=(0, 1))
kernel_fft = np.fft.rfft2(padded_kernel)

# --- 3. The Growth Function ---
# The growth function determines the next state of a cell.
# This is a simple bell-shaped growth function.
mu = 0.15
sigma = 0.015
def growth(x):
    return np.exp(-((x - mu)**2) / (2 * sigma**2)) * 2 - 1

# --- 4. The Simulation Loop ---
def update(frame):
    global world
    # Perform convolution using FFT
    world_fft = np.fft.rfft2(world)
    potential_fft = world_fft * kernel_fft
    potential = np.fft.irfft2(potential_fft, s=world.shape)
    
    # Apply the growth function
    growth_val = growth(potential)
    
    # Update the world
    world = np.clip(world + TIMESTEP * growth_val, 0, 1)
    
    # Update the plot
    im.set_array(world)
    return [im]

# --- 5. Visualization ---
fig = plt.figure()
im = plt.imshow(world, cmap='viridis', animated=True)

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)

ani.save("lenia.gif", writer='pillow', fps=15)

print("Lenia simulation finished. The animation has been saved to lenia.gif")
