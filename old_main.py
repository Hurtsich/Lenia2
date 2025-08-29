import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation parameters
GRID_SIZE = 256
TIMESTEP = 0.1
KERNEL_RADIUS = 13

# --- 1. The World ---
# Initialize the world grid with random values
world = np.random.rand(GRID_SIZE, GRID_SIZE)

# --- 2. The Kernel ---
# The kernel defines the neighborhood influence.
# We will create a simple Gaussian-like kernel.
X, Y = np.ogrid[-KERNEL_RADIUS:KERNEL_RADIUS+1, -KERNEL_RADIUS:KERNEL_RADIUS+1]
distance = np.sqrt(X**2 + Y**2)
kernel = np.zeros((2 * KERNEL_RADIUS + 1, 2 * KERNEL_RADIUS + 1))
# A simple ring-like kernel
kernel[distance > 5] = 1
kernel[distance > 8] = 0

# Normalize the kernel
kernel = kernel / np.sum(kernel)

# --- 3. The Growth Function ---
# The growth function determines the next state of a cell.
# This is a simple bell-shaped growth function.
def growth(x):
    return np.exp(-((x - 0.5)**2) / 0.02) * 2 - 1

# --- 4. The Simulation Loop ---
# We use convolution from scipy for performance
from scipy.signal import convolve2d

def update(frame):
    global world
    # Perform convolution
    potential = convolve2d(world, kernel, mode='same', boundary='wrap')
    
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
