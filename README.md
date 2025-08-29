# Lenia 2: GPU-Accelerated Lenia Simulation

This project is a high-performance, interactive Python implementation of the Lenia cellular automata, originally created by Bert Chan. It leverages GPU acceleration for both the simulation logic and rendering to support large, high-resolution grids and smooth, real-time interaction.

![Lenia Simulation](lenia.gif)

## About

Lenia is a continuous cellular automaton that produces a wide variety of complex, life-like, self-organizing patterns. This implementation was built from a simple script into a robust application with a focus on performance and interactivity.

The core simulation loop is executed on the GPU using CuPy for numerical computations and a Fast Fourier Transform (FFT) based convolution, which is significantly faster than traditional methods. Rendering is also offloaded to the GPU using ModernGL, allowing for a high frame rate even with large grid sizes.

## Features

- **GPU-Accelerated Computation**: Core simulation logic runs on the GPU via CuPy, using an efficient FFT-based convolution.
- **GPU-Accelerated Rendering**: Real-time visualization is handled by ModernGL, bypassing slow CPU-based drawing.
- **Interactive GUI**: A feature-rich control panel built with `pygame-gui` allows for real-time manipulation of all key simulation parameters.
- **Correct & Tunable Algorithm**: Implements the canonical Lenia growth function with known-good parameters as a default, while allowing users to explore the parameter space.
- **Clean, Separated Layout**: The simulation and control panel are rendered into two distinct, non-overlapping canvases for a clear user experience.
- **Robust Test Suite**: Includes a smoke test to ensure the application launches and a regression test (`pytest`) to verify the correctness of the simulation algorithm.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

This project requires an **NVIDIA GPU** with the **CUDA Toolkit** installed, as it relies on CuPy for GPU computation.

- Python 3.8+
- `venv` for virtual environments
- An NVIDIA GPU
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (the version should be compatible with the `cupy-cudaXXx` package in `requirements.txt`)

### Installation

1. **Clone the repository:**
   ```sh
   git clone <your-repository-url>
   cd Lenia2
   ```

2. **Create and activate a Python virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate
   ```
   *(On Windows, use `venv\Scripts\activate`)*

3. **Install the required packages:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

To run the main application, execute the following command:

```sh
python main.py
```

You can interact with the simulation using the control panel at the bottom of the window. The sliders and dropdowns allow you to change the kernel, timestep, and growth function parameters in real-time.

### Running the Test Suite

The project includes a test suite to verify its integrity and the correctness of the algorithm.

1. **Run all tests:**
   ```sh
   pytest
   ```

2. **Updating the Algorithm Baseline:**
   If you intentionally modify the core Lenia algorithm in `lenia_core.py`, the regression test will fail. To accept the new changes and create a new "golden master" file for the test to compare against, run the following script:
   ```sh
   python scripts/generate_master.py
   ```
   After regenerating the master file, run `pytest` again to confirm the tests pass.

## Technologies Used

- **Simulation**: [CuPy](https://cupy.dev/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/)
- **Graphics & UI**: [Pygame](https://www.pygame.org/), [Pygame GUI](https://pygame-gui.readthedocs.io/), [ModernGL](https://moderngl.readthedocs.io/)
- **Testing**: [Pytest](https://docs.pytest.org/)
