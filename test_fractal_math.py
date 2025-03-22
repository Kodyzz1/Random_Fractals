# test_fractal_math.py
import cupy as cp
import fractal_math

# Test Mandelbrot
c_mandelbrot = cp.array([[complex(-0.1, 0.1), complex(-0.1, 0.1)]])
iterations_mandelbrot = fractal_math.mandelbrot_gpu(c_mandelbrot, 100)
print("Mandelbrot Iterations:", iterations_mandelbrot)
print(f"z: {z}")

# Test Julia
c_julia = cp.array([[complex(-0.8, 0.156), complex(-0.7, 0.2)]])
z_julia = cp.array([[complex(-0.5, 0.5), complex(0.0, 0.0)]])
iterations_julia = fractal_math.julia_set_gpu(c_julia, z_julia, 100)
print("Julia Iterations:", iterations_julia)

# Test Burning Ship
c_burning_ship = cp.array([[complex(-0.5, 0.5), complex(0.0, 0.0)]])
iterations_burning_ship = fractal_math.burning_ship_gpu(c_burning_ship, 100)
print("Burning Ship Iterations:", iterations_burning_ship)

# Test Noisy Mandelbrot
c_noisy_mandelbrot = cp.array([[complex(-0.5, 0.5), complex(0.0, 0.0)]])
iterations_noisy_mandelbrot = fractal_math.noisy_mandelbrot_gpu(c_noisy_mandelbrot, 100, 5.0, 0.1, 6, 0.5, 2.0, seed=42)
print("Noisy Mandelbrot Iterations:", iterations_noisy_mandelbrot)
print(f"z: {z}")