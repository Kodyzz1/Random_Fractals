import cupy as cp

try:
    print("CuPy version:", cp.__version__)
    print("CuPy available:", cp.is_available())
except ImportError as e:
    print("CuPy import error:", e)