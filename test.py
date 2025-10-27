# Import required packages
import timeit
import platform
import sys
# import warnings  # Import warnings module <-- 已移除
import numpy as np
import scipy
import scipy.linalg as la
# Sparse imports removed
import scipy.fft as fft
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from sksparse.cholmod import cholesky
HAVE_CHOLMOD = True

# --- Benchmark configuration ---
# Number of samples per benchmark
N_SAMPLES = 10
# Dense matrix size (N x N)
MATRIX_SIZE = 2000
# Vector/array size
VECTOR_SIZE = 20_000_000
# Signal length for FFT and DSP
SIGNAL_SIZE = 20_000_000
# Number of data points for linear regression test
REGRESSION_POINTS = 5_000_000
# Interpolation points
INTERP_BASE_POINTS = 100_000
INTERP_INTP_POINTS = 500_000

# Sparse matrix size and density
N_SAMPLES = 10
SPARSE_N = 200_000
SPRAND_N = 200_000
SPRAND_DENSITY = 1.5e-4

# --- Helper functions ---
def laplacian_1d_py(n: int) -> sp.csc_matrix:
    """
    Builds a 1D Poisson (Laplacian) SPD sparse matrix.
    This matrix is tridiagonal with 2 on the main diagonal
    and -1 on the off-diagonals.
    
    Returns the matrix in CSC (Compressed Sparse Column) format,
    as this is the required format for CHOLMOD (scikit-sparse).
    """
    main = np.full(n, 2.0)
    off = np.full(n - 1, -1.0)
    
    # sp.diags is a convenient way to build sparse matrices from diagonals
    return sp.diags(
        [off, main, off], 
        [-1, 0, 1], 
        shape=(n, n), 
        format='csc' # Force CSC format
    )

def rosenbrock(x):
    """
    Rosenbrock function, a standard test problem in optimization.
    NumPy vectorized implementation.
    """
    return np.sum(1000.0 * (x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def lorenz(t, u, sigma, rho, beta):
    """
    Lorenz dynamical system definition for scipy.integrate.solve_ivp.
    y[0] = x, y[1] = y, y[2] = z
    """
    du = np.zeros(3)
    du[0] = sigma * (u[1] - u[0])
    du[1] = u[0] * (rho - u[2]) - u[1]
    du[2] = u[0] * u[1] - beta * u[2]
    return du

# laplacian_1d function removed

# --- Main benchmark runner ---
def run_benchmarks():
    # --- Environment info ---
    print("--- Python Scientific Computing Benchmark ---")
    print(f"Operating System: {platform.system()} ({platform.release()})")
    print(f"Python Version: {sys.version.split()[0]}")
    print(f"NumPy Version: {np.__version__}")
    print(f"SciPy Version: {scipy.__version__}")
    
    
    print("-" * 70)
    print(f"Dense Matrix Size: {MATRIX_SIZE}x{MATRIX_SIZE}")
    print(f"Vector Size: {VECTOR_SIZE}")
    print(f"Signal Size: {SIGNAL_SIZE}")
    print(f"Regression Points: {REGRESSION_POINTS}")
    print(f"Interpolation Base/Interp Points: {INTERP_BASE_POINTS} / {INTERP_INTP_POINTS}")
    # Sparse configuration prints removed
    print("-" * 70)
    
    # Dense data
    np.random.seed(42)
    mat_a = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    mat_b = np.random.rand(MATRIX_SIZE, MATRIX_SIZE)
    vec_a = np.random.rand(VECTOR_SIZE)
    vec_b = np.random.rand(VECTOR_SIZE)
    signal_data = np.random.randn(SIGNAL_SIZE)
    solve_vec = np.random.rand(MATRIX_SIZE)
    filter_b = np.random.rand(100)
    filter_a = np.random.rand(100)
    interp_x = np.linspace(0, 1000, INTERP_BASE_POINTS)
    interp_y = np.sin(interp_x)
    interp_points = np.linspace(0, 1000, INTERP_INTP_POINTS)

    A_spd_csc = laplacian_1d_py(SPARSE_N) # Generate in CSC format
    A_spd_csr = A_spd_csc.tocsr() # Create a CSR view (for efficient SpMV)
    b_spd = np.random.randn(SPARSE_N)

    # ODE (Lorenz) setup
    lorenz_u0 = [0.0, 1.0, 1.05]
    lorenz_tspan = (-1000.0, 1000.0) # Note: Julia version was (-5000, 5000)
    lorenz_p = (10.0, 28.0, 8/3) # sigma, rho, beta (passed as args)

    # Linear regression synthetic data: y = 3.5x + 2.0 + ε
    x_reg = np.linspace(0, 10, REGRESSION_POINTS)
    y_reg = 3.5 * x_reg + 2.0 + np.random.randn(REGRESSION_POINTS)
    # Design matrix [1 x]
    X_reg = np.vstack([np.ones(REGRESSION_POINTS), x_reg]).T 

    # Sparse data preparation removed
    
    # Initial guess for optimization
    optim_guess = np.random.randn(100)

    print("\n--- Benchmark Results (Average Time) ---")

    # Helper to run and print each benchmark result
    def run_and_print(name: str, benchmark_func):
        """
        Runs a function using timeit.repeat and prints the mean time.
        """
        # timeit.repeat runs the callable `benchmark_func`
        # `number=1` time for each "repeat"
        # `repeat=N_SAMPLES` times
        times = timeit.repeat(benchmark_func, number=1, repeat=N_SAMPLES)
        
        # Calculate mean and convert from seconds to milliseconds
        t_avg_ms = (np.mean(times) * 1000)
        
        print(f"{name:<55} {t_avg_ms:5.4f} ms")

    # --- Core language & dense linear algebra benchmarks ---
    run_and_print("NumPy: Elementwise Add", lambda: vec_a + vec_b)
    run_and_print("NumPy: Elementwise Mul", lambda: vec_a * vec_b)
    run_and_print("NumPy: Array Sorting (sort)", lambda: np.sort(vec_a))
    run_and_print("NumPy: Matrix Mul (@)", lambda: mat_a @ mat_b)
    run_and_print("SciPy (Linalg): Matrix Inv", lambda: la.inv(mat_a))
    run_and_print("SciPy (Linalg): Linear Solve (solve)", lambda: la.solve(mat_a, solve_vec))
    run_and_print("SciPy (Linalg): SVD", lambda: np.linalg.svd(mat_a))
    run_and_print("SciPy (Linalg): LU", lambda: la.lu(mat_a))
    run_and_print("SciPy (Linalg): QR", lambda: la.qr(mat_a))
    run_and_print("SciPy (FFT): FFT", lambda: fft.fft(signal_data))
    run_and_print("SciPy (Integrate): Numerical Integration (quad)", 
                  lambda: integrate.quad(lambda x: np.exp(-np.sqrt(x)) * np.sin(x**2), 0, 5000, limit=50000))
    run_and_print("SciPy (Optimize): Function Optimization (BFGS)", 
                  lambda: optimize.minimize(rosenbrock, optim_guess, method='BFGS'))
    run_and_print("SciPy (Signal): Signal Filtering (lfilter)", 
                  lambda: signal.lfilter(filter_b, filter_a, signal_data))
    run_and_print("SciPy (Interpolate): Cubic Interpolation", 
                  lambda: interpolate.CubicSpline(interp_x, interp_y)(interp_points))
    run_and_print("SciPy (Integrate): Lorenz System (RK45/solve_ivp)", 
                  lambda: integrate.solve_ivp(lorenz, lorenz_tspan, lorenz_u0, method='RK45', args=lorenz_p))
    run_and_print("NumPy (Linalg): Linear Regression (lstsq)", 
                  lambda: np.linalg.lstsq(X_reg, y_reg, rcond=None))
    run_and_print("Sparse (SPD Laplacian): SpMV (A*x)", 
                  lambda: A_spd_csr @ b_spd)
    run_and_print("Sparse (SPD Laplacian): Direct Solve (spsolve)", 
                  lambda: spsolve(A_spd_csc, b_spd))
    run_and_print("Sparse (SPD Laplacian): Cholesky Factorization", 
                  lambda: cholesky(A_spd_csc))
    
    print("-" * 70)

# --- Run all benchmarks ---
if __name__ == "__main__":
    run_benchmarks()