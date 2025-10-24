# Import required packages
using Pkg
Pkg.activate(".")

using BenchmarkTools
using LinearAlgebra
using SparseArrays
using FFTW
using QuadGK
using Optim
using DSP
using Interpolations
using Printf
using Random
using DifferentialEquations # for ODE solving
using IterativeSolvers      # for Krylov/iterative sparse solvers

# using MKL  # Uncomment if you want to include MKL backend for Intel CPU
# Use AppleAccelerate on Apple Silicon if available (guarded import)
if Sys.isapple()
    try
        @eval using AppleAccelerate
    catch
        @warn "AppleAccelerate not found; continuing without it."
    end
end

# --- Benchmark configuration ---
# Number of samples per benchmark
const N_SAMPLES = 10
# Dense matrix size (N x N)
const MATRIX_SIZE = 2000
# Vector/array size
const VECTOR_SIZE = 20_000_000
# Signal length for FFT and DSP
const SIGNAL_SIZE = 20_000_000
# Number of data points for linear regression test
const REGRESSION_POINTS = 5_000_000
# Interpolation points
const INTERP_BASE_POINTS = 100_000
const INTERP_INTP_POINTS = 500_000

# Sparse problem sizes
# 1D Poisson (tridiagonal) matrix of order N has ~3N nonzeros -> scalable and memory friendly
const SPARSE_N = 200_000         # order of the sparse system (adjust if you run out of memory)
const SPRAND_N = 200_000         # order for a random sparse matrix
const SPRAND_DENSITY = 1.5e-4    # density for sprand (approx nnz ≈ density*N^2)

# --- Helper functions ---

# Rosenbrock function is a standard test problem in optimization.
function rosenbrock(x::Vector)
    n = length(x)
    return sum(1000 * (x[i+1] - x[i]^2)^2 + (1 - x[i])^2 for i in 1:n-1)
end

# Lorenz dynamical system definition (classic nonlinear ODE system).
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Build a 1D Poisson (Laplacian) SPD sparse matrix with Dirichlet boundary conditions.
# A is tridiagonal with 2 on the main diagonal and -1 on the off-diagonals.
function laplacian_1d(n::Int)
    main  = fill(2.0, n)
    off   = fill(-1.0, n-1)
    return spdiagm(-1 => off, 0 => main, 1 => off)
end

# --- Main benchmark runner ---
function run_benchmarks()
    # --- Environment info ---
    println("--- Julia Scientific Computing Benchmark ---")
    println("Operating System: ", Sys.iswindows() ? "Windows" : Sys.isapple() ? "macOS" : Sys.islinux() ? "Linux" : "Unknown")
    println("Julia Version: ", VERSION)
    println("-" ^ 70)
    println("Dense Matrix Size: ", MATRIX_SIZE, "x", MATRIX_SIZE)
    println("Vector Size: ", VECTOR_SIZE)
    println("Signal Size: ", SIGNAL_SIZE)
    println("Regression Points: ", REGRESSION_POINTS)
    println("Interpolation Base/Interp Points: ", INTERP_BASE_POINTS, " / ", INTERP_INTP_POINTS)
    println("Sparse Poisson Order: ", SPARSE_N, " (nnz ≈ ", 3 * SPARSE_N, ")")
    println("sprand Size/Density: ", SPRAND_N, " / ", SPRAND_DENSITY)
    println("-" ^ 70)

    # --- Prepare test data ---
    Random.seed!(42)
    # Dense data
    mat_a = rand(Float64, MATRIX_SIZE, MATRIX_SIZE)
    mat_b = rand(Float64, MATRIX_SIZE, MATRIX_SIZE)
    vec_a = rand(Float64, VECTOR_SIZE)
    vec_b = rand(Float64, VECTOR_SIZE)
    signal_data = randn(Float64, SIGNAL_SIZE)
    solve_vec = rand(Float64, MATRIX_SIZE)
    filter_b = rand(Float64, 100)
    filter_a = rand(Float64, 100)
    interp_x = range(0, 1000, length=INTERP_BASE_POINTS)
    interp_y = sin.(interp_x)
    interp_points = range(0, 1000, length=INTERP_INTP_POINTS)

    # ODE (Lorenz) setup
    lorenz_u0 = [0.0, 1.0, 1.05]
    lorenz_tspan = (-5000.0, 5000.0)
    lorenz_p = (10.0, 28.0, 8/3) # standard sigma, rho, beta
    lorenz_prob = ODEProblem(lorenz!, lorenz_u0, lorenz_tspan, lorenz_p)

    # Linear regression synthetic data: y = 3.5x + 2.0 + ε
    x_reg = range(0, 10, length=REGRESSION_POINTS)
    y_reg = 3.5 .* x_reg .+ 2.0 .+ randn(REGRESSION_POINTS)
    X_reg = [ones(REGRESSION_POINTS) x_reg] # design matrix [1 x]

    # Sparse data
    # Structured SPD Laplacian (good for CG/Cholesky)
    A_spd = laplacian_1d(SPARSE_N)
    b_spd = randn(SPARSE_N)

    # Simple Jacobi (diagonal) preconditioner M ≈ A
    # For IterativeSolvers.cg, we pass M as a linear operator applying inv(D)*x
    Dinv = Diagonal(1.0 ./ diag(A_spd))
    M_jacobi = (x -> Dinv * x)

    # Random sparse matrix (not guaranteed SPD)
    # Add small diagonal for numerical stability, avoid exact singularity
    A_rand = sprand(SPRAND_N, SPRAND_N, SPRAND_DENSITY)
    A_rand += spdiagm(0 => 1e-2 .* ones(SPRAND_N))
    b_rand = randn(SPRAND_N)

    println("\n--- Benchmark Results (Average Time) ---")

    # Helper to run and print each benchmark result
    function run_and_print(name::String, benchmark_func::Function)
        BenchmarkTools.DEFAULT_PARAMETERS.samples = N_SAMPLES
        b = @benchmark ($benchmark_func)()
        t = mean(b).time / 1e6 # convert ns to ms
        @printf("%-55s %5.4f ms\n", name, t)
    end

    # --- Core language & dense linear algebra benchmarks ---
    run_and_print("Core: Elementwise Add", () -> vec_a .+ vec_b)
    run_and_print("Core: Elementwise Mul", () -> vec_a .* vec_b)
    run_and_print("Core: Array Sorting (sort)", () -> sort(vec_a))
    run_and_print("LinearAlgebra (Dense): Matrix Mul (*)", () -> mat_a * mat_b)
    run_and_print("LinearAlgebra (Dense): Matrix Inv", () -> inv(mat_a))
    run_and_print("LinearAlgebra (Dense): Linear Solve (\\)", () -> mat_a \ solve_vec)
    run_and_print("LinearAlgebra (Dense): SVD", () -> svd(mat_a))
    run_and_print("LinearAlgebra (Dense): LU", () -> lu(mat_a))
    run_and_print("LinearAlgebra (Dense): QR", () -> qr(mat_a))
    run_and_print("FFTW: FFT", () -> fft(signal_data))
    run_and_print("QuadGK: Numerical Integration (quadgk)", () -> quadgk(x -> exp(-sqrt(x)) * sin(x^2), 0, 5000))
    run_and_print("Optim: Function Optimization (BFGS)", () -> optimize(rosenbrock, randn(100), BFGS()))
    run_and_print("DSP: Signal Filtering (filt)", () -> filt(filter_b, filter_a, signal_data))
    run_and_print("Interpolations: Cubic Interpolation", () -> CubicSplineInterpolation(interp_x, interp_y).(interp_points))
    run_and_print("DifferentialEquations: Lorenz System (Tsit5/RK45)", () -> DifferentialEquations.solve(lorenz_prob, Tsit5()))
    run_and_print("Statistics: Linear Regression (OLS, \\)", () -> X_reg \ y_reg)

    # --- Sparse linear algebra benchmarks ---
    # Structured SPD Laplacian: SpMV
    run_and_print("Sparse (SPD Laplacian): SpMV (A*x)", () -> A_spd * b_spd)
    # Structured SPD Laplacian: direct solve via \
    run_and_print("Sparse (SPD Laplacian): Direct Solve (\\)", () -> A_spd \ b_spd)
    # Structured SPD Laplacian: Cholesky factorization (SPD)
    run_and_print("Sparse (SPD Laplacian): Cholesky Factorization", () -> cholesky(A_spd))
    # Random sparse (nonsymmetric/indefinite likely): SpMV
    run_and_print("Sparse (sprand): SpMV (A*x)", () -> A_rand * b_rand)

    println("-" ^ 70)
end

# --- Run all benchmarks ---
run_benchmarks()