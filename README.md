# SciCompBench
A simple benchmark for scientific computing.

## Benchmark Results
### Macbook Pro 14-inch with M4Pro 12C
Julia v1.11.7 with 48G RAM on macOS 15.7.1 using AppleAccelerate:
```
Core: Elementwise Add                                   9.9466 ms
Core: Elementwise Mul                                   10.0754 ms
Core: Array Sorting (sort)                              138.7471 ms
LinearAlgebra (Dense): Matrix Mul (*)                   20.8690 ms
LinearAlgebra (Dense): Matrix Inv                       43.4287 ms
LinearAlgebra (Dense): Linear Solve (\)                 17.7204 ms
LinearAlgebra (Dense): SVD                              474.8078 ms
LinearAlgebra (Dense): LU                               16.6187 ms
LinearAlgebra (Dense): QR                               60.1234 ms
FFTW: FFT                                               209.9977 ms
QuadGK: Numerical Integration (quadgk)                  7.2921 ms
Optim: Function Optimization (BFGS)                     56.8914 ms
DSP: Signal Filtering (filt)                            274.9101 ms
Interpolations: Cubic Interpolation                     7.0348 ms
DifferentialEquations: Lorenz System (Tsit5/RK45)       44.1740 ms
Statistics: Linear Regression (OLS, \)                  23.8641 ms
Sparse (SPD Laplacian): SpMV (A*x)                      0.6197 ms
Sparse (SPD Laplacian): Direct Solve (\)                20.7969 ms
Sparse (SPD Laplacian): Cholesky Factorization          19.0360 ms
Sparse (sprand): SpMV (A*x)                             5.6554 ms
```

### Intel Xeon Gold 6248 80C
Julia v1.11.7 with 128G RAM on Ubuntu 18.04.6 LTS using MKL 40 threads
```
Core: Elementwise Add                                   102.8006 ms
Core: Elementwise Mul                                   157.9772 ms
Core: Array Sorting (sort)                              1010.6328 ms
LinearAlgebra (Dense): Matrix Mul (*)                   17.9823 ms
LinearAlgebra (Dense): Matrix Inv                       159.3643 ms
LinearAlgebra (Dense): Linear Solve (\)                 564.2903 ms
LinearAlgebra (Dense): SVD                              800.5115 ms
LinearAlgebra (Dense): LU                               33.1822 ms
LinearAlgebra (Dense): QR                               246.9505 ms
FFTW: FFT                                               947.2075 ms
QuadGK: Numerical Integration (quadgk)                  17.9191 ms
Optim: Function Optimization (BFGS)                     178.3238 ms
DSP: Signal Filtering (filt)                            594.9602 ms
Interpolations: Cubic Interpolation                     21.6988 ms
DifferentialEquations: Lorenz System (Tsit5/RK45)       133.6334 ms
Statistics: Linear Regression (OLS, \)                  127.0945 ms
Sparse (SPD Laplacian): SpMV (A*x)                      1.9484 ms
Sparse (SPD Laplacian): Direct Solve (\)                68.8523 ms
Sparse (SPD Laplacian): Cholesky Factorization          45.4432 ms
Sparse (sprand): SpMV (A*x)                             27.2457 ms
```

### AMD Ryzen 7 7840H 8C
Julia v1.11.7 with 64G RAM on Windows 11 using OpenBLAS 8 threads
```
Core: Elementwise Add                                   25.1144 ms
Core: Elementwise Mul                                   24.3024 ms
Core: Array Sorting (sort)                              286.5380 ms
LinearAlgebra (Dense): Matrix Mul (*)                   43.1021 ms
LinearAlgebra (Dense): Matrix Inv                       71.1630 ms
LinearAlgebra (Dense): Linear Solve (\)                 23.6477 ms
LinearAlgebra (Dense): SVD                              887.0305 ms
LinearAlgebra (Dense): LU                               21.9648 ms
LinearAlgebra (Dense): QR                               61.1964 ms
FFTW: FFT                                               668.3893 ms
QuadGK: Numerical Integration (quadgk)                  8.6644 ms
Optim: Function Optimization (BFGS)                     39.7787 ms
DSP: Signal Filtering (filt)                            313.0888 ms
Interpolations: Cubic Interpolation                     12.1900 ms
DifferentialEquations: Lorenz System (Tsit5/RK45)       81.5512 ms
Statistics: Linear Regression (OLS, \)                  38.9502 ms
Sparse (SPD Laplacian): SpMV (A*x)                      0.7430 ms
Sparse (SPD Laplacian): Direct Solve (\)                35.8809 ms
Sparse (SPD Laplacian): Cholesky Factorization          31.2228 ms
Sparse (sprand): SpMV (A*x)                             9.5694 ms
```