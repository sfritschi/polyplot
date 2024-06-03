# polyplot
Compute and plot roots of polynomial in complex plane from coefficients entered as command line arguments.
Uses DGEEV to find the eigenvalues of the companion matrix, which coincide
with the roots of the given polynomial. The roots are then plotted by
passing them to Python script invoked by the C code directly.  

# Dependencies
- CMake >= 3.0
- LAPACK & BLAS
- Python 3 (Interpreter & Development components)
    - matplotlib
    - numpy
