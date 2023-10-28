#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <assert.h>

#include <Python.h>  // plotting

#define EPSILON (1e-16)

// Lapack general (non-symmetric) eigenvalue/eigenvector function
extern void dgeev_(
    char   *JOBVL,
    char   *JOBVR,
    int    *N,
    double *A,
    int    *LDA,
    double *WR,
    double *WI,
    double *VL,
    int    *LDVL,
    double *VR,
    int    *LDVR,
    double *WORK,
    int    *LWORK,
    int    *INFO 
);
    
typedef double complex dcomplex;

// IN: coef = [a_0, a_1, ..., a_d], polynomial degree d, (complex) argument z
// OUT: \sum_k=0^d a_k * z^k
// For the time being, consider only polynomials with real-valued coefficients
dcomplex PolyEval(const double *coef, int degree, const dcomplex z)
{
    assert(degree >= 0 && "Expected non-negative degree");
    
    // Horner's method for polynomial evaluation
    dcomplex val = coef[degree];
    for (int i = degree - 1; i >= 0; --i)
    {
        val = val * z + coef[i];
    }
    
    return val;
}

// IN: coef = [a_0, a_1, ..., a_d], polynomial degree d, (complex) argument z
// OUT: \sum_k=1^d k * a_k * z^k-1
// For the time being, consider only polynomials with real-valued coefficients
dcomplex PolyDeriv(const double *coef, int degree, const dcomplex z)
{
    assert(degree >= 0 && "Expected non-negative degree");
    
    // Horner's method for polynomial derivative evaluation
    dcomplex deriv = degree * coef[degree];
    for (int i = degree - 1; i > 0; --i)
    {
        deriv = deriv * z + i * coef[i];
    }
    
    return deriv;
}

// IN: coef = [a_0, a_1, ..., a_d], polynomial degree d, (complex) argument z
// OUT: val = \sum_k=0^d a_k * z^k, deriv = \sum_k=1^d k * a_k * z^k-1
// For the time being, consider only polynomials with real-valued coefficients
void PolyEvalAndDeriv(const double *coef, int degree, 
    const dcomplex z, dcomplex *val, dcomplex *deriv)
{
    assert(degree >= 0 && "Expected non-negative degree");
    
    // Horner's method for polynomial evaluation
    *val = coef[degree];
    *deriv = degree * coef[degree];
    for (int i = degree - 1; i > 0; --i)
    {
        *val = (*val) * z + coef[i];
        *deriv = (*deriv) * z + i * coef[i];
    }
    // Final step for polynomial evaluation only
    if (degree > 0)
    {
        *val = (*val) * z + coef[0];
    }
}

// Issue: Different roots may erroneously converge to same root
dcomplex *PolyRoots(const double *coef, int degree, int maxIter)
{
    dcomplex *roots = (dcomplex *) malloc(degree * sizeof(dcomplex)); assert(roots);
    // Initialize (initial guesses)
    for (int i = 0; i < degree; ++i)
    {
        const double sign = (i % 2 == 0) ? 1.0 : -1.0;
        roots[i] = i + sign * I;
    }
    
    for (int i = 0; i < maxIter; ++i)
    {
        for (int j = 0; j < degree; ++j)
        {
            const dcomplex zj = roots[j];
            dcomplex dz = 0.0;
            
            for (int k = 0; k < degree && k != j; ++k)
            {
                dz += 1.0 / (zj - roots[k]);
            }

            // Evaluate polynomial as well as its derivative at current guess
            dcomplex val, deriv;
            PolyEvalAndDeriv(coef, degree, zj, &val, &deriv);
            
            // Newton update step
            dz = deriv / val - dz;
            roots[j] = zj - 1.0 / dz;
        }
    }
    
    return roots;
}

dcomplex *PolyRootsEV(const double *coef, int degree)
{
    assert(degree > 0 && "Expected positive degree");
    
    if (fabs(coef[degree]) < EPSILON)
    {
        fprintf(stderr, "Encountered too small coefficient of highest degree monomial\n");
        exit(EXIT_FAILURE);
    }
    
    dcomplex *roots = (dcomplex *) malloc(degree * sizeof(dcomplex)); assert(roots);
    
    double *A = (double *) calloc(degree * degree, sizeof(double)); assert(A);
    // Initialize companion matrix A (column-major storage order)
    const int lastCol = (degree - 1) * degree;
    for (int i = 0; i < degree; ++i)
    {
        if (i > 0) {
            // Lower diagonal entries: row = i, col = i - 1
            A[(i - 1) * degree + i] = 1.0;
        }
        // Entries in last column
        A[lastCol + i] = -coef[i] / coef[degree];  // -a_i / a_d = -b_i
    }
    
    // Note: Could create structure of arrays for real and imaginary parts
    //       of roots (stored separately)
    // Initialize arrays containing eigenvalues (real and imaginary parts)
    double *wr = (double *) malloc(degree * sizeof(double)); assert(wr);
    double *wi = (double *) malloc(degree * sizeof(double)); assert(wi);
    
    int lwork = -1;
    int one = 1;
    int info = 0;
    
    // Determine optimal size of work array (specified by lwork == -1)
    double workOpt = 0.0;  // will contain optimal size
    dgeev_("N", "N", &degree, A, &degree, wr, wi, NULL, &one, 
        NULL, &one, &workOpt, &lwork, &info);
    
    if (info != 0)
    {
        fprintf(stderr, "dgeev_ failed to determine optimal workspace size\n");
        exit(EXIT_FAILURE);
    }
    
    // Allocate work array
    lwork = (int)workOpt; assert(lwork > 0);
    double *work = (double *) malloc(lwork * sizeof(double)); assert(work);
    // Compute eigenvalues of companion matrix
    dgeev_("N", "N", &degree, A, &degree, wr, wi, NULL, &one, 
        NULL, &one, work, &lwork, &info);
    
    if (info != 0)
    {
        fprintf(stderr, "dgeev_ failed to compute the eigenvalues of A\n");
        exit(EXIT_FAILURE);
    }
    
    // Construct roots from eigenvalues stored as separate components
    for (int i = 0; i < degree; ++i)
    {
        roots[i] = wr[i] + wi[i] * I;
    }
    
    // Cleanup
    free(A);
    free(wr);
    free(wi);
    free(work);
    
    return roots;
}

void PolyPlotRoots(const dcomplex *roots, int degree, const char *programName)
{
    assert(degree > 0 && "Expected positive degree");
    
    // Initialize Python interpreter
    wchar_t *program = Py_DecodeLocale(programName, NULL);
    if (!program)
    {
        fprintf(stderr, "Failed to decode argv[0]\n");
        exit(EXIT_FAILURE);
    }
    Py_SetProgramName(program);
    Py_Initialize();  // starts Python interpreter
    
    PyObject *realList = PyList_New(degree); assert(realList);
    PyObject *imagList = PyList_New(degree); assert(imagList);
    
    // Add real and imaginary parts of roots to respective lists
    for (Py_ssize_t i = 0; i < degree; ++i)
    {
        if (PyList_SetItem(realList, i, PyFloat_FromDouble(creal(roots[i]))))
        {
            fprintf(stderr, "Failed to set item of realList\n");
            exit(EXIT_FAILURE);
        }
        if (PyList_SetItem(imagList, i, PyFloat_FromDouble(cimag(roots[i]))))
        {
            fprintf(stderr, "Failed to set item of realList\n");
            exit(EXIT_FAILURE);
        }
    }
    
    PyObject *locals = PyDict_New();
    PyObject *globals = PyDict_New();
    
    // Associate variables in python script with initialized python lists
    PyDict_SetItemString(locals, "real", realList);
    PyDict_SetItemString(locals, "imag", imagList);
    
    // Run python script with bound local variables
    PyRun_FileExFlags(
        fopen("../plot.py", "r"), 
        "plot.py",
        Py_file_input,
        globals,
        locals,
        1,
        NULL
    );
    
    if (Py_FinalizeEx() < 0)
    {
        fprintf(stderr, "Error occurred while running plot.py\n");
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        printf("Usage: ./polynomial a0 a1[ ...] (coefficients of polynomial)\n");
        return EXIT_SUCCESS;
    }
    
    const int size = argc - 1;
    const int degree = size - 1;
    
    // Put coefficients from commandline into array
    double *poly = (double *) malloc(size * sizeof(double)); assert(poly);
    for (int i = 1; i < argc; ++i)
    {
        poly[i - 1] = atof(argv[i]);
    }
    
    // Print polynomial
    printf("Polynomial entered:\n");
    printf("p(z) = %f", poly[0]);
    for (int i = 1; i < size; ++i)
    {
        const char sign = (poly[i] >= 0.0) ? '+' : '-';
        if (i > 1)
        {
            printf(" %c %fz^%d", sign, fabs(poly[i]), i);
        }
        else
        {
            printf(" %c %fz", sign, fabs(poly[i]));
        }
    }
    printf("\n");
    
    // Compute roots of polynomial
    dcomplex *roots = PolyRootsEV(poly, degree);
    
    printf("\nRoots:\n");
    for (int i = 0; i < degree; ++i)
    {
        if (fabs(cimag(roots[i])) < EPSILON)
        {
            // Only print real part
            printf("z_%d = %f, p(z_%d) = %e\n", i+1, 
                creal(roots[i]), i+1, 
                cabs(PolyEval(poly, degree, roots[i])));
        }
        else
        {
            const char sign = (cimag(roots[i]) >= 0.0) ? '+' : '-';
            printf("z_%d = %f %c %fi, p(z_%d) = %e\n", i+1, 
                creal(roots[i]), sign, fabs(cimag(roots[i])), 
                i+1, cabs(PolyEval(poly, degree, roots[i])));
        }
    }
    
    // Show a plot of the roots in the complex plane using pyplot
    PolyPlotRoots(roots, degree, argv[0]);
    
    // Cleanup
    free(poly);
    free(roots);
    
    return EXIT_SUCCESS;
}
