#include <R.h>
#include <cuda.h> 
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "cudadefine.h"

SEXP R_getRObjectPointer(SEXP r_obj)
{
	void *ptr = NULL;
	switch(TYPEOF(r_obj)) {
	case INTSXP:
		ptr = INTEGER(r_obj);
		break;
	case LGLSXP:
		ptr = LOGICAL(r_obj);
		break;
	case REALSXP:
		ptr = REAL(r_obj);
		break;
	case EXTPTRSXP:
		ptr = R_ExternalPtrAddr(r_obj);
		break;
	default:
		PROBLEM "unhandled case for getRObjectPointer"
			ERROR;
		break;
	}
	return(R_MakeExternalPtr(ptr, R_NilValue, R_NilValue));
}


//vector element-wise sum
extern void cuda_sum (double *, double *, double *, int);
SEXP vector_sum (SEXP ina, SEXP inb, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_sum (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inb),
		R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector element-wise subtract
extern void cuda_subtract (double *, double *, double *, int);
SEXP vector_subtract (SEXP ina, SEXP inb, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_subtract (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inb),
		R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector element-wise multiply
extern void cuda_multi (double *, double *, double *, int);
SEXP vector_multi (SEXP ina, SEXP inb, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_multi (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inb), 
		R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector element-wise divide
extern void cuda_divide (double *, double *, double *, int);
SEXP vector_divide (SEXP ina, SEXP inb, SEXP N) 
{

	int *n=INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_divide (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inb), 
		R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//vector element-wise exp
extern void cuda_exp (double *, double *, int);
SEXP vector_exp (SEXP ina, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_exp (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector element-wise log
extern void cuda_log (double *, double *, int);
SEXP vector_log (SEXP ina, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_log (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector element-wise square root
extern void cuda_sqrt (double *, double *, int);
SEXP vector_sqrt (SEXP ina, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_sqrt (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 


//vector element-wise gamma
extern void cuda_gamma (double *, double *, int);
SEXP vector_gamma (SEXP ina, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_gamma (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector element-wise beta
extern void cuda_beta (double *, double *, double *, int);
SEXP vector_beta (SEXP ina, SEXP inb, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_beta (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inb), 
                  R_ExternalPtrAddr(inc), *n);
	out=R_getRObjectPointer(inc);
	return(out);
} 





//vector element-wise vector power
extern void cuda_power (double *, double *, int, double);
SEXP vector_power (SEXP ina, SEXP inc, SEXP N, SEXP alpha) 
{
	SEXP out;
	int *n=INTEGER(N);
	double *vectorpow = REAL(alpha);
	cuda_power (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	*n, *vectorpow);
	out=R_getRObjectPointer(inc);
	return(out);
} 

//vector normal pdf function
extern void cuda_normal_density (double *,
	double *, int , double, double );
SEXP cudanormaldensity (SEXP ina, SEXP inc, SEXP N, SEXP m, SEXP s) 
{
	SEXP out;
	int *n=INTEGER(N);
	double *mean = REAL(m);
       double *sd = REAL(s);
	cuda_normal_density (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	*n, *mean, *sd);
	out=R_getRObjectPointer(inc);
	return(out);
} 


//vector normal CDF function
extern void cuda_normal_CDF (double *,
	double *, int );
SEXP cudanormalCDF (SEXP ina, SEXP inc, SEXP N) 
{
	SEXP out;
	int *n=INTEGER(N);
	cuda_normal_CDF (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	*n);
	out=R_getRObjectPointer(inc);
	return(out);
} 





//sample variance
extern void cudavariance (double *, double *, int, double);
SEXP cudavarGPU (SEXP ina, SEXP inc, SEXP N, SEXP m) 
{
	SEXP out;
	int *n=INTEGER(N);
	double *mean = REAL(m);
	cudavariance (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	*n, *mean);
	out=R_getRObjectPointer(inc);
	return(out);
} 



//vector reduction sum
extern double cuda_reduction (double *,int);
SEXP vector_reduction (SEXP ina, SEXP N) 
{
	SEXP out; 
	PROTECT(out = allocVector(REALSXP, 1));
	int *n=INTEGER(N);
	double *t = malloc(1 * sizeof(double));
	*t = cuda_reduction (R_ExternalPtrAddr(ina), *n);
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	return(out);
} 

//GPU device query

SEXP devicequery()
{
	SEXP out; 
	PROTECT(out = allocVector(REALSXP, 1));
	gpuquery();
	UNPROTECT(1); 
	return(out);
}	
