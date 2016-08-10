#include <R.h>  
#include <cuda.h> 
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include "cudadefine.h"
  
SEXP R_getRObjectPointer(SEXP r_obj)
{
	void *ptr = NULL;
	switch(TYPEOF(r_obj)) 
       {  
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

//vector element-wise divide
extern void cuda_divide(double *, double *, double *, int);
SEXP vector_divide(SEXP ina, SEXP inb, SEXP N) 
{
	int *n = INTEGER(N);
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
extern void cuda_exp(double *, double *, int);
SEXP vector_exp (SEXP ina, SEXP N) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_exp(R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//vector element-wise log
extern void cuda_log(double *, double *, int);
SEXP vector_log(SEXP ina, SEXP N) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_log(R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//vector element-wise square root
extern void cuda_sqrt(double *, double *, int);
SEXP vector_sqrt(SEXP ina, SEXP N) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_sqrt(R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//vector element-wise gamma
extern void cuda_gamma(double *, double *, int);
SEXP vector_gamma(SEXP ina, SEXP N) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_gamma(R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//vector element-wise beta
extern void cuda_beta(double *, double *, double *, int);
SEXP vector_beta(SEXP ina, SEXP inb, SEXP N) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_beta(R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inb), 
                 R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//vector element-wise vector power
extern void cuda_power(double *, double *, int, double);
SEXP vector_power(SEXP ina, SEXP N, SEXP alpha) 
{
	int *n = INTEGER(N);
	double *vectorpow = REAL(alpha);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_power(R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	    *n, *vectorpow);
	return(inc);
} 

//vector normal pdf function
extern void cuda_normal_density(double *, double *, int , double, double );
SEXP cudanormaldensity(SEXP ina, SEXP N, SEXP m, SEXP s) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	double *mean = REAL(m);
       double *sd = REAL(s);
	cuda_normal_density (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	              *n, *mean, *sd);
	return(inc);
} 

//vector normal CDF function
extern void cuda_normal_CDF(double *, double *, int );
SEXP cudanormalCDF(SEXP ina, SEXP N) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	cuda_normal_CDF (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), *n);
	return(inc);
} 

//sample variance
extern void cudavariance(double *, double *, int, double);
SEXP cudavarGPU(SEXP ina,SEXP N, SEXP m) 
{
	int *n = INTEGER(N);
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	double *mean = REAL(m);
	cudavariance (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	       *n, *mean);
	return(inc);
} 

//this function returns the specified subset of given vector
extern void vector_subset(double *a, double *c, int n, int *index);
SEXP subset_GPU(SEXP ina, SEXP N, SEXP sub) 
{
	int *n = INTEGER(N);
	int *indexcpu = INTEGER(sub);
       int *index;
	cudacall(cudaMalloc((void**)&index, *n * sizeof(int)));
       cudacall(cudaMemcpy(index, indexcpu, sizeof(int) * (*n), 
                cudaMemcpyHostToDevice));
	double *x;
	cudaMalloc((void**)&x, *n * sizeof(double));
	//protect the R external pointer from finalizer
	SEXP inc = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(inc, _finalizer, TRUE);
       UNPROTECT(1);
	vector_subset (R_ExternalPtrAddr(ina), R_ExternalPtrAddr(inc), 
       	        *n, index);     
	return(inc);
} 

/*
define function to generate gamma distributied random
number, input is length of vector, k, theta
and seed, output is pointer pointing to a GPU vector(device)
*/
extern void gammarng(double a, double b, int n, double seed, double* number);
SEXP gammaRNGGPU(SEXP n, SEXP alpha, SEXP beta, SEXP seed) 
{
	int *lenthN = INTEGER(n);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *s = REAL(seed);
	double *x;
	cudacall(cudaMalloc((void**)&x, (*lenthN) * sizeof(double)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	gammarng(*a, *b, *lenthN, *s, R_ExternalPtrAddr(out));
	UNPROTECT(1);    
	return out;
} 

/*
define function to generate beta distributied random
number, input is length of vector, alpha, beta
and seed, output is pointer pointing to a GPU vector(device)
*/
extern void betarng(double a, double b, int n, double seed, double* number);
SEXP betaRNGGPU(SEXP n, SEXP alpha, SEXP beta, SEXP seed) 
{
	int *lenthN = INTEGER(n);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *s = REAL(seed);
	double *x;
	cudacall(cudaMalloc((void**)&x, (*lenthN) * sizeof(double)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	betarng(*a, *b, *lenthN, *s, R_ExternalPtrAddr(out));
	UNPROTECT(1);    
	return out;
} 

//this function compute the summation of given vector
extern double cuda_reduction(double *, int);
SEXP vector_reduction(SEXP ina, SEXP N) 
{
	SEXP out; 
	PROTECT(out = allocVector(REALSXP, 1));
	int *n = INTEGER(N);
	double *t = malloc(1 * sizeof(double));
	*t = cuda_reduction (R_ExternalPtrAddr(ina), *n);
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	return(out);
} 

//this function returns the information of GPU device
SEXP devicequery()
{
	SEXP out; 
	PROTECT(out = allocVector(REALSXP, 1));
	gpuquery();
	UNPROTECT(1); 
	return(out);
}	
