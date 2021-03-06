#include "cudadefine.h"
/*
This file contains all R interfaces of CUBLAS and CURAND functions
*/
  
/*
define function to create a vector in GPU 
by transfering a R's vector to GPU.
input is R's vector and its length, 
output is a R external pointer
pointing to GPU vector(device)
*/
SEXP createGPU(SEXP input, SEXP n)
{  
	int *lenth = INTEGER(n);
       PROTECT (input = AS_NUMERIC (input));
       double * temp; 
       temp = REAL(input);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenth * sizeof(double)));
	//protect the R external pointer from finalizer
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	//copying data from CPU to GPU
       cudacall(cudaMemcpy(x, temp, sizeof(double)*(*lenth), 
                cudaMemcpyHostToDevice));
       UNPROTECT(2);
	return ext;
}

/*
define function to copy a GPU vector back to R.
input is R external pointer pointing to GPU vector, 
output is a R vector(host).
*/
SEXP gatherGPU(SEXP ext, SEXP n)
{ 
	checkExternalPrt(ext);
	int *lenth = INTEGER(n);
	SEXP out;
	PROTECT(out = NEW_NUMERIC(*lenth)); 
	//copying data from GPU back to CPU
       cudacall(cudaMemcpy(NUMERIC_POINTER(out), R_ExternalPtrAddr(ext),
                sizeof(double)*(*lenth), cudaMemcpyDeviceToHost));  
	UNPROTECT(1); 
	return out;  
}

/*CULBLAS level 1 functions*/
/*
define function to find the (smallest) 
index of the element of the minimum magnitude
input is a R's external pointer pointing to GPU vector
and its length, output is the index(host or device) 
*/
SEXP minGPU(SEXP ext, SEXP n)
{
	checkExternalPrt(ext);
	//initialize CUBLAS
	cublasHandle_t handle;
	//create CUBLAS handle
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	//the cublasIdamin only takes pointer as the output argument
	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	int *t = malloc(1 * sizeof(int));
	cublascall(cublasIdamin(handle, *lenth, R_ExternalPtrAddr(ext), 1, t));
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	cublascall(cublasDestroy_v2(handle));
	return out;
}

/*
define function to find the (smallest) 
index of the element of the maximum magnitude
input is a R's external pointer pointing to GPU vector
and its length, output is the index(host or device) 
*/
SEXP maxGPU(SEXP ext, SEXP n)
{
	checkExternalPrt(ext);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	int *t = malloc(1 * sizeof(int));
	cublascall(cublasIdamax(handle, *lenth, R_ExternalPtrAddr(ext), 1, t));
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	cublascall(cublasDestroy_v2(handle));
	return out;
}


/*
define function to compute the summation 
of the absolute values of elements of given vector/matrix
input is a R's external pointer pointing to GPU vector
and its length, output is the summation(host or device) 
*/
SEXP asumGPU(SEXP ext, SEXP n)
{
	checkExternalPrt(ext);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	double *t = malloc(1 * sizeof(double));
	cublascall(cublasDasum(handle, *lenth, R_ExternalPtrAddr(ext), 1, t));
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	cublascall(cublasDestroy_v2(handle));
	return out;
}


/*
define function to multiply the vector x by the scalar a and adds it to 
the vector y overwriting the latest vector with the result 
input is a R's external pointer pointing to GPU vector
and its length, output is the summation(host or device) 
*/
SEXP axpyGPU(SEXP ext1, SEXP ext2, SEXP n, SEXP alpha)
{
	checkExternalPrt(ext1);
	checkExternalPrt(ext2);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	double *a = REAL(alpha);
	cublascall(cublasDaxpy(handle, *lenth, a, R_ExternalPtrAddr(ext1),
                  1, R_ExternalPtrAddr(ext2), 1));
	cublascall(cublasDestroy_v2(handle));
	return ext2;
}


/*
define function to copy the vector x into the vector y 
input is a R's external pointer pointing to GPU vector
and its length, output is the summation(host or device) 
*/
SEXP copyGPU(SEXP ext1, SEXP ext2, SEXP n)
{
	checkExternalPrt(ext1);
	checkExternalPrt(ext2);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	cublascall(cublasDcopy(handle, *lenth, R_ExternalPtrAddr(ext1),
                  1, R_ExternalPtrAddr(ext2), 1));
	cublascall(cublasDestroy_v2(handle));
	return ext2;
}


/*
define function to scale the vector x by the scalar a 
and overwrites it with the result 
input is a R's external pointer pointing to GPU vector
and its length, output is the summation(host or device) 
*/
SEXP scalGPU(SEXP ext, SEXP n, SEXP alpha)
{
	checkExternalPrt(ext);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	double *a = REAL(alpha);
	cublascall(cublasDscal(handle, *lenth, a, R_ExternalPtrAddr(ext), 1));
	cublascall(cublasDestroy_v2(handle));
	return ext;
}


/*
define function to calculate the Euclidean norm of vector
input is a R's external pointer pointing to GPU vector
and its length, output is the Euclidean norm (host or device) 
*/
SEXP norm2GPU(SEXP ext, SEXP n)
{
	checkExternalPrt(ext);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenth = INTEGER(n);
	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	double *t = malloc(1 * sizeof(double));
	cublascall(cublasDnrm2(handle, *lenth, R_ExternalPtrAddr(ext), 1, t));
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	cublascall(cublasDestroy_v2(handle));
	return out;
}

/*
define function to calculate the dot product of 2 vectors
input is 2 R's external pointers pointing to 2 GPU vectors
and their lengths, output is dotproduct(host or device) 
*/
SEXP dotGPU(SEXP extV1, SEXP extV2, SEXP n)
{
	checkExternalPrt(extV1);
	checkExternalPrt(extV2);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	SEXP out;
	PROTECT(out = allocVector(REALSXP, 1));
	double *t = malloc(1 * sizeof(double));
	cublascall(cublasDdot (handle, *lenthN, R_ExternalPtrAddr(extV1),
		    1, R_ExternalPtrAddr(extV2), 1, t));
	REAL(out)[0] = *t;
	free(t);
	UNPROTECT(1); 
	cublascall(cublasDestroy_v2(handle));
	return out; 
}

/*
define function to scale one vector
input is a R's external pointer pointing to a GPU vectors
,its length and the scale scalar,
output is a R's external pointer pointing to the result GPU vectors(device)
*/
SEXP scaleGPU(SEXP input, SEXP n, SEXP alpha)
{  
	checkExternalPrt(input);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));   
	int *lenthN = INTEGER(n);
	double *a = REAL(alpha);  
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthN * sizeof(double)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(out, _finalizer, TRUE);
	cublascall(cublasDcopy(handle, *lenthN, R_ExternalPtrAddr(input), 
		    1, R_ExternalPtrAddr(out), 1));
	cublascall(cublasDscal(handle, *lenthN, a, R_ExternalPtrAddr(out), 1));
	UNPROTECT(1); 
	cublascall(cublasDestroy_v2(handle));
	return out;
}

/*CULBLAS level 2 functions*/

/*
define function to perform the matrix vector multiplication
y = a op ( A ) x + � y where A is a m � n matrix stored in column-major format,
x and y are vectors, and a and � are scalars. 
*/

SEXP gemvGPU(SEXP extA, SEXP extx, SEXP exty, SEXP alpha, 
             SEXP beta, SEXP m, SEXP n, SEXP trans)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(exty);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM=INTEGER(m);
	int *lenthN=INTEGER(n);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *t = REAL(trans);
	cublasOperation_t transa;
	transa = cublasop(*t);
	cublascall(cublasDgemv(handle, transa, *lenthM, *lenthN, a, 
		    R_ExternalPtrAddr(extA), *lenthM, 
		    R_ExternalPtrAddr(extx), 1, b, 
		    R_ExternalPtrAddr(exty), 1));
	cublascall(cublasDestroy_v2(handle));
	return(exty); 
}


/*
define function to perform the banded matrix vector multiplication
y = a op ( A ) x + � y where A is a banded m � n matrix stored in 
column-major format, x and y are vectors, and a and � are scalars. 
*/

SEXP gbmvGPU(SEXP extA, SEXP extx, SEXP exty, SEXP alpha, 
             SEXP beta, SEXP m, SEXP n, SEXP kl, SEXP ku, SEXP trans)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(exty);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM=INTEGER(m);
	int *lenthN=INTEGER(n);
	int *bkl=INTEGER(kl);
	int *bku=INTEGER(ku);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *t = REAL(trans);
	cublasOperation_t transa;
	transa = cublasop(*t);
	cublascall(cublasDgbmv(handle, transa, *lenthM, *lenthN, 
                  *bkl, *bku, a, R_ExternalPtrAddr(extA), *lenthM, 
		    R_ExternalPtrAddr(extx), 1, b, 
		    R_ExternalPtrAddr(exty), 1));
	cublascall(cublasDestroy_v2(handle));
	return(exty); 
}


/*
define function to perform the symmetric banded matrix vector multiplication
y = a op ( A ) x + � y where A is a symmetric banded n � n matrix stored in 
column-major format, x and y are vectors, and a and � are scalars. 
*/

SEXP sbmvGPU(SEXP extA, SEXP extx, SEXP exty, SEXP alpha, 
             SEXP beta, SEXP n, SEXP k, SEXP fillmod)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(exty);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *t = REAL(fillmod);
	int *bk=INTEGER(k);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*t);
	cublascall(cublasDsbmv(handle, fillmode, *lenthN, *bk, 
                  a, R_ExternalPtrAddr(extA), *lenthN, 
		    R_ExternalPtrAddr(extx), 1, b, 
		    R_ExternalPtrAddr(exty), 1));
	cublascall(cublasDestroy_v2(handle));
	return(exty); 
}


/*
define function to perform the symmetric matrix vector multiplication
y = a op ( A ) x + � y where A is a symmetric n � n matrix stored in 
column-major format, x and y are vectors, and a and � are scalars. 
*/

SEXP symvGPU(SEXP extA, SEXP extx, SEXP exty, SEXP alpha, 
             SEXP beta, SEXP n, SEXP fillmod)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(exty);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *t = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*t);
	cublascall(cublasDsymv(handle, fillmode, *lenthN, 
                  a, R_ExternalPtrAddr(extA), *lenthN, 
		    R_ExternalPtrAddr(extx), 1, b, 
		    R_ExternalPtrAddr(exty), 1));
	cublascall(cublasDestroy_v2(handle));
	return(exty); 
}


/*
define function to perform the the rank-1 update A = a x y T + A,
where A is a m � n matrix stored in column-major format, 
x and y are vectors, and a is a scalar
*/

SEXP gerGPU(SEXP extA, SEXP extx, SEXP exty, SEXP alpha, 
             SEXP m, SEXP n)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(exty);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM=INTEGER(m);
	int *lenthN=INTEGER(n);
	double *a = REAL(alpha);
	cublascall(cublasDger(handle, *lenthM, *lenthN, a,  
		    R_ExternalPtrAddr(extx), 1,  
		    R_ExternalPtrAddr(exty), 1,
		    R_ExternalPtrAddr(extA), *lenthM));
	cublascall(cublasDestroy_v2(handle));
	return(extA); 
}


/*
define function to perform the the symmetric rank-1 update A = a x x T + A,
where A is a n � n matrix stored in column-major format, 
x is vector, and a is a scalar
*/

SEXP syrGPU(SEXP extA, SEXP extx, SEXP alpha, 
             SEXP n, SEXP fillmod)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *t = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*t);
	double *a = REAL(alpha);
	cublascall(cublasDsyr(handle, fillmode, *lenthN, a,  
		    R_ExternalPtrAddr(extx), 1,  
		    R_ExternalPtrAddr(extA), *lenthN));
	cublascall(cublasDestroy_v2(handle));
	return(extA); 
}


/*
define function to perform the the symmetric rank-2 update 
A = a x y T + x y T + A, where A is a n � n matrix stored 
in column-major format, x and y are vectors, and a is a scalar
*/

SEXP syr2GPU(SEXP extA, SEXP extx, SEXP exty, SEXP alpha, 
             SEXP n, SEXP fillmod)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(exty);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *t = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*t);
	double *a = REAL(alpha);
	cublascall(cublasDsyr2(handle, fillmode, *lenthN, a,  
		    R_ExternalPtrAddr(extx), 1,  
		    R_ExternalPtrAddr(exty), 1,
		    R_ExternalPtrAddr(extA), *lenthN));
	cublascall(cublasDestroy_v2(handle));
	return(extA); 
}


/*
define function to perform the triangular banded matrix vector multiplication
x =  op ( A ) x  where A is a triangular banded n � n matrix stored in 
column-major format, x is a vector.
*/

SEXP tbmvGPU(SEXP extA, SEXP extx, SEXP n, SEXP k, SEXP fillmod,
             SEXP trans, SEXP diag)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	int *bk=INTEGER(k);
	double *t = REAL(trans);
	double *f = REAL(fillmod);
	double *d = REAL(diag);
	cublasOperation_t transa;
	transa = cublasop(*t);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	cublasDiagType_t diagmode;
	diagmode = cublasdiagtype(*d);
	cublascall(cublasDtbmv(handle, fillmode, transa, diagmode, *lenthN, *bk, 
                  R_ExternalPtrAddr(extA), *lenthN, 
		    R_ExternalPtrAddr(extx), 1));
	cublascall(cublasDestroy_v2(handle));
	return(extx); 
}


/*
define function to solve the triangular banded linear system
 with a single right-hand-side
op ( A ) x = b,  where A is a triangular banded n � n matrix stored in 
column-major format, x and b are vectors.
*/

SEXP tbsvGPU(SEXP extA, SEXP extx, SEXP n, SEXP k, SEXP fillmod,
             SEXP trans, SEXP diag)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	int *bk=INTEGER(k);
	double *t = REAL(trans);
	double *f = REAL(fillmod);
	double *d = REAL(diag);
	cublasOperation_t transa;
	transa = cublasop(*t);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	cublasDiagType_t diagmode;
	diagmode = cublasdiagtype(*d);
	cublascall(cublasDtbsv(handle, fillmode, transa, diagmode, *lenthN, *bk, 
                  R_ExternalPtrAddr(extA), *lenthN, 
		    R_ExternalPtrAddr(extx), 1));
	cublascall(cublasDestroy_v2(handle));
	return(extx); 
}


/*
define function to perform triangular matrix-vector multiplication
x = op ( A ) x,  where A is a triangular n � n matrix stored in 
column-major format, x is vector.
*/

SEXP trmvGPU(SEXP extA, SEXP extx, SEXP n, SEXP fillmod,
             SEXP trans, SEXP diag)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *t = REAL(trans);
	double *f = REAL(fillmod);
	double *d = REAL(diag);
	cublasOperation_t transa;
	transa = cublasop(*t);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	cublasDiagType_t diagmode;
	diagmode = cublasdiagtype(*d);
	cublascall(cublasDtrmv(handle, fillmode, transa, diagmode, *lenthN, 
                  R_ExternalPtrAddr(extA), *lenthN, 
		    R_ExternalPtrAddr(extx), 1));
	cublascall(cublasDestroy_v2(handle));
	return(extx); 
}


/*
define function to solve the triangular linear system
with a single right-hand-side
op ( A ) x = b,  where A is a triangular n � n matrix stored in 
column-major format, x and b are vectors.
*/

SEXP trsvGPU(SEXP extA, SEXP extx, SEXP n, SEXP fillmod,
             SEXP trans, SEXP diag)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *t = REAL(trans);
	double *f = REAL(fillmod);
	double *d = REAL(diag);
	cublasOperation_t transa;
	transa = cublasop(*t);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	cublasDiagType_t diagmode;
	diagmode = cublasdiagtype(*d);
	cublascall(cublasDtrsv(handle, fillmode, transa, diagmode, *lenthN, 
                  R_ExternalPtrAddr(extA), *lenthN, 
		    R_ExternalPtrAddr(extx), 1));
	cublascall(cublasDestroy_v2(handle));
	return(extx); 
}


/*
define function to calculate the element-wise addition
of 2 vectors. Input is 2 R's external pointers pointing to 2 GPU vectors
and their lengths, output is R's external pointer pointing to GPU
vector(device)
*/
SEXP addGPU(SEXP extM1, SEXP extM2, SEXP m, SEXP n)
{
	checkExternalPrt(extM1);
	checkExternalPrt(extM2);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM = INTEGER(m);
	int *lenthN = INTEGER(n);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthM * (*lenthN) * sizeof(double)));
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	UNPROTECT(1);
	double *alpha = malloc(1 * sizeof(double));
	alpha[0] = 1;
	double *beta = malloc(1 * sizeof(double));
	beta[0] = 1;	
	cublascall(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, *lenthM, 
                  *lenthN, alpha, R_ExternalPtrAddr(extM1),
		    *lenthM, beta, R_ExternalPtrAddr(extM2),*lenthM,
		    R_ExternalPtrAddr(ext), *lenthM));
	cublascall(cublasDestroy_v2(handle));
	return(ext);
}

/*
define function to calculate the element-wise subtraction
of 2 vectors. Input is 2 R's external pointers pointing to 2 GPU vectors
and their lengths, output is R's external pointer pointing to GPU
vector(device)
*/
SEXP subtractGPU(SEXP extM1, SEXP extM2, SEXP m, SEXP n)
{
	checkExternalPrt(extM1);
	checkExternalPrt(extM2);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM = INTEGER(m);
	int *lenthN = INTEGER(n);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthM * (*lenthN) * sizeof(double)));
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	UNPROTECT(1);
	double *alpha = malloc(1 * sizeof(double));
	alpha[0] = 1;
	double *beta = malloc(1 * sizeof(double));
	beta[0] = -1;	
	cublascall(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, *lenthM, 
		    *lenthN,alpha, R_ExternalPtrAddr(extM1),
		    *lenthM, beta, R_ExternalPtrAddr(extM2),*lenthM,
		    R_ExternalPtrAddr(ext), *lenthM));
	cublascall(cublasDestroy_v2(handle));
	return(ext);
}

/*
define function to calculate the element-wise multiplication
of 2 vectors. Input is 2 R's external pointers pointing to 2 GPU vectors
and their lengths, output is R's external pointer pointing to GPU
vector(device)
*/
SEXP vvGPU(SEXP extM, SEXP extV, SEXP n)
{
	checkExternalPrt(extM);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthN=INTEGER(n);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthN * sizeof(double)));
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	UNPROTECT(1);
	cublascall(cublasDdgmm(handle, CUBLAS_SIDE_RIGHT, 1, *lenthN, 
		    R_ExternalPtrAddr(extM), 1, 
		    R_ExternalPtrAddr(extV), 1, 
		    R_ExternalPtrAddr(ext), 1));
	return(ext); 
}

/*
define function to calculate the matrix times vector
Input is 2 R's external pointers pointing to 2 GPU vectors
and dimension of the matrix and vector, 
output is R's external pointer pointing to GPU
vector(device)
*/
SEXP mvGPU(SEXP extM, SEXP extV, SEXP m, SEXP n)
{
	checkExternalPrt(extM);
	checkExternalPrt(extV);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM=INTEGER(m);
	int *lenthN=INTEGER(n);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthM * sizeof(double)));
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	UNPROTECT(1);
	//alpha and beta here is the scale factor, for regular multiplication
	//we set alpha and beta to be 1 and 0 repectively.
	double *alpha = malloc(1 * sizeof(double));
	alpha[0] = 1;
	double *beta = malloc(1 * sizeof(double));
	beta[0] = 0;
	cublascall(cublasDgemv(handle, CUBLAS_OP_N, *lenthM, *lenthN, alpha, 
		    R_ExternalPtrAddr(extM), *lenthM, 
		    R_ExternalPtrAddr(extV), 1, beta, 
		    R_ExternalPtrAddr(ext), 1));
	cublascall(cublasDestroy_v2(handle));
	return(ext); 
}


/*CULBLAS level 3 functions*/


/*
define function to calculate the matrix-matrix multiplication
C = a op ( A ) op ( B ) + b C where a and b are scalars, 
and A , B and C are matrices stored in column-major format 
with dimensions op ( A ) m � k , op ( B ) k � n and C m � n 
, respectively.
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP gemmGPU(SEXP extA, SEXP extB, SEXP extC, SEXP lda, SEXP ldb, SEXP ldc,
             SEXP m, SEXP n, SEXP k, SEXP transA, SEXP transB,
             SEXP alpha, SEXP beta)
{
	checkExternalPrt(extA);
	checkExternalPrt(extB);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldB = INTEGER(ldb);
	int *ldC = INTEGER(ldc);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *tA = REAL(transA);
	double *tB = REAL(transB);
	int *rA = INTEGER(m);
	int *cB = INTEGER(n);
	int *cA = INTEGER(k);     
	cublasOperation_t transa;
	cublasOperation_t transb;
	transa = cublasop(*tA);
	transb = cublasop(*tB);
	cublascall(cublasDgemm(handle, transa, transb, *rA, 
		    *cB, *cA, a, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extB),
		    *ldB, b, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to calculate the symmetric matrix-matrix multiplication
C = a  A  B  + b C where a and b are scalars, 
and A , B and C are sysmmetric matrices stored in column-major format 
with dimensions  A  m � m ,  B and C are m � n 
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP symmGPU(SEXP extA, SEXP extB, SEXP extC, SEXP lda, SEXP ldb, SEXP ldc,
             SEXP fillmod, SEXP side, SEXP m, SEXP n, SEXP alpha, SEXP beta)
{
	checkExternalPrt(extA);
	checkExternalPrt(extB);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldB = INTEGER(ldb);
	int *ldC = INTEGER(ldc);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *sideMode = REAL(side);	
	cublasSideMode_t sidemode;
	sidemode = cublasSideMode(*sideMode);
	double *f = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	int *rA = INTEGER(m);
	int *cB = INTEGER(n);    
	cublascall(cublasDsymm(handle, sidemode, fillmode, *rA, 
		    *cB, a, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extB),
		    *ldB, b, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to perform the symmetric rank k update
C = a op(A) op(A)T  + b C where a and b are scalars, 
A is matrix with dimension n x k, C is sysmmetric matrix stored in column-major format  
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP syrkGPU(SEXP extA, SEXP extC, SEXP lda, SEXP ldc,
             SEXP fillmod, SEXP trans, SEXP n, SEXP k, SEXP alpha, SEXP beta)
{
	checkExternalPrt(extA);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldC = INTEGER(ldc);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *t = REAL(trans);	
	cublasOperation_t transa;
	transa = cublasop(*t);
	double *f = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	int *rA = INTEGER(n);
	int *cA = INTEGER(k);    
	cublascall(cublasDsyrk(handle, fillmode, transa, *rA, 
		    *cA, a, R_ExternalPtrAddr(extA),
		    *ldA, b, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to perform the symmetric rank 2k update
C = a(op(A) op(B)T + op(B) op(A)T) + b C where a and b are scalars, 
A and B are matrices with dimensions n x k and n x k,respectivley. 
C is sysmmetric matrix stored in column-major format  
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP syr2kGPU(SEXP extA, SEXP extB, SEXP extC, SEXP lda, SEXP ldb, SEXP ldc,
             SEXP fillmod, SEXP trans, SEXP n, SEXP k, SEXP alpha, SEXP beta)
{
	checkExternalPrt(extA);
	checkExternalPrt(extB);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldB = INTEGER(ldb);
	int *ldC = INTEGER(ldc);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *t = REAL(trans);	
	cublasOperation_t transa;
	transa = cublasop(*t);
	double *f = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	int *rA = INTEGER(n);
	int *cA = INTEGER(k);    
	cublascall(cublasDsyr2k(handle, fillmode, transa, *rA, 
		    *cA, a, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extB), *ldB, 
                  b, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to calculate the triangle matrix-matrix multiplication
C = a op(A) B, where a is scalar 
and A is triangle matrix stored in column-major format 
B and C are m � n matrices
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP trmmGPU(SEXP extA, SEXP extB, SEXP extC, SEXP lda, SEXP ldb, SEXP ldc,
             SEXP trans, SEXP fillmod, SEXP diag, SEXP side, SEXP m, SEXP n, SEXP alpha)
{
	checkExternalPrt(extA);
	checkExternalPrt(extB);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldB = INTEGER(ldb);
	int *ldC = INTEGER(ldc);
	double *a = REAL(alpha);
	double *t = REAL(trans);	
	cublasOperation_t transa;
	transa = cublasop(*t);
	double *sideMode = REAL(side);	
	cublasSideMode_t sidemode;
	sidemode = cublasSideMode(*sideMode);
	double *f = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	double *d = REAL(diag);
	cublasDiagType_t diagmode;
	diagmode = cublasdiagtype(*d);
	int *rA = INTEGER(m);
	int *cB = INTEGER(n);    
	cublascall(cublasDtrmm(handle, sidemode, fillmode, transa, diagmode,  
		    *rA, *cB, a, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extB),
		    *ldB, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to solve the triangle linear system with multiple right-hand-sides
op(A) X = a B or X op(A)= a B, where a is scalar 
and A is triangle matrix stored in column-major format 
X and B are m � n matrices
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP trsmGPU(SEXP extA, SEXP extB, SEXP lda, SEXP ldb, 
             SEXP trans, SEXP fillmod, SEXP diag, SEXP side, SEXP m, SEXP n, SEXP alpha)
{
	checkExternalPrt(extA);
	checkExternalPrt(extB);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldB = INTEGER(ldb);
	double *a = REAL(alpha);
	double *t = REAL(trans);	
	cublasOperation_t transa;
	transa = cublasop(*t);
	double *sideMode = REAL(side);	
	cublasSideMode_t sidemode;
	sidemode = cublasSideMode(*sideMode);
	double *f = REAL(fillmod);
	cublasFillMode_t fillmode;
	fillmode = cublasfillmod(*f);
	double *d = REAL(diag);
	cublasDiagType_t diagmode;
	diagmode = cublasdiagtype(*d);
	int *rA = INTEGER(m);
	int *cB = INTEGER(n);    
	cublascall(cublasDtrsm(handle, sidemode, fillmode, transa, diagmode,  
		    *rA, *cB, a, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extB),
		    *ldB));
	cublascall(cublasDestroy_v2(handle));
	return(extB);
}


/*
define function to calculate the matrix times matrix
Input is 2 R's external pointers pointing to 2 GPU vectors
and the dimensions of the 2 matrices, 
output is R's external pointer pointing to GPU
vector(device)
*/
SEXP mmGPU(SEXP extM1, SEXP extM2, SEXP m, SEXP n, SEXP k)
{
	checkExternalPrt(extM1);
	checkExternalPrt(extM2);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM = INTEGER(m);
	int *lenthN = INTEGER(n);
	int *lenthK = INTEGER(k);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthM * (*lenthN) * sizeof(double)));
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	UNPROTECT(1);
	double *alpha = malloc(1 * sizeof(double));
	alpha[0] = 1;
	double *beta = malloc(1 * sizeof(double));
	beta[0] = 0;	
	cublascall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, *lenthM, 
		    *lenthN, *lenthK, alpha, R_ExternalPtrAddr(extM1),
		    *lenthM, R_ExternalPtrAddr(extM2),
		    *lenthK, beta, R_ExternalPtrAddr(ext), *lenthM));
	cublascall(cublasDestroy_v2(handle));
	return(ext);
}

/*
define function to transpose a matrix
Input is a R's external pointers pointing to a GPU vectors
and the dimensions of the matrix, 
output is R's external pointer pointing to GPU
vector(device)
*/
SEXP tGPU(SEXP extM, SEXP m, SEXP n)
{
	checkExternalPrt(extM);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *lenthM = INTEGER(m);
	int *lenthN = INTEGER(n);
	double *x;
	cudacall(cudaMalloc((void**)&x, *lenthM * (*lenthN) * sizeof(double)));
	SEXP ext = PROTECT(R_MakeExternalPtr(x, R_NilValue,R_NilValue));
	R_RegisterCFinalizerEx(ext, _finalizer, TRUE);
	UNPROTECT(1);
	double *alpha = malloc(1 * sizeof(double));
	alpha[0]=1;
	double *beta = malloc(1 * sizeof(double));
	beta[0]=0;	
	cublascall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, *lenthN, 
		    *lenthM, alpha, R_ExternalPtrAddr(extM), *lenthM, beta,
		    R_ExternalPtrAddr(extM), *lenthN,
		    R_ExternalPtrAddr(ext), *lenthN));
	cublascall(cublasDestroy_v2(handle));
	return(ext);
}

/*CULBLAS extension functions*/


/*
define function to performs the matrix-matrix addition/transposition
C = a op ( A ) + b op ( B ) where a and b are scalars, 
and A , B and C are matrices stored in column-major format 
with dimensions m x n
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP geamGPU(SEXP extA, SEXP extB, SEXP extC, SEXP lda, SEXP ldb, SEXP ldc,
             SEXP m, SEXP n, SEXP transA, SEXP transB,
             SEXP alpha, SEXP beta)
{
	checkExternalPrt(extA);
	checkExternalPrt(extB);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *ldB = INTEGER(ldb);
	int *ldC = INTEGER(ldc);
	double *a = REAL(alpha);
	double *b = REAL(beta);
	double *tA = REAL(transA);
	double *tB = REAL(transB);
	int *rA = INTEGER(m);
	int *cB = INTEGER(n);   
	cublasOperation_t transa;
	cublasOperation_t transb;
	transa = cublasop(*tA);
	transb = cublasop(*tB);
	cublascall(cublasDgeam(handle, transa, transb, *rA, 
		    *cB, a, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extB),
		    *ldB, b, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to performs the matrix-matrix multiplication
C = A diag(x) or C = diag(x) A where a is scalar, 
A and C are matrices stored in column-major format 
with dimensions m x n, X is a vector of size n or m depends on 
CUBLAS_SIDE setup.
Input is 3 R's external pointers pointing to 3 GPU matrices
output is R's external pointer pointing to GPU
matrix(device)
*/
SEXP dgmmGPU(SEXP extA, SEXP extx, SEXP extC, SEXP lda, SEXP incx, SEXP ldc,
             SEXP m, SEXP n, SEXP side)
{
	checkExternalPrt(extA);
	checkExternalPrt(extx);
	checkExternalPrt(extC);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle));
	int *ldA = INTEGER(lda);
	int *incX = INTEGER(incx);
	int *ldC = INTEGER(ldc);
	int *rA = INTEGER(m);
	int *cB = INTEGER(n);   
	double *sideMode = REAL(side);	
	cublasSideMode_t sidemode;
	sidemode = cublasSideMode(*sideMode);
	cublascall(cublasDdgmm(handle, sidemode, *rA, 
		    *cB, R_ExternalPtrAddr(extA),
		    *ldA, R_ExternalPtrAddr(extx),
		    *incX, R_ExternalPtrAddr(extC), *ldC));
	cublascall(cublasDestroy_v2(handle));
	return(extC);
}


/*
define function to inverse a square matrix by LU decomposition
Input is a R's external pointers pointing to a GPU vectors
and the dimensions of the matrix, 
output is R's external pointer pointing to GPU
vector(device)
note:this function is only for small size of function, the 
cublas call launch overhead is a significant factor
*/
SEXP inversGPU(SEXP ext, SEXP n)
{
	checkExternalPrt(ext);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle)); 
	int *lenthN = INTEGER(n);  
	//cublas function is originally written for batched matrices
	//we set batchsize to be 1 to perform inverse of a single matrix
	int batchSize = 1;
	//vector P is the pivot for LU decomposition
	int *P;
	cudacall(cudaMalloc((void**)&P, *lenthN * batchSize * sizeof(int)));
	int *info;
	//info is the information about LU decomposition function
	cudacall(cudaMalloc((void**)&info, batchSize * sizeof(int)));
	double* input;
	cudacall(cudaMalloc((void**)&input, *lenthN * 
                           (*lenthN) * sizeof(double)));
	cublascall(cublasDcopy(handle, *lenthN * (*lenthN), 
		R_ExternalPtrAddr(ext), 1, input, 1));
	//input of cublasDgetrfBatched is supposed to be array of pointers 
	//to matrices, here we create a double pointer pointing to
	//size-one array of matrix(device)
	double *A[] = {input};
	double** A_d;
	cudacall(cudaMalloc((void**)&A_d, batchSize * sizeof(A)));
	cublascall(cublasSetVector(batchSize, sizeof(A), A, 1, A_d, 1));
	//perform LU decomposition and output the pivot to vector P(device)
	cublascall(cublasDgetrfBatched(handle, *lenthN, A_d, 
		    *lenthN, P, info, batchSize));
	//output of cublasDgetrfBatched is supposed to be array of pointers 
	//to matrices, here we create a double pointer pointing to 
	//size-one array of matrix(device)
	double* output;
	cudacall(cudaMalloc((void**)&output, *lenthN * (*lenthN) * sizeof(double)));
	double* C[] = {output};
	double** C_d;
	cudacall(cudaMalloc((void**)&C_d, batchSize * sizeof(C)));
	cublascall(cublasSetVector(batchSize, sizeof(C), C, 1, C_d, 1));   
	//perform matrix inverse by taking the pivot P as input
	cublascall(cublasDgetriBatched(handle, *lenthN, A_d, *lenthN, P, 
		    C_d, *lenthN, info, batchSize));
	SEXP out = PROTECT(R_MakeExternalPtr(output, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(out, _finalizer, TRUE);
	UNPROTECT(1);
	cublascall(cublasDestroy_v2(handle));
	return(out);
}

/*
define function to find the least square solution for matrix
by applying the QR factorization
Input is 2 R's external pointers pointing to a GPU vectors
and the dimensions of the matrix, 
output is R's external pointer pointing to GPU
vector(device)
note:this function is only for small size of function, the 
cublas call launch overhead is a significant factor
*/
SEXP lsGPU(SEXP extM, SEXP extV, SEXP m, SEXP n)
{
	checkExternalPrt(extM);
	checkExternalPrt(extV);
	cublasHandle_t handle;
	cublascall(cublasCreate_v2(&handle)); 
	int *lenthN = INTEGER(n);  
	int *lenthM = INTEGER(m);  
	//cublas function is originally written for batched matrices
	//we set batchsize to be 1 to perform solution for single matrix
	int batchSize = 1;
	int *devinfoArray;
	cudacall(cudaMalloc((void**)&devinfoArray, batchSize * sizeof(int)));
	int *info;
	cudacall(cudaMalloc((void**)&info, batchSize * sizeof(int)));
	double* input;
	cudacall(cudaMalloc((void**)&input, *lenthN * (*lenthM) * sizeof(double)));
	cublascall(cublasDcopy(handle, *lenthN * (*lenthM), 
		R_ExternalPtrAddr(extM), 1, input, 1));
	//input of cublasDgelsBatched is supposed to be array of pointers 
	//to matrices, here we create a double pointer pointing to
	//size-one array of matrix(device)
	double *A[] = {input};
	double** A_d;
	cudacall(cudaMalloc((void**)&A_d, batchSize* sizeof(A)));
	cublascall(cublasSetVector(batchSize, sizeof(A), A, 1, A_d, 1));
	//output of cublasDgelsBatched is supposed to be array of pointers 
	//to matrices, here we create a double pointer pointing to
	//size-one array of matrix(device)
	double* output;
	cudaMalloc((void**)&output, *lenthN * sizeof(double));
	cublasDcopy(handle, *lenthN, R_ExternalPtrAddr(extV), 1, output, 1);
	double* C[] = {output};
	double** C_d;
	cudacall(cudaMalloc((void**)&C_d, batchSize* sizeof(C)));
	cublascall(cublasSetVector(batchSize, sizeof(C), C, 1, C_d, 1));   
	//	cublascall(cublasDgelsBatched(handle, CUBLAS_OP_N, *lenthM, *lenthN, 1, A_d,
	//	                              *lenthM, C_d, *lenthN, info,
	//					  devinfoArray, batchSize));
	SEXP out = PROTECT(R_MakeExternalPtr(output, R_NilValue, R_NilValue));
	R_RegisterCFinalizerEx(out, _finalizer, TRUE);
	UNPROTECT(1);
	cublascall(cublasDestroy_v2(handle));
	return(out);
}

/*CURAND random generator functions*/
/*
define function to generate uniform distributied random
number, input is length of vector and seed 
output is pointer pointing to a GPU vector(device)
*/
SEXP uniformRNGGPU(SEXP n, SEXP seed)
{
	curandGenerator_t gen;
	double *s = REAL(seed);
	int *lenthN = INTEGER(n); 
	double *x;
	cudacall(cudaMalloc((void**)&x, (*lenthN) * sizeof(double)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	UNPROTECT(1);
	//create generator
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	//create psedo random generator
	curandSetPseudoRandomGeneratorSeed(gen, *s);
	//generate uniform random number of lenth n
	curandGenerateUniformDouble(gen, R_ExternalPtrAddr(out), *lenthN);
	curandDestroyGenerator(gen);
	return out;
}

/*
define function to generate normal distributied random
number, input is length of vector, mean, standard deviation
and seed, output is pointer pointing to a GPU vector(device)
*/
SEXP normRNGGPU(SEXP n, SEXP mean, SEXP sd, SEXP seed)
{
	curandGenerator_t gen;
	double *s = REAL(seed);
	int *lenthN = INTEGER(n); 
	double *m = REAL(mean);
	double *d = REAL(sd);
	double *x;
	cudacall(cudaMalloc((void**)&x, (*lenthN) * sizeof(double)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	UNPROTECT(1);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, *s);
	curandGenerateNormalDouble(gen, R_ExternalPtrAddr(out),
		                    *lenthN, *m, *d);
	curandDestroyGenerator(gen);
	return out;
}

/*
define function to generate Poisson distributied random
number, input is length of vector, lambda
and seed, output is pointer pointing to a GPU vector(device)
*/
SEXP poissonRNGGPU(SEXP n, SEXP lambda, SEXP seed)
{
	curandGenerator_t gen;
	double *s = REAL(seed);
	int *lenthN = INTEGER(n); 
	double *L = REAL(lambda);
	int *x;
	cudacall(cudaMalloc((void**)&x, (*lenthN) * sizeof(int)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	UNPROTECT(1);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, *s);
	curandGeneratePoisson(gen, R_ExternalPtrAddr(out), *lenthN, *L);
	curandDestroyGenerator(gen);
	return out;
}

/*
define function to generate log-normal distributied random
number, input is length of vector, mean, standard deviation
and seed, output is pointer pointing to a GPU vector(device)
*/
SEXP lognormRNGGPU(SEXP n, SEXP mean, SEXP sd, SEXP seed)
{
	curandGenerator_t gen;
	double *s = REAL(seed);
	int *lenthN = INTEGER(n); 
	double *m = REAL(mean);
	double *d = REAL(sd);
	double *x;
	cudacall(cudaMalloc((void**)&x, (*lenthN) * sizeof(double)));
	SEXP out = PROTECT(R_MakeExternalPtr(x, R_NilValue, R_NilValue));
	UNPROTECT(1);
	curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, *s);
	curandGenerateLogNormalDouble(gen, R_ExternalPtrAddr(out),
		                       *lenthN, *m, *d);
	curandDestroyGenerator(gen);
	return out;
}
