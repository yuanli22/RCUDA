/*
This source file contains CUDA and CUBLAS safe call function, R external
pointer finalizer and check functions.  
*/
#include <R.h>
#include <cuda.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>

/*
define cuda call error check function,
any cudacall will return error message, 
if any error occur, this function will 
return the error information and stop the 
program.
*/
#define cudacall(call)                                                           \
	do                                                                        \
	{                                                                         \
	cudaError_t err = (call);                                                 \
	if(cudaSuccess != err)                                                    \
		{                                                                  \
		fprintf(stderr, "CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n",\
		__FILE__, __LINE__, cudaGetErrorString(err));                      \
		cudaDeviceReset();                                                 \
		exit(EXIT_FAILURE);                                                \
		}                                                                  \
	}                                                                         \
	while (0)

/*
define cudablas call error check function,
any cudablas function call will return error message, 
if any error occur, this function will 
return the error information and stop the 
program.
*/
#define cublascall(call)                                                      \
	do                                                                        \
	{                                                                         \
	cublasStatus_t status = (call);                                       \
	if (CUBLAS_STATUS_NOT_INITIALIZED == status)                     \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_NOT_INITIALIZED\n", __FILE__, __LINE__ );\
		}                                                                     \
		else if (CUBLAS_STATUS_ALLOC_FAILED == status)                        \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_ALLOC_FAILED\n", __FILE__, __LINE__ );   \
		}                                                                     \
		else if (CUBLAS_STATUS_INVALID_VALUE == status)                       \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_INVALID_VALUE\n", __FILE__, __LINE__ );   \
		}                                                                     \
		else if (CUBLAS_STATUS_ARCH_MISMATCH == status)                       \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_ARCH_MISMATCH\n", __FILE__, __LINE__ );   \
		}                                                                     \
		else if (CUBLAS_STATUS_MAPPING_ERROR == status)                       \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_MAPPING_ERROR\n", __FILE__, __LINE__ );   \
		}                                                                     \
		else if (CUBLAS_STATUS_EXECUTION_FAILED == status)                    \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_EXECUTION_FAILED\n", __FILE__, __LINE__ );\
		}                                                                     \
		else if (CUBLAS_STATUS_INTERNAL_ERROR == status)                      \
		{                                                                     \
		fprintf(stderr, "CUBLAS Error:\nFile = %s\nLine =                 \
%d\nCUBLAS_STATUS_INTERNAL_ERROR\n", __FILE__, __LINE__ );  \
		}                                                                     \
	}                                                                         \
	while(0)

/*
define finalizer of R external pointer,
input is R external pointer, function will finalize the pointer 
when it is not in use and free the allocated memory.
*/
static void _finalizer(SEXP ext)
{
	if (!R_ExternalPtrAddr(ext))
		return;
       double * ptr = (double *) R_ExternalPtrAddr(ext);
	cudacall(cudaFree(ptr));
	R_ClearExternalPtr(ext);
}

/*
define check function for R external pointer
return error message if input is 
not R external pointer.
*/
SEXP  checkExternalPrt(SEXP input)
{
	if (TYPEOF(input) != EXTPTRSXP)
	{
		error("argument not external pointer");
	}

}


/*
# convert input integer to cublasOperation_t
*/
cublasOperation_t cublasop(double input)
{
	cublasOperation_t transa;
	if (input == 1){
	transa = CUBLAS_OP_N;    
 	} else if (input == 2) { 
                transa = CUBLAS_OP_T;
	}  else if (input == 3) { 
                transa = CUBLAS_OP_C;}
       return transa;
}


/*
# convert input integer to cublasFillMode_t
*/
cublasFillMode_t cublasfillmod(double input)
{
	cublasFillMode_t fillmod;
	if (input == 1){
	fillmod = CUBLAS_FILL_MODE_UPPER;    
 	} else if (input == 2) { 
         fillmod = CUBLAS_FILL_MODE_LOWER;
	}  
       return fillmod;
}


/*
# convert input integer to cublasDiagType_t
*/
cublasDiagType_t cublasdiagtype(double input)
{
	cublasDiagType_t diagtype;
	if (input == 1){
	diagtype = CUBLAS_DIAG_NON_UNIT;    
 	} else if (input == 2) { 
         diagtype = CUBLAS_DIAG_UNIT;
	}  
       return diagtype;
}


/*
# convert input integer to cublasSideMode_t side
*/
cublasSideMode_t cublasSideMode(double input)
{
	cublasSideMode_t sidemode;
	if (input == 1){
	sidemode = CUBLAS_SIDE_LEFT;    
 	} else if (input == 2) { 
         sidemode = CUBLAS_SIDE_RIGHT;
	}  
       return sidemode;
}
