#include "cudadefine.h"
/*
First 2 functions are the data allocation 
and transformation between R and C
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
	 
	//copying CPU to GPU
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
	//copying GPU vector back to CPU
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
	//the cublasIdamin only takes pointer for the output argument
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
	cublascall(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, *lenthM, *lenthN,
		alpha, R_ExternalPtrAddr(extM1),
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
	cublascall(cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, *lenthM, *lenthN,
		alpha, R_ExternalPtrAddr(extM1),
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
	cublascall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, *lenthM, *lenthN,
		*lenthK, alpha, R_ExternalPtrAddr(extM1),
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
	cublascall(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, *lenthN, *lenthM,
		alpha, R_ExternalPtrAddr(extM), *lenthM, beta,
		R_ExternalPtrAddr(extM), *lenthN,
		R_ExternalPtrAddr(ext), *lenthN));
	cublascall(cublasDestroy_v2(handle));
	return(ext);
}

/*CULBLAS extension functions*/
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
	cudacall(cudaMalloc((void**)&input, *lenthN * (*lenthN) * sizeof(double)));
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
	//perform inverse by taking the pivot P as input
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
define function to generate poisson distributied random
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
