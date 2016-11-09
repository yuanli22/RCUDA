#include <stdio.h>
#include <curand_kernel.h>
#include <unistd.h>
#include <curand.h>
#define M 512
#define CUDART_PI_F 3.141592654f

  
// the CUDA kernel for vector addition
__global__ void sum(double *a, double *b, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = a[idx] + b[idx];
	}
}

// the CUDA kernel for vector subtract
__global__ void subtract(double *a, double *b, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = a[idx] - b[idx];
	}
}

// the CUDA kernel for vector multiply
__global__ void multi(double *a, double *b, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = a[idx] * b[idx];
	}
}

// the CUDA kernel for element-wise vector divide
__global__ void divide(double *a, double *b, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = __ddiv_rn(a[idx], b[idx]);
	}
}

// the CUDA kernel for vector exp
__global__ void cudaexp(double *a,	double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = exp(a[idx]);
	}
}

// the CUDA kernel for vector log
__global__ void cudalog(double *a,	double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = log(a[idx]);
	}
}

// the CUDA kernel for vector square root
__global__ void cudasqrt(double *a, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = sqrt(a[idx]);
	}
}

// the CUDA kernel for gamma function
__global__ void cudagamma(double *a, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = tgamma(a[idx]);
	}
}

// the CUDA kernel for gamma pdf
__global__ void cudagammapdf(double *a, double *out, int n, double k, double theta)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = (1 / (tgamma(k) * pow(theta, k))) * pow(a[idx], 
                         (k - 1)) * exp(-a[idx] / theta);
	}
}

// the CUDA kernel for beta pdf
__global__ void cudabetapdf(double *a, double *out, int n, double k, double theta)
{
 	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = pow(a[idx], (k-1)) * pow((1 - a[idx]), (theta - 1)) 
                         / (tgamma(k) * tgamma(theta)
                         / tgamma(k + theta));	
	}
}

// the CUDA kernel for beta function
__global__ void cudabeta(double *a, double *b, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = tgamma(a[idx]) * tgamma(b[idx])
                         / tgamma(a[idx] + b[idx]);
	}
}

// the CUDA kernel for vector power
__global__ void cudapower(double *a, double *out, int n, double alpha)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = pow(a[idx], alpha);
	}
}

// the CUDA kernel for vector plus constant
__global__ void cudavecincre(double *a, double *out, int n, double incre)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = a[idx] + incre;
	}
}

// the CUDA kernel for normal pdf
__global__ void cudanormdensity(double *a, double *out, int n,
                                double mean, double sd)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = (1 / (sd  *sqrt(2 * CUDART_PI_F)))
                         * exp(-pow((a[idx] - mean), 2) / (2 * pow(sd, 2)));
	}
}

// the CUDA kernel for normal CDF
__global__ void cudanormCDF(double *a, double *out, int n)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = normcdf(a[idx]);
	}
}

//the CUDA kernel for sample variance
 __global__ void cuda_var(double *input, double *out, int n, double mean)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = pow(input[idx] - mean, 2);
	}
}

// the kernel for vector reduction summation
__global__ void cudareduction(double * input, double * output, int len) 
{
	// Load a segment of the input vector into shared memory
	__shared__ double partialSum[2 * M];
	int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int t = threadIdx.x;
	unsigned int start = 2 * blockIdx.x * blockDim.x;
	if ((start + t) < len)
	{
		partialSum[t] = input[start + t];      
	}
	else
	{       
		partialSum[t] = 0.0;
	}
	if ((start + blockDim.x + t) < len)
	{   
		partialSum[blockDim.x + t] = input[start + blockDim.x + t];
	}
	else
	{
		partialSum[blockDim.x + t] = 0.0;
	}
	// Traverse reduction tree
	for (unsigned int stride = blockDim.x; stride > 0; stride /= 2)
	{
		__syncthreads();
		if (t < stride)
			partialSum[t] += partialSum[t + stride];
	}
	__syncthreads();
	// Write the computed sum of the block to the output vector at correct index
	if (t == 0 && (globalThreadId * 2) < len)
	{
		output[blockIdx.x] = partialSum[t];
	}
}

// the CUDA kernel for vector subset copying
__global__ void vectorsubset(double *a, double *out, int n, int *index)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx < n) {
		out[idx] = a[index[idx] - 1];
	}
}

/*
Following part contains the CUDA kernels invocation functions called 
from the host side (CPU) 
*/

// CUDA vector reduction summation kernel invocation function
extern "C" void cuda_sum(double *a, double *b, double *c, int n)
{
	sum<<<(n + M - 1) / M, M>>>(a, b, c, n);
	return;
}

// CUDA vector subtraction kernel invocation function
extern "C" void cuda_subtract(double *a, double *b, double *c, int n)
{
	subtract<<<(n + M - 1) / M , M>>>(a, b, c, n);
	return;
}

// CUDA vector multiplication kernel invocation function
extern "C" void cuda_multi(double *a, double *b, double *c, int n)
{
	multi<<<(n + M - 1) / M, M>>>(a, b, c, n);
	return;
}

// CUDA vector division kernel invocation function
extern "C" void cuda_divide(double *a, double *b, double *c, int n)
{
	divide<<<(n + M - 1) / M, M>>>(a, b, c, n);
	return;
}

// CUDA vector exponential kernel invocation function
extern "C" void cuda_exp(double *a, double *c, int n)
{
	cudaexp<<<(n + M - 1) / M, M>>>(a, c, n);
	return;
}

// CUDA vector logarithm kernel invocation function
extern "C" void cuda_log(double *a, double *c, int n)
{
	cudalog<<<(n + M - 1) / M, M>>>(a, c, n);
	return;
}

// CUDA vector square root kernel invocation function
extern "C" void cuda_sqrt(double *a, double *c, int n)
{
	cudasqrt<<<(n + M - 1) / M, M>>>(a, c, n);
	return;
}

// CUDA vector gamma function kernel invocation function
extern "C" void cuda_gamma(double *a, double *c, int n)
{
	cudagamma<<<(n + M - 1) / M, M>>>(a, c, n);
	return;
}

// CUDA vector beta function kernel invocation function
extern "C" void cuda_beta(double *a, double *b, double *c, int n)
{
	cudabeta<<<(n + M - 1) / M, M>>>(a, b, c, n);
	return;
}

// CUDA vector gamma pdf function kernel invocation function
extern "C" void cuda_gammapdf(double *a, double *c, int n, double k,
                              double theta)
{
	cudagammapdf<<<(n + M - 1) / M, M>>>(a, c, n, k, theta);
	return;
}

// CUDA vector beta pdf function kernel invocation function
extern "C" void cuda_betapdf(double *a, double *c, int n, 
                          double k, double theta)
{
	cudabetapdf<<<(n + M - 1) / M, M>>>(a, c, n, k, theta);
	return;
}

// CUDA vector power kernel invocation function
extern "C" void cuda_power(double *a, double *c, int n, double alpha)
{
	cudapower<<<(n + M - 1) / M, M>>>(a, c, n, alpha);
	return;
}

// CUDA vector plus constant invocation function
extern "C" void cuda_vecincre(double *a, double *c, int n, double alpha)
{
	cudavecincre<<<(n + M - 1) / M, M>>>(a, c, n, alpha);
	return;
}

// CUDA vector normal density function kernel invocation function
extern "C" void cuda_normal_density(double *a, double *c, int n, 
                                     double mean, double sd)
{
	cudanormdensity<<<(n + M - 1) / M, M>>>(a, c, n, mean, sd);
	return;
}

// CUDA vector normal cumulative density function kernel invocation function
extern "C" void cuda_normal_CDF(double *a, double *c, int n)
{
	cudanormCDF<<<(n + M - 1) / M, M>>>(a, c, n);
	return;
}

// CUDA vector variance kernel invocation function
extern "C" void cudavariance(double *a, double *c, int n, double mean)
{
	cuda_var<<<(n + M - 1) / M, M>>>(a, c, n, mean);
	return;
}

// CUDA vector subset copy kernel invocation function
extern "C" void vector_subset(double *a, double *c, int n, int *index)
{
	vectorsubset<<<(n + M - 1) / M, M>>>(a, c, n, index);
	return;
}

// CUDA vector summation kernel invocation function
extern "C" double cuda_reduction(double *a, int n)
{
	int numOutputElements = n / (M<<1);
	if (n % (M<<1)) 
	{
		numOutputElements++;
	}
	double * hostOutput = (double*) malloc(numOutputElements 
                                              * sizeof(double));
	double * deviceOutput;
	cudaMalloc((void **)&deviceOutput, numOutputElements * sizeof(double));
	dim3 DimGrid( numOutputElements, 1, 1);
	dim3 DimBlock(M, 1, 1);
	cudareduction<<<DimGrid, DimBlock>>>(a, deviceOutput, n);  
	cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(double), 
		    cudaMemcpyDeviceToHost);
	for (int ii = 1; ii < numOutputElements; ii++) 
	{
		hostOutput[0] += hostOutput[ii];
	}
	cudaFree(deviceOutput); 
	return hostOutput[0];
}

// Print GPU device information
void printDevProp(cudaDeviceProp devProp)
{
	printf("Major revision number:         %d\n",  devProp.major);
	printf("Minor revision number:         %d\n",  devProp.minor);
	printf("Name:                          %s\n",  devProp.name);
	printf("Total global memory:           %u\n",  devProp.totalGlobalMem);
	printf("Total shared memory per block: %u\n",  devProp.sharedMemPerBlock);
	printf("Total registers per block:     %d\n",  devProp.regsPerBlock);
	printf("Warp size:                     %d\n",  devProp.warpSize);
	printf("Maximum memory pitch:          %u\n",  devProp.memPitch);
	printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", 
                      i, devProp.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, 
                      devProp.maxGridSize[i]);
	printf("Clock rate:                    %d\n",  devProp.clockRate);
	printf("Total constant memory:         %u\n",  devProp.totalConstMem);
	printf("Texture alignment:             %u\n",  devProp.textureAlignment);
	printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? 
		"Yes" : "No"));
	printf("Number of multiprocessors:     %d\n",  
               devProp.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n",  
              (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
}

extern "C" void gpuquery()
{
	// Number of CUDA devices
	int devCount;
	cudaGetDeviceCount(&devCount);
	printf("CUDA Device Query...\n");
	printf("There are %d CUDA devices.\n", devCount);
	// Iterate through devices
	for (int i = 0; i < devCount; ++i)
	{
		// Get device properties
		printf("\nCUDA Device #%d\n", i);
		cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, i);
		printDevProp(devProp);
	}
}

  

//kernel to initalize curandState
__global__ void init(double seed, curandState_t* states, int length) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < length) curand_init(seed, id, 0, &states[id]);
}

//kernel to generate gamma random variable by using George Marsaglia and Wai Wan Tsang's method
__global__ void gammarand(curandState* states, double a, double b, double* numbers, int length) 

{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < length){

    if (a > 1)
 {
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1){
        do{
            x = curand_normal_double(&states[id]);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(&states[id]);

        if (u < 1 - 0.0331 * x * x * x * x) 
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    numbers[id] = b * d * v;
 }
    else 
 {
    const double ap = a + 1.0;
    double x, v, u, up;
    double d = ap - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);
    up = curand_uniform_double(&states[id]);
    while (1){
        do{
            x = curand_normal_double(&states[id]);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(&states[id]);

        if (u < 1 - 0.0331 * x * x * x * x) 
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    numbers[id] = b * d * v;
    numbers[id] = numbers[id] * pow (up, 1.0 / a);
 }
}
}

// CUDA gamma RNG kernel invocation function
extern "C" void gammarng(double a, double b, int n, double seed, double* number)
{
  curandState* states;
  cudaMalloc((void**) &states, n * sizeof(curandState));
  init<<<n + M - 1, M>>>(seed, states, n);
  gammarand<<<(n + M - 1) / M, M>>>(states, a , b, number, n);
}


//kernel to generate beta random variable by using beta-gamma relationship
__global__ void betarand(curandState* states, double a, double b, double* numbers, int length) 

{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < length){

    if (a > 1)
 {
    double x, v, u;
    double d = a - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1){
        do{
            x = curand_normal_double(&states[id]);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(&states[id]);

        if (u < 1 - 0.0331 * x * x * x * x) 
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    numbers[id] = 1 * d * v;
 }
    else 
 {
    const double ap = a + 1;
    double x, v, u, up;
    double d = ap - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);
    up = curand_uniform_double(&states[id]);
    while (1){
        do{
            x = curand_normal_double(&states[id]);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(&states[id]);

        if (u < 1 - 0.0331 * x * x * x * x) 
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    numbers[id] = 1 * d * v;
    numbers[id] = numbers[id] * pow (up, 1.0 / a);
 }
  double y;
  if (b > 1)
 {
    double x, v, u;
    double d = b - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);

    while (1){
        do{
            x = curand_normal_double(&states[id]);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(&states[id]);

        if (u < 1 - 0.0331 * x * x * x * x) 
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    y = 1 * d * v;
 }
    else 
 {
    const double bp = b + 1;
    double x, v, u, up;
    double d = bp - 1.0 / 3.0;
    double c = (1.0 / 3.0) / sqrt (d);
    up = curand_uniform_double(&states[id]);
    while (1){
        do{
            x = curand_normal_double(&states[id]);
            v = 1.0 + c * x;
        } while (v <= 0);

        v = v * v * v;
        u = curand_uniform_double(&states[id]);

        if (u < 1 - 0.0331 * x * x * x * x) 
            break;

        if (log (u) < 0.5 * x * x + d * (1 - v + log (v)))
            break;
    }
    y = 1 * d * v;
    y = y * pow (up, 1.0 / b);
 }
    numbers[id] = numbers[id] / (numbers[id] + y);
}
}


// CUDA beta RNG kernel invocation function
extern "C" void betarng(double a, double b, int n, double seed, double* numbers)
{
  curandState* states;
  cudaMalloc((void**) &states, n * sizeof(curandState));
  init<<<n + M - 1, M>>>(seed, states, n);
  betarand<<<(n + M - 1) / M, M>>>(states, a , b, numbers, n);

}

// CUDA beta dirichlet kernel invocation function
extern "C" void dirichletrng(double a, int K, int n, double seed, double* numbers)
{
  curandState* states;
  cudaMalloc((void**) &states, n * sizeof(curandState));
  init<<<n + M - 1, M>>>(seed, states, n);
 

}

