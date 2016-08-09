#include <unistd.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

#define N 500
#define blocksize 512
#define MAX 100

//kernel to initalize curandState
__global__ void init(unsigned int seed, curandState_t* states, const int length) {
  int id = threadIdx.x + blockIdx.x * blockDim.x;
  if (id < length) curand_init(seed, id, 0, &states[id]);
}

//kernel to generate gamma random variable by using George Marsaglia and Wai Wan Tsang's method
__global__ void randoms(curandState* states, const double a, const double b, double* numbers, const int length) 

{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < length)
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
}


int main( ) {
  curandState* states;
  cudaMalloc((void**) &states, N * sizeof(curandState));
  init<<<N + blocksize - 1, blocksize>>>(time(0), states, N);
  double cpu_nums[N];
  double* gpu_nums;
  cudaMalloc((void**) &gpu_nums, N * sizeof(double));
  randoms<<<(N + blocksize - 1) / blocksize, blocksize>>>(states, 5.5,1,gpu_nums, N);
  cudaMemcpy(cpu_nums, gpu_nums, N * sizeof(double), cudaMemcpyDeviceToHost);
  
  for (int i = 0; i < N; i++) {
    printf("%d = %f\n", i,cpu_nums[i]);
  }


  cudaFree(states);
  cudaFree(gpu_nums);

  return 0;
}