# RCUDA

This is an R GPU computing package via NVIDIA CUDA framework. It consists of wrappers of cuBLAS cuRAND libraries and self-defined CUDA functions. By defining gpu objective in R environment, we want to provide a high performance GPU solution of linear algebra. Our package enables user to keep as much work on GPU side as possible that avoids unnecessary memory operation between CPU and GPU.
 
## Prerequisities

* R version over 3.2.0 available at https://www.r-project.org/

* NVIDIA GPU with computing capacity over 2.0 (you can check your hardware spec at https://en.wikipedia.org/wiki/Nvidia_Tesla)

* CUDA toolkit version over 7.5 available at (https://developer.nvidia.com/cuda-downloads)

* LINUX system with GNU Compiler Collection GCC version over 4.8.4 available at https://gcc.gnu.org/

* RCUDA is currently LINUX-only package


## Installing package
To install RCUDA, user need to speficy the path for both R and CUDA includes and library.

1. **R_HOME** specifies the R root path, for example 'R_HOME = /usr/bin/R' 

2. **R_INC** specifies the R root path, for example 'R_INC := /usr/share/R/include'

3. **R_LIB** specifies the R root path, for example 'R_LIB := /usr/lib/' 

4. **CUDA_HOME** specifies the R root path, for example 'CUDA_HOME := /usr/local/cuda-7.5'

5. **CUDA_INC** specifies the R root path, for example 'CUDA_INC := $(CUDA_HOME)/include'

6. **CUDA_LIB** specifies the R root path, for example 'CUDA_LIB := $(CUDA_HOME)/lib64' (depends on your system structure 32-bit or 64-bit)



## Running the tests
* Build the package in linux by command 'R CMD build RCUDA' to get the file **RCUDA_1.0.tar.gz**

* In R, compile your package by command 'install.packages("yourpath/RCUDA_1.0.tar.gz", repos = NULL)'

* In R, load the library by command 'library(RCUDA)'

* In R, run the test file (located at the 'tests' folder) by command 'source("yourpath/srcs/test.R")'


## Sample code
Suppose we already installed and loaded the library in R, here is a sample code to perform matrix-vector multiplication,
```{r} 
creategpu(1:4, 2, 2) -> A     ##create a 2 by 2 matrix in GPU and assign it to object A

creategpu(1:2) -> b       ##create a vector in GPU and assign it to object b

mvgpu(A, b) -> c      ##compute the result A*b and store the result in GPU object c. 
                      ##Here mvgpu is the matrix-vector multiplication function in RCUDA
                        
gathergpu(c) -> result       ##transfer the outcome from c (GPU object) to result (CPU object) 
                             ##which can be applied to native R function later
```


## Available functions
### BLAS implementation progress

#### Level 1
CUBLAS functions:

* [x] amax
* [x] amin
* [x] asum
* [x] axpy
* [x] copy
* [x] dot
* [x] nrm2
* [ ] rot  
* [ ] rotg  
* [ ] rotm 
* [ ] rotmg  
* [x] scal
* [ ] swap  

#### Level 2

Key:
* `ge`: general
* `gb`: general banded
* `sy`: symmetric
* `sb`: symmetric banded
* `sp`: symmetric packed
* `tr`: triangular
* `tb`: triangular banded
* `tp`: triangular packed
* `he`: hermitian
* `hb`: hermitian banded
* `hp`: hermitian packed

CUBLAS functions:

* [x] gbmv  
* [x] gemv  
* [x] ger  
* [x] sbmv  
* [ ] spmv
* [ ] spr
* [ ] spr2
* [x] symv  
* [x] syr 
* [ ] syr2
* [x] tbmv
* [x] tbsv
* [ ] tpmv
* [ ] tpsv
* [x] trmv  
* [x] trsv  
* [ ] hemv  
* [ ] hbmv
* [ ] hpmv
* [ ] her 
* [ ] her2
* [ ] hpr
* [ ] hpr2

#### Level 3

CUBLAS functions:

* [x] gemm  
* [x] gemmBatched
* [x] symm  
* [x] syrk  
* [x] syr2k 
* [ ] syrkx
* [x] trmm  
* [x] trsm  
* [x] trsmBatched
* [ ] hemm
* [ ] herk  
* [ ] her2k  
* [ ] herkx

#### BLAS-like extensions

* [x] geam
* [x] dgmm
* [x] getrfBatched
* [x] getriBatched
* [x] geqrfBatched
* [x] gelsBatched
* [ ] tpttr
* [ ] trttp


### Random number generators

*  Gaussian
*  Log-Gaussian
*  Poisson
*  Uniform
*  Gamma
*  Beta


### High level statistical functions

*  sample Mean
*  sample Variance
*  Covariance
*  Gaussian density function
*  Gamma density function
*  Beta density function
 



## Authors

* **Yuan Li** - *Initial work* 
* **Hua Zhou** - *Package design*
 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

 
