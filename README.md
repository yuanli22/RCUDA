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

## Available functions
### CUBLAS implementation progress

The following sections list the CUBLAS functions shown on the CUBLAS
documentation page:

http://docs.nvidia.com/cuda/cublas/index.html

### Level 1 (13 functions)

CUBLAS functions:

* [x] amax
* [x] amin
* [x] asum
* [x] axpy
* [x] copy
* [x] dot, dotc, dotu
* [x] nrm2
* [ ] rot (not implemented in julia blas.jl)
* [ ] rotg (not implemented in julia blas.jl)
* [ ] rotm (not implemented in julia blas.jl)
* [ ] rotmg (not implemented in julia blas.jl)
* [x] scal
* [ ] swap (not implemented in julia blas.jl)

### Level 2

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

* [x] gbmv (in julia/blas.jl)
* [x] gemv (in julia/blas.jl)
* [x] ger (in julia/blas.jl)
* [x] sbmv (in julia/blas.jl)
* [ ] spmv
* [ ] spr
* [ ] spr2
* [x] symv (in julia/blas.jl)
* [x] syr (in julia/blas.jl)
* [ ] syr2
* [x] tbmv
* [x] tbsv
* [ ] tpmv
* [ ] tpsv
* [x] trmv (in julia/blas.jl)
* [x] trsv (in julia/blas.jl)
* [x] hemv (in julia/blas.jl)
* [x] hbmv
* [ ] hpmv
* [x] her (in julia/blas.jl)
* [x] her2
* [ ] hpr
* [ ] hpr2

### Level 3

CUBLAS functions:

* [x] gemm (in julia/blas.jl)
* [x] gemmBatched
* [x] symm (in julia/blas.jl)
* [x] syrk (in julia/blas.jl)
* [x] syr2k (in julia/blas.jl)
* [ ] syrkx
* [x] trmm (in julia/blas.jl)
* [x] trsm (in julia/blas.jl)
* [x] trsmBatched
* [x] hemm
* [x] herk (in julia/blas.jl)
* [x] her2k (in julia/blas.jl)
* [ ] herkx

### BLAS-like extensions

* [x] geam
* [x] dgmm
* [x] getrfBatched
* [x] getriBatched
* [x] geqrfBatched
* [x] gelsBatched
* [ ] tpttr
* [ ] trttp


## Authors

* **Yuan Li** - *Initial work* 
* **Hua Zhou** - *Package design*
 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

 
