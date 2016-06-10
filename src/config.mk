#set R include and library path
R_HOME := $(shell R RHOME)
R_INC := /usr/share/R/include
R_LIB := /usr/lib/

#set CUDA home path
CUDA_HOME := /usr/local/cuda-7.5

#set NVCC compiler path
NVCC := $(CUDA_HOME)/bin/nvcc

#set CUDA include path
CUDA_INC := $(CUDA_HOME)/include

#set CUDA library path
CUDA_LIB := $(CUDA_HOME)/lib64

#set system structure
OS := $(shell uname -s)
ifeq ($(OS), Darwin)
    ifeq ($(getconf LONG_BIT), 64)
        DEVICEOPTS := -m64
    endif
    CUDA_LIB := $(CUDA_HOME)/lib
    R_FRAMEWORK := -F$(R_HOME)/.. -framework R
    RPATH := -rpath $(CUDA_LIB)
endif
