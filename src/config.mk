#set your R include and library path
R_HOME := $(shell R RHOME)
#R_HOME := /usr/local/r-3.1.0/lib64/R
R_INC := /usr/share/R/include
R_LIB := /usr/lib/

#set NVCC compiler path
# replace these three lines with
 CUDA_HOME := /usr/local/cuda-7.5
ifndef CUDA_HOME
    CUDA_HOME := /usr/local/cuda-7.5
endif
NVCC := $(CUDA_HOME)/bin/nvcc

# set CUDA_INC to CUDA header dir on your system
CUDA_INC := $(CUDA_HOME)/include

# set CUDA_LIB to CUDA library dir on your system
CUDA_LIB := $(CUDA_HOME)/lib64

OS := $(shell uname -s)
ifeq ($(OS), Darwin)
    ifeq ($(getconf LONG_BIT), 64)
        DEVICEOPTS := -m64
    endif
    CUDA_LIB := $(CUDA_HOME)/lib
    R_FRAMEWORK := -F$(R_HOME)/.. -framework R
    RPATH := -rpath $(CUDA_LIB)
endif

CPICFLAGS := $(shell R CMD config CPICFLAGS)
