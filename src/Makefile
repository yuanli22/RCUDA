include config.mk

#source code file extension 
EXT1 := cu
EXT2 := c 
OBJS11 := supplementc.o 
OBJS12 := supplementcu.o 
OBJS2 := R_interface.o 

#compiler/preprocessor options
CPICFLAGS := $(shell R CMD config CPICFLAGS)
INCS := -I$(R_INC) -I$(CUDA_INC)  
PARAMS1 := $(DEVICEOPTS) -Xcompiler $(CPICFLAGS)
PARAMS3 := -fpic -c

#linker options
LD_PARAMS1 := $(DEVICEOPTS) -Xlinker '$(RPATH) $(R_FRAMEWORK)'
LD_PARAMS2 := $(DEVICEOPTS)  '$(RPATH) $(R_FRAMEWORK)'
LIBS := -L$(CUDA_LIB) -lcudart -lcuda -lcublas -lcurand  -L$(R_LIB) -lR  

#shared library 
TARGETS1 := supplement.so
TARGETS2 := cudalib.so

##setup compiler
NVCC := $(CUDA_HOME)/bin/nvcc 
GCC := gcc 

all: $(TARGETS1) $(TARGETS2)

$(OBJS12): %.o: %.$(EXT1)
	$(NVCC) -c $(INCS) $(PARAMS1) $^ -o $@

$(OBJS11): %.o: supplementc.$(EXT2)
	$(GCC) $(INCS) -fpic -c $^ -o $@

$(TARGETS1): $(OBJS11) $(OBJS12)
	$(NVCC) -shared $(LD_PARAMS1) $(LIBS) $(OBJS11) $(OBJS12) -o $@

$(TARGETS2): $(OBJS2)
	$(GCC) -shared -o $@ $(OBJS2) $(LIBS) 

$(OBJS2): %.o: R_interface.$(EXT2)
	$(GCC) $(INCS) $(PARAMS3) $^ -o $@
 
clean:
	rm -rf *o

.PHONY: all clean

