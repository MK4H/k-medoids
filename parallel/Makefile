CPP=mpicxx
STD=-std=c++17
CFLAGS=-Wextra -g $(STD)
NVCCFLAGS=-ccbin $(CPP) $(STD) -gencode=arch=compute_52,code=sm_52 -gencode=arch=compute_70,code=sm_70 -G -Xcompiler -rdynamic -lineinfo
INCLUDE=/usr/local/cuda/include /mnt/home/_teaching/bpplib/include ./headers ../shared
LDFLAGS=
LIBS=
LIBDIRS=/usr/local/cuda/lib64
HEADERS=$(shell find ../shared -name '*.hpp') $(shell find . -name '*.hpp')
CU_HEADERS=$(shell find . -name '*.cuh')
TARGET=parallel


.PHONY: all clear clean purge

all: $(TARGET)



# Building Targets

$(TARGET): $(TARGET).cpp $(HEADERS) $(CU_HEADERS) kernels.obj
	@echo Compiling and linking executable "$@" ...
	@$(CPP) $(CFLAGS) $(addprefix -I,$(INCLUDE)) $(LDFLAGS) $(addprefix -L,$(LIBDIRS)) $(addprefix -l,$(LIBS)) -lcudart kernels.obj $< -o $@

kernels.obj: cuda/kernels.cu
	@echo Compiling kernels ...
	@nvcc $(NVCCFLAGS) $(addprefix -I,$(INCLUDE)) --compile -cudart static $< -o $@
# Cleaning Stuff

clear:
	@echo Removing object files ...
	-@rm -f *.obj

clean: clear

purge: clear
	@echo Removing executable ...
	-@rm -f $(TARGET)
