NVCC = nvcc
CUDA_ARCH = -arch=sm_86

all: runcuda

.PHONY: runcuda
runcuda:
	$(NVCC) $(CUDA_ARCH) -DUSE_CUBLAS=1 -g -o runcuda llama3.cu -lm -lcublas

.PHONY: clean
clean:
	rm -f runcuda
