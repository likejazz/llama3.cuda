NVCC = nvcc

all: runcuda

.PHONY: runcuda
runcuda:
	$(NVCC) -DUSE_CUBLAS=1 -g -o runcuda llama3.cu -lm -lcublas

.PHONY: clean
clean:
	rm -f runcuda
