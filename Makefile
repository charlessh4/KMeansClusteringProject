CC      = gcc
NVCC    = nvcc
CFLAGS  = -O3 -march=native -fopenmp -std=gnu99
NVFLAGS = -O3 --gpu-architecture=compute_70 --gpu-code=sm_70 
LDFLAGS = -lm

all: generate_data kmeans_cpu kmeans_gpu

generate_data: generate_data.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

kmeans_cpu: kmeans_cpu.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

kmeans_gpu: kmeans_gpu.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

generate: generate_data
cpu:      kmeans_cpu
gpu:      kmeans_gpu

clean:
	rm -f generate_data kmeans_cpu kmeans_gpu
	rm -f data_100k.bin data_1M.bin data_10M.bin

