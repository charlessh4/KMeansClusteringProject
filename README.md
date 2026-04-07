# K-means Clustering: CPU vs GPU Performance Analysis

EECE 5640 Final Project — Shawn Charles

Benchmarks K-means clustering on an Intel Xeon CPU (OpenMP) against an NVIDIA Tesla V100 GPU (CUDA) using synthetic 2D point datasets of varying sizes.

---

## Files

| File: Description |
| `generate_data.c`: Generates reproducible binary datasets of 2D points |
| `kmeans_cpu.c`: K-means on CPU using OpenMP, with thread-scaling support |
| `kmeans_gpu.cu`: K-means on GPU using CUDA, with H2D/kernel/D2H timing |
| `Makefile`: Builds all three executables |
| `run_experiments.sh`: Runs the full benchmark suite and saves output to `results.txt` |

---

## Build

```bash
module load Cuda
make all
```

To build individually:
```bash
make generate   # generate_data only
make cpu        # kmeans_cpu only
make gpu        # kmeans_gpu only
```
---

## Usage

### 1. Generate datasets
```bash
./generate_data <num_points> <output_file>

./generate_data 100000   data_100k.bin
./generate_data 1000000  data_1M.bin
./generate_data 10000000 data_10M.bin
```

### 2. Run CPU (OpenMP)
```bash
srun --partition=courses --constraint=cascadelake --nodes=1 --cpus-per-task=28 --pty /bin/bash
```

```bash
module load Vtune
./kmeans_cpu data_1M.bin 10 28 100
```

Experiments:
```bash
module load Vtune
./run_experiments cpu
```
### 3. Run GPU (CUDA)

First, request an interactive GPU node:
```bash
srun -p courses-gpu --gres=gpu:v100-sxm2:1 --pty --time=03:00:00 /bin/bash
module load cuda
module load nsight
```

Then run the GPU binary:
```bash
./kmeans_gpu <input_file> <K> [max_iter]

./kmeans_gpu data_1M.bin 10 100
```

Experiments:
```bash
./run_experiments cpu
```



## Dataset Format

Binary file written by `generate_data`:
```
[int32 N] [float32 x0] [float32 y0] [float32 x1] [float32 y1] ...
```
Points are seeded deterministically (seed 42) around 10 ground-truth cluster centers, making results reproducible across runs.

---

