/*
 * kmeans_gpu.cu
 * K-means clustering on NVIDIA GPU using CUDA.
 *
 * Timing is broken into three sections using CUDA events:
 *   1. Host-to-Device (H2D) memory transfer
 *   2. Kernel execution (all iterations combined)
 *   3. Device-to-Host (D2H) memory transfer
 *
 * Convergence is checked each iteration by copying back the centroids
 * (K * 2 * 4 bytes — negligible for any reasonable K) and comparing
 * against the previous iteration.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>

// CUDA error checking
#define check_cuda(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                           \
        if (_e != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(_e));           \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel parameters
#define BLOCK_SIZE 256  // threads per block for point-level kernels 

// Kernel 1: assign each point to its nearest centroid
__global__ void kernel_assign(const float * __restrict__ x,
                              const float * __restrict__ y,
                              const float * __restrict__ cx,
                              const float * __restrict__ cy,
                              int *labels,
                              int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = x[i], yi = y[i];
    float min_d = FLT_MAX;
    int   best  = 0;

    for (int k = 0; k < K; k++) {
        float dx = xi - cx[k];
        float dy = yi - cy[k];
        float d  = dx * dx + dy * dy;
        if (d < min_d) { min_d = d; best = k; }
    }
    labels[i] = best;
}

// Kernel 2: accumulate per-cluster sums using atomics
// Each thread adds its point's coordinates to its cluster's bucket
__global__ void kernel_accumulate(const float * __restrict__ x,
                                  const float * __restrict__ y,
                                  const int * __restrict__ labels,
                                  float *sum_x, float *sum_y,
                                  int *counts,
                                  int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int k = labels[i];
    atomicAdd(&sum_x[k],  x[i]);
    atomicAdd(&sum_y[k],  y[i]);
    atomicAdd(&counts[k], 1);
}

// Kernel 3: divide accumulated sums to get new Centroids
__global__ void kernel_update(float *cx, float *cy,
                              const float * __restrict__ sum_x,
                              const float * __restrict__ sum_y,
                              const int   * __restrict__ counts,
                              int K)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    if (counts[k] > 0) {
        cx[k] = sum_x[k] / (float)counts[k];
        cy[k] = sum_y[k] / (float)counts[k];
    }
}

// Read dataset written by generate_data
static int read_dataset(const char *path, int *out_N,
                        float **out_x, float **out_y)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    int N;
    if (fread(&N, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Cannot read point count\n");
        fclose(f); return -1;
    }

    /* Read interleaved [x0 y0 x1 y1 ...] */
    float *buf = (float *)malloc((size_t)N * 2 * sizeof(float));
    if (!buf) { fprintf(stderr, "OOM\n"); fclose(f); return -1; }

    size_t got = fread(buf, sizeof(float), (size_t)N * 2, f);
    fclose(f);
    if ((int)(got / 2) != N) {
        fprintf(stderr, "Short read: expected %d, got %zu points\n", N, got/2);
        free(buf); return -1;
    }

    // De-interleave 
    float *x = (float *)malloc((size_t)N * sizeof(float));
    float *y = (float *)malloc((size_t)N * sizeof(float));
    if (!x || !y) { fprintf(stderr, "OOM\n"); free(buf); return -1; }

    for (int i = 0; i < N; i++) {
        x[i] = buf[2 * i];
        y[i] = buf[2 * i + 1];
    }
    free(buf);

    *out_N = N;  *out_x = x;  *out_y = y;
    return 0;
}

// Elapsed ms between two CUDA events
static float event_ms(cudaEvent_t start, cudaEvent_t stop) {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

// Main                          
int main(int argc, char *argv[]) {
    if (argc < 3 || argc > 4) {
        fprintf(stderr, "Usage: %s <input_file> <K> [max_iter]\n", argv[0]);
        return 1;
    }
    const char *infile   = argv[1];
    int         K        = atoi(argv[2]);
    int         max_iter = (argc == 4) ? atoi(argv[3]) : 100;

    if (K <= 0 || max_iter <= 0) {
        fprintf(stderr, "K and max_iter must be > 0\n"); return 1;
    }

    // Print GPU info
    {
        cudaDeviceProp prop;
        check_cuda(cudaGetDeviceProperties(&prop, 0));
        printf("GPU: %s  (SM %d.%d, %.0f MB)\n",
               prop.name, prop.major, prop.minor,
               prop.totalGlobalMem / (1024.0 * 1024.0));
    }

    // Load data on host
    int N;  float *h_x, *h_y;
    if (read_dataset(infile, &N, &h_x, &h_y) != 0) return 1;
    printf("Loaded %d points from %s\n\n", N, infile);

    // Host centroid arrays
    float *h_cx = (float *)malloc(K * sizeof(float));
    float *h_cy = (float *)malloc(K * sizeof(float));
    float *h_cx_prev = (float *)malloc(K * sizeof(float));
    float *h_cy_prev = (float *)malloc(K * sizeof(float));
    for (int k = 0; k < K; k++) {
        h_cx[k] = h_x[k];
        h_cy[k] = h_y[k];
    }

    // Device allocations
    float *d_x, *d_y, *d_cx, *d_cy, *d_sum_x, *d_sum_y;
    int *d_labels, *d_counts;

    check_cuda(cudaMalloc(&d_x, (size_t)N * sizeof(float)));
    check_cuda(cudaMalloc(&d_y, (size_t)N * sizeof(float)));
    check_cuda(cudaMalloc(&d_cx, K * sizeof(float)));
    check_cuda(cudaMalloc(&d_cy, K * sizeof(float)));
    check_cuda(cudaMalloc(&d_sum_x, K * sizeof(float)));
    check_cuda(cudaMalloc(&d_sum_y, K * sizeof(float)));
    check_cuda(cudaMalloc(&d_labels, (size_t)N * sizeof(int)));
    check_cuda(cudaMalloc(&d_counts, K * sizeof(int)));

    // CUDA events for section timing 
    cudaEvent_t ev_h2d_s, ev_h2d_e;
    cudaEvent_t ev_ker_s,  ev_ker_e;
    cudaEvent_t ev_d2h_s, ev_d2h_e;
    check_cuda(cudaEventCreate(&ev_h2d_s)); check_cuda(cudaEventCreate(&ev_h2d_e));
    check_cuda(cudaEventCreate(&ev_ker_s));  check_cuda(cudaEventCreate(&ev_ker_e));
    check_cuda(cudaEventCreate(&ev_d2h_s)); check_cuda(cudaEventCreate(&ev_d2h_e));

    // SECTION 1: Host to Device transfer
    check_cuda(cudaEventRecord(ev_h2d_s));
    check_cuda(cudaMemcpy(d_x,  h_x,  (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_y,  h_y,  (size_t)N * sizeof(float), cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_cx, h_cx, K * sizeof(float),         cudaMemcpyHostToDevice));
    check_cuda(cudaMemcpy(d_cy, h_cy, K * sizeof(float),         cudaMemcpyHostToDevice));
    check_cuda(cudaEventRecord(ev_h2d_e));
    check_cuda(cudaEventSynchronize(ev_h2d_e));

    // SECTION 2: Kernel execution
    int blocks_N = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int blocks_K = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Convergence threshold: max centroid shift < 1e-4 
    const float CONV_THRESH = 1e-4f;
    int iter = 0;
    int converged = 0;

    check_cuda(cudaEventRecord(ev_ker_s));

    while (!converged && iter < max_iter) {

        // Save previous centroids for convergence check
        memcpy(h_cx_prev, h_cx, K * sizeof(float));
        memcpy(h_cy_prev, h_cy, K * sizeof(float));

        // Reset accumulators
        check_cuda(cudaMemset(d_sum_x,  0, K * sizeof(float)));
        check_cuda(cudaMemset(d_sum_y,  0, K * sizeof(float)));
        check_cuda(cudaMemset(d_counts, 0, K * sizeof(int)));

        // Assign points to nearest centroid
	kernel_assign<<<blocks_N, BLOCK_SIZE>>>(d_x, d_y, d_cx, d_cy,
						d_labels, N, K);
	check_cuda(cudaGetLastError());

	// Accumulate points
	kernel_accumulate<<<blocks_N, BLOCK_SIZE>>>(d_x, d_y, d_labels,
						    d_sum_x, d_sum_y,
						    d_counts, N);
	check_cuda(cudaGetLastError());

	// Update centroid
	kernel_update<<<blocks_K, BLOCK_SIZE>>>(d_cx, d_cy,
						d_sum_x, d_sum_y,
						d_counts, K);

	check_cuda(cudaGetLastError());

        check_cuda(cudaDeviceSynchronize());
        
        // Copy new centroids back (K*2*4 bytes — negligible) for convergence 
        check_cuda(cudaMemcpy(h_cx, d_cx, K * sizeof(float), cudaMemcpyDeviceToHost));
        check_cuda(cudaMemcpy(h_cy, d_cy, K * sizeof(float), cudaMemcpyDeviceToHost));

        // Check max centroid shift 
        float max_shift = 0.0f;
        for (int k = 0; k < K; k++) {
            float dx = h_cx[k] - h_cx_prev[k];
            float dy = h_cy[k] - h_cy_prev[k];
            float d  = sqrtf(dx * dx + dy * dy);
            if (d > max_shift) max_shift = d;
        }
        if (max_shift < CONV_THRESH) converged = 1;

        iter++;
    }

    check_cuda(cudaEventRecord(ev_ker_e));
    check_cuda(cudaEventSynchronize(ev_ker_e));
    check_cuda(cudaDeviceSynchronize());  // ensure all kernels finished 

    // SECTION 3: Device → Host transfer (labels + final centroids)
    int *h_labels = (int *)malloc((size_t)N * sizeof(int));
    if (!h_labels) { fprintf(stderr, "OOM\n"); return 1; }

    check_cuda(cudaEventRecord(ev_d2h_s));
    check_cuda(cudaMemcpy(h_labels, d_labels, (size_t)N * sizeof(int),
                          cudaMemcpyDeviceToHost));
    // Centroids already copied for convergence; re-copy to keep timing honest 
    check_cuda(cudaMemcpy(h_cx, d_cx, K * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaMemcpy(h_cy, d_cy, K * sizeof(float), cudaMemcpyDeviceToHost));
    check_cuda(cudaEventRecord(ev_d2h_e));
    check_cuda(cudaEventSynchronize(ev_d2h_e));

    // Collect cluster sizes from labels
    int *cluster_count = (int *)calloc(K, sizeof(int));
    for (int i = 0; i < N; i++) cluster_count[h_labels[i]]++;

    // Timing summary 
    float ms_h2d = event_ms(ev_h2d_s, ev_h2d_e);
    float ms_ker = event_ms(ev_ker_s,  ev_ker_e);
    float ms_d2h = event_ms(ev_d2h_s, ev_d2h_e);
    float ms_tot = ms_h2d + ms_ker + ms_d2h;

    printf("GPU K-means Results\n");
    printf("  Input file: %s\n", infile);
    printf("  Points (N): %d\n", N);
    printf("  Clusters (K): %d\n", K);
    printf("  Iterations: %d  (%s)\n", iter,
           converged ? "converged" : "max_iter reached");
    printf("\n  Timing breakdown:\n");
    printf("H2D transfer: %10.3f ms  (%5.1f%%)\n",
           ms_h2d, 100.0f * ms_h2d / ms_tot);
    printf("Kernel exec: %10.3f ms  (%5.1f%%)\n",
           ms_ker,  100.0f * ms_ker  / ms_tot);
    printf("D2H transfer: %10.3f ms  (%5.1f%%)\n",
           ms_d2h, 100.0f * ms_d2h / ms_tot);
    printf("Total GPU: %10.3f ms  (%.4f s)\n",
           ms_tot, ms_tot / 1000.0f);
    printf("\nFinal centroids:\n");
    for (int k = 0; k < K; k++)
        printf("[%2d]  (%.4f, %.4f)  n=%d\n",
               k, h_cx[k], h_cy[k], cluster_count[k]);

    // Cleanup
    cudaFree(d_x);  cudaFree(d_y);  cudaFree(d_cx);  cudaFree(d_cy);
    cudaFree(d_sum_x);  cudaFree(d_sum_y);
    cudaFree(d_labels); cudaFree(d_counts);

    free(h_x); free(h_y); free(h_cx); free(h_cy);
    free(h_cx_prev); free(h_cy_prev); free(h_labels); free(cluster_count);

    cudaEventDestroy(ev_h2d_s); cudaEventDestroy(ev_h2d_e);
    cudaEventDestroy(ev_ker_s);  cudaEventDestroy(ev_ker_e);
    cudaEventDestroy(ev_d2h_s); cudaEventDestroy(ev_d2h_e);

    return 0;
}
