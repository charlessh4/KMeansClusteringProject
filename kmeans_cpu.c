/*
 * kmeans_cpu.c
 * K-means clustering on CPU using OpenMP for parallelization.
 *
 * Measures:
 *   - Total clock time for the full algorithm
 *   - Per-iteration breakdown 
 *   - Strong scaling across thread counts
 *
 * Usage: ./kmeans_cpu <input_file> <K> <num_threads> [max_iter]
 * Example: ./kmeans_cpu data_1M.bin 10 28 100
 *
 * Compile: gcc -O3 -march=native -fopenmp -o kmeans_cpu kmeans_cpu.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <omp.h>

// Timing helper
static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

// Read dataset written by generate_data                               
static int read_dataset(const char *path, int *out_N,
                        float **out_x, float **out_y) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    int N;
    if (fread(&N, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "Error reading point count\n");
        fclose(f); return -1;
    }

    float *buf = malloc((size_t)N * 2 * sizeof(float));
    if (!buf) {
        fprintf(stderr, "Out of memory allocating %d points\n", N);
        fclose(f); return -1;
    }

    size_t got = fread(buf, sizeof(float), (size_t)N * 2, f);
    fclose(f);
    if ((int)(got / 2) != N) {
        fprintf(stderr, "Short read: expected %d points, got %zu\n", N, got / 2);
        free(buf); return -1;
    }

    // Take apart into separate x / y arrays for better cache behaviour 
    float *x = malloc((size_t)N * sizeof(float));
    float *y = malloc((size_t)N * sizeof(float));
    if (!x || !y) { fprintf(stderr, "OOM\n"); free(buf); return -1; }

    for (int i = 0; i < N; i++) {
        x[i] = buf[2 * i];
        y[i] = buf[2 * i + 1];
    }
    free(buf);

    *out_N = N;  *out_x = x;  *out_y = y;
    return 0;
}

// Main
int main(int argc, char *argv[]) {
    if (argc < 4 || argc > 5) {
        fprintf(stderr, "Usage: %s <input_file> <K> <num_threads> [max_iter]\n", argv[0]);
        return 1;
    }

    const char *infile   = argv[1];
    int K = atoi(argv[2]);
    int nthreads = atoi(argv[3]);
    int max_iter = (argc == 5) ? atoi(argv[4]) : 100;

    if (K <= 0 || nthreads <= 0 || max_iter <= 0) {
        fprintf(stderr, "K, num_threads, and max_iter must all be > 0\n");
        return 1;
    }

    omp_set_num_threads(nthreads);

    // Load data
    int N;  float *x, *y;
    if (read_dataset(infile, &N, &x, &y) != 0) return 1;
    printf("Loaded %d points from %s\n", N, infile);

    // Allocate working arrays
    float *cx      = malloc(K * sizeof(float));  // centroid x 
    float *cy      = malloc(K * sizeof(float));  // centroid y
    float *new_cx  = malloc(K * sizeof(float));
    float *new_cy  = malloc(K * sizeof(float));
    int   *counts  = malloc(K * sizeof(int));
    int   *labels  = calloc(N, sizeof(int));
    if (!cx || !cy || !new_cx || !new_cy || !counts || !labels) {
        fprintf(stderr, "OOM\n"); return 1;
    }

    // Initialize centroids: pick first K points
    for (int k = 0; k < K; k++) {
        cx[k] = x[k];
        cy[k] = y[k];
    }

    // Allocate per-thread accumulators once (reused across iterations)
    float **all_lcx = malloc(nthreads * sizeof(float *));
    float **all_lcy = malloc(nthreads * sizeof(float *));
    int   **all_lct = malloc(nthreads * sizeof(int *));
    if (!all_lcx || !all_lcy || !all_lct) { fprintf(stderr, "OOM\n"); return 1; }
    for (int t = 0; t < nthreads; t++) {
        all_lcx[t] = malloc(K * sizeof(float));
        all_lcy[t] = malloc(K * sizeof(float));
        all_lct[t] = malloc(K * sizeof(int));
        if (!all_lcx[t] || !all_lcy[t] || !all_lct[t]) {
            fprintf(stderr, "OOM\n"); return 1;
        }
    }

    // Run K-means
    printf("\nRunning K-means: K=%d, threads=%d, max_iter=%d\n",
           K, nthreads, max_iter);

    // Convergence threshold: matches GPU so max centroid shift < 1e-4 
    const float CONV_THRESH = 1e-4f;

    double t_assign_total = 0.0;
    double t_update_total = 0.0;
    double t_wall_start   = now_sec();

    int   iter      = 0;
    int   converged = 0;

    while (!converged && iter < max_iter) {

        // Assignment step
        double ta = now_sec();

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++) {
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

        t_assign_total += now_sec() - ta;

        // Update step (thread-local accumulators and merge)
        double tu = now_sec();

        memset(new_cx, 0, K * sizeof(float));
        memset(new_cy, 0, K * sizeof(float));
        memset(counts, 0, K * sizeof(int));

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            float *lcx = all_lcx[tid];
            float *lcy = all_lcy[tid];
            int   *lct = all_lct[tid];

            memset(lcx, 0, K * sizeof(float));
            memset(lcy, 0, K * sizeof(float));
            memset(lct, 0, K * sizeof(int));

            #pragma omp for schedule(static)
            for (int i = 0; i < N; i++) {
                int k = labels[i];
                lcx[k] += x[i];
                lcy[k] += y[i];
                lct[k]++;
            }

            #pragma omp critical
            {
                for (int k = 0; k < K; k++) {
                    new_cx[k] += lcx[k];
                    new_cy[k] += lcy[k];
                    counts[k] += lct[k];
                }
            }
        }

        // Compute new centroids and check max centroid shift
        float max_shift = 0.0f;
        for (int k = 0; k < K; k++) {
            if (counts[k] > 0) {
                float new_x = new_cx[k] / counts[k];
                float new_y = new_cy[k] / counts[k];
                float dx = new_x - cx[k];
                float dy = new_y - cy[k];
                float shift = sqrtf(dx * dx + dy * dy);
                if (shift > max_shift) max_shift = shift;
                cx[k] = new_x;
                cy[k] = new_y;
            }
        }

        t_update_total += now_sec() - tu;
        iter++;

        if (max_shift < CONV_THRESH) converged = 1;
    }

    double t_wall = now_sec() - t_wall_start;

    // Report 
    printf("\n=== CPU K-means Results ===\n");
    printf("  Input file: %s\n", infile);
    printf("  Points (N): %d\n", N);
    printf("  Clusters (K): %d\n", K);
    printf("  Threads: %d\n", nthreads);
    printf("  Iterations: %d  (%s)\n", iter,
           converged ? "converged" : "max_iter reached");
    printf("  Assignment: %.4f s\n", t_assign_total);
    printf("  Update: %.4f s\n", t_update_total);
    printf("  Total wall: %.4f s\n", t_wall);
    printf("\n  Final centroids:\n");
    for (int k = 0; k < K; k++)
        printf("[%2d]  (%.4f, %.4f)  n=%d\n", k, cx[k], cy[k], counts[k]);

    for (int t = 0; t < nthreads; t++) {
        free(all_lcx[t]); free(all_lcy[t]); free(all_lct[t]);
    }
    free(all_lcx); free(all_lcy); free(all_lct);
    free(x); free(y); free(cx); free(cy);
    free(new_cx); free(new_cy); free(counts); free(labels);
    return 0;
}
