/*
 * generate_data.c
 * Generates synthetic 2D point datasets for K-means benchmarking.
 *
 * Binary format: [int32 N][float32 x0][float32 y0] ... [float32 xN-1][float32 yN-1]
 *
 * Usage: ./generate_data <num_points> <output_file>
 * Example: ./generate_data 1000000 data_1M.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// 10 well-separated cluster centers used  
static const float CENTERS[10][2] = {
    {1.0f, 1.0f}, {4.0f, 1.0f}, {7.0f, 1.0f},
    {1.0f, 4.0f}, {4.0f, 4.0f}, {7.0f, 4.0f},
    {1.0f, 7.0f}, {4.0f, 7.0f}, {7.0f, 7.0f},
    {4.0f, 10.0f}
};
#define NUM_CENTERS 10
#define SPREAD      0.6f   // std-dev-like spread around each center 

// Simple LCG for reproducible, fast random numbers 
static unsigned int lcg_state = 42;
static float lcg_randf(void) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return (float)(lcg_state >> 8) / (float)(1 << 24);
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <num_points> <output_file>\n", argv[0]);
        return 1;
    }

    long N = atol(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Error: num_points must be positive\n");
        return 1;
    }
    const char *outfile = argv[2];

    FILE *f = fopen(outfile, "wb");
    if (!f) {
        perror("fopen");
        return 1;
    }

    // Write point count 
    int n32 = (int)N;
    fwrite(&n32, sizeof(int), 1, f);

    // Write points in bulk chunks to avoid N individual fwrite calls 
    const int CHUNK = 1 << 16;  // 64K points per chunk 
    float *buf = malloc(CHUNK * 2 * sizeof(float));
    if (!buf) {
        fprintf(stderr, "Out of memory\n");
        fclose(f);
        return 1;
    }

    long written = 0;
    while (written < N) {
        int batch = (N - written < CHUNK) ? (int)(N - written) : CHUNK;
        for (int i = 0; i < batch; i++) {
            int c = (int)(lcg_randf() * NUM_CENTERS) % NUM_CENTERS;
            // Box-Muller approximation: sum of 4 uniforms shifted to [-1,1] 
            float nx = (lcg_randf() + lcg_randf() - 1.0f) * SPREAD;
            float ny = (lcg_randf() + lcg_randf() - 1.0f) * SPREAD;
            buf[2 * i]     = CENTERS[c][0] + nx;
            buf[2 * i + 1] = CENTERS[c][1] + ny;
        }
        fwrite(buf, sizeof(float), batch * 2, f);
        written += batch;
    }

    free(buf);
    fclose(f);

    double mb = (double)N * 2 * sizeof(float) / (1024.0 * 1024.0);
    printf("Generated %ld points -> %s  (%.1f MB data)\n", N, outfile, mb);
    return 0;
}
