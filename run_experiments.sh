#!/bin/bash
set -e

TARGET="$1"

if [ -z "$TARGET" ]; then
    echo "Usage: $0 {cpu|gpu}"
    exit 1
fi

if [ "$TARGET" != "cpu" ] && [ "$TARGET" != "gpu" ]; then
    echo "Usage: $0 {cpu|gpu}"
    exit 1
fi

K=10
MAX_ITERS=100
THREADS_LIST="1 2 4 8 16 28"
DATA_SIZES="100000 1000000 10000000"

DATA_DIR="data"
RESULTS_DIR="results"
CPU_RAW_DIR="${RESULTS_DIR}/cpu_raw"
GPU_RAW_DIR="${RESULTS_DIR}/gpu_raw"
CPU_PROFILE_DIR="${RESULTS_DIR}/cpu_profiles"
GPU_PROFILE_DIR="${RESULTS_DIR}/gpu_profiles"

GEN_EXE="./generate_data"
CPU_EXE="./kmeans_cpu"
GPU_EXE="./kmeans_gpu"

CPU_SUMMARY="${RESULTS_DIR}/cpu_summary.csv"
GPU_SUMMARY="${RESULTS_DIR}/gpu_summary.csv"

mkdir -p "${DATA_DIR}" "${RESULTS_DIR}"

# Always overwrite summary files
echo "dataset,N,K,threads,iters,status,assignment_s,update_s,total_s" > "${CPU_SUMMARY}"
echo "dataset,N,K,iters,status,h2d_ms,kernel_ms,d2h_ms,total_ms,total_s" > "${GPU_SUMMARY}"

echo "Building executables..."
make clean
make

echo "Generating datasets if needed..."
for N in ${DATA_SIZES}; do
    FILE="${DATA_DIR}/data_${N}.bin"

    if [ ! -f "${FILE}" ]; then
        echo "Creating ${FILE}"
        ${GEN_EXE} "${N}" "${FILE}"
    else
        echo "Found ${FILE} (skipping generation)"
    fi
done

if [ "$TARGET" = "cpu" ]; then
    echo
    echo "CPU Experiments"

    # Always overwrite previous results
    rm -rf "${CPU_RAW_DIR}" "${CPU_PROFILE_DIR}"
    mkdir -p "${CPU_RAW_DIR}" "${CPU_PROFILE_DIR}"


    for N in ${DATA_SIZES}; do
        DATASET="${DATA_DIR}/data_${N}.bin"

        for T in ${THREADS_LIST}; do
            OUTFILE="${CPU_RAW_DIR}/cpu_N${N}_T${T}.txt"
            echo "Running CPU: N=${N}, threads=${T}"

            export OMP_NUM_THREADS=${T}
            export OMP_PROC_BIND=TRUE
	    export OMP_PLACES=cores

            ${CPU_EXE} "${DATASET}" ${K} ${T} ${MAX_ITERS} | tee "${OUTFILE}"
	    K_FIELD=$(grep 'Clusters (K):' "${OUTFILE}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
	    N_FIELD=$(grep 'Points (N):' "${OUTFILE}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
	    THREAD_FIELD=$(grep 'Threads:' "${OUTFILE}" | awk -F':' '{gsub(/ /,"",$2); print $2}')

	    ITER_FIELD=$(grep 'Iterations:' "${OUTFILE}" | awk '{print $2}')
	    STATUS_FIELD=$(grep 'Iterations:' "${OUTFILE}" | sed -n 's/.*(\(.*\)).*/\1/p')

	    ASSIGN_FIELD=$(grep 'Assignment:' "${OUTFILE}" | awk '{print $2}')
	    UPDATE_FIELD=$(grep 'Update:' "${OUTFILE}" | awk '{print $2}')
	    TOTAL_FIELD=$(grep 'Total wall:' "${OUTFILE}" | awk '{print $3}')

	    echo "${DATASET},${N_FIELD},${K_FIELD},${THREAD_FIELD},${ITER_FIELD},${STATUS_FIELD},${ASSIGN_FIELD},${UPDATE_FIELD},${TOTAL_FIELD}" >> "${CPU_SUMMARY}"
	done
    done

    echo
    echo "VTune Profiling (CPU)"
    module load intel/vtune-2021.2.0 >/dev/null 2>&1 || true

    if command -v vtune >/dev/null 2>&1; then
        VTUNE_DIR="${CPU_PROFILE_DIR}/vtune_cpu"
        rm -rf "${VTUNE_DIR}"

        export OMP_NUM_THREADS=28
        export OMP_PROC_BIND=TRUE

        vtune -collect hotspots -result-dir "${VTUNE_DIR}" \
	    ${CPU_EXE} "${DATA_DIR}/data_1000000.bin" ${K} 28 ${MAX_ITERS}
        vtune -report summary -result-dir "${VTUNE_DIR}" \
            > "${CPU_PROFILE_DIR}/vtune_summary.txt" || true

        echo "VTune results saved to ${CPU_PROFILE_DIR}/vtune_summary.txt"
    else
        echo "VTune not found"
    fi

    echo
    echo "CPU summary written to ${CPU_SUMMARY}"
fi

if [ "$TARGET" = "gpu" ]; then
    echo
    echo "GPU Experiments"

    # Always overwrite previous results
    rm -rf "${GPU_RAW_DIR}" "${GPU_PROFILE_DIR}"
    mkdir -p "${GPU_RAW_DIR}" "${GPU_PROFILE_DIR}"

    for N in ${DATA_SIZES}; do
        DATASET="${DATA_DIR}/data_${N}.bin"
        OUTFILE="${GPU_RAW_DIR}/gpu_N${N}.txt"

        echo "Running GPU: N=${N}"
        ${GPU_EXE} "${DATASET}" ${K} ${MAX_ITERS} | tee "${OUTFILE}"

        K_FIELD=$(grep '^  Clusters (K):' "${OUTFILE}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
        N_FIELD=$(grep '^  Points (N):' "${OUTFILE}" | awk -F':' '{gsub(/ /,"",$2); print $2}')
        ITER_FIELD=$(grep '^  Iterations:' "${OUTFILE}" | awk -F':' '{print $2}' | awk '{print $1}')
        STATUS_FIELD=$(grep '^  Iterations:' "${OUTFILE}" | sed -n 's/.*(\(.*\)).*/\1/p')
        H2D_FIELD=$(grep '^H2D transfer:' "${OUTFILE}" | awk '{print $3}')
        KERNEL_FIELD=$(grep '^Kernel exec:' "${OUTFILE}" | awk '{print $3}')
        D2H_FIELD=$(grep '^D2H transfer:' "${OUTFILE}" | awk '{print $3}')
        TOTAL_MS_FIELD=$(grep '^Total GPU:' "${OUTFILE}" | awk '{print $3}')
        TOTAL_S_FIELD=$(grep '^Total GPU:' "${OUTFILE}" | sed -n 's/.*(\(.*\) s).*/\1/p')

        echo "${DATASET},${N_FIELD},${K_FIELD},${ITER_FIELD},${STATUS_FIELD},${H2D_FIELD},${KERNEL_FIELD},${D2H_FIELD},${TOTAL_MS_FIELD},${TOTAL_S_FIELD}" >> "${GPU_SUMMARY}"
    done

    echo
    echo "Nsight Systems Profiling (GPU)"

    module load cuda/12.1.1 >/dev/null 2>&1 || true
    module load nvidia-hpc-sdk/24.7 >/dev/null 2>&1 || true

    NSYS_OUT="${GPU_PROFILE_DIR}/nsys_gpu"

    rm -f ${NSYS_OUT}*

    nsys profile --trace=cuda,osrt -o ${NSYS_OUT} \
        ${GPU_EXE} "${DATA_DIR}/data_1000000.bin" ${K} ${MAX_ITERS}

    echo
    echo "Nsight Summary"

   nsys stats --report cuda_api_sum results/gpu_profiles/nsys_gpu.nsys-rep > results/gpu_profiles/nsys_api.txt

   nsys stats --report cuda_gpu_kern_sum results/gpu_profiles/nsys_gpu.nsys-rep > results/gpu_profiles/nsys_kernels.txt

   nsys stats --report cuda_gpu_mem_time_sum results/gpu_profiles/nsys_gpu.nsys-rep > results/gpu_profiles/nsys_mem.txt 
fi

echo
echo "Done."
