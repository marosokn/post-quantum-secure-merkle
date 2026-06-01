// the cpu-callable side of the gpu poseidon code. the hashing math itself is
// in poseidon.cuh. this file allocates gpu memory, launches the kernels, and
// builds the merkle tree level by level.
//
// the extern "C" functions are the ones python loads through ctypes
// (see bindings/cuda_poseidon.py):
//   poseidon_load_constants : upload the round constants + MDS to the gpu
//   poseidon_hash_batch     : hash a batch of inputs in parallel
//   merkle_build_poseidon   : build a poseidon merkle tree, return the root
#include "poseidon.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


// if a cuda call returned an error, print where it happened and exit
#define CUDA_OK(call) do {                                                  \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)


// one thread per hash
__global__ void poseidon_hash_kernel(const uint64_t* __restrict__ inputs,
                                     int inputs_per_hash,
                                     size_t n,
                                     uint64_t* __restrict__ outputs) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const uint64_t* in_ptr = inputs + tid * (size_t)inputs_per_hash * FP_LIMBS;
    uint64_t* out_ptr = outputs + tid * FP_LIMBS;
    poseidon_hash_device(in_ptr, inputs_per_hash, out_ptr);
}

// one thread per parent node: hashes its two children into the parent
__global__ void merkle_level_kernel(const uint64_t* __restrict__ in_level,
                                    uint64_t* __restrict__ out_level,
                                    size_t pairs) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pairs) return;

    // copy the two children side by side so the hash sees one contiguous input
    uint64_t pair[2 * FP_LIMBS];
    const uint64_t* left  = in_level + (2 * tid)     * FP_LIMBS;
    const uint64_t* right = in_level + (2 * tid + 1) * FP_LIMBS;
    #pragma unroll
    for (int k = 0; k < FP_LIMBS; k++) pair[k]            = left[k];
    #pragma unroll
    for (int k = 0; k < FP_LIMBS; k++) pair[FP_LIMBS + k] = right[k];

    poseidon_hash_device(pair, 2, out_level + tid * FP_LIMBS);
}


// upload the round constants and MDS matrix to the gpu (call once at startup)
extern "C" void poseidon_load_constants(const uint64_t* rc_mont,
                                        const uint64_t* mds_mont) {
    CUDA_OK(cudaMemcpyToSymbol(
        d_RC, rc_mont,
        POSEIDON_TOTAL_RC * FP_LIMBS * sizeof(uint64_t)));
    CUDA_OK(cudaMemcpyToSymbol(
        d_MDS, mds_mont,
        POSEIDON_T * POSEIDON_T * FP_LIMBS * sizeof(uint64_t)));
}

// hash a batch of inputs on the gpu and copy the results back
extern "C" void poseidon_hash_batch(const uint64_t* inputs,
                                    int inputs_per_hash,
                                    size_t n,
                                    uint64_t* outputs) {
    if (n == 0) return;

    uint64_t* d_in;
    uint64_t* d_out;
    size_t in_bytes  = n * (size_t)inputs_per_hash * FP_LIMBS * sizeof(uint64_t);
    size_t out_bytes = n * FP_LIMBS * sizeof(uint64_t);
    CUDA_OK(cudaMalloc(&d_in,  in_bytes));
    CUDA_OK(cudaMalloc(&d_out, out_bytes));
    CUDA_OK(cudaMemcpy(d_in, inputs, in_bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    int blocks = (int)((n + threads - 1) / threads);
    poseidon_hash_kernel<<<blocks, threads>>>(d_in, inputs_per_hash, n, d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(outputs, d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_out));
}


// build the whole merkle tree on the gpu and return the root
extern "C" void merkle_build_poseidon(const uint64_t* leaves_canon,
                                      size_t n_leaves,
                                      uint64_t* root_canon) {
    if (n_leaves == 0) return;

    // round the leaf count up to a power of two, padded slots are 0
    size_t n_padded = 1;
    while (n_padded < n_leaves) n_padded <<= 1;

    // two buffers, ping-pong between them one level at a time
    uint64_t* d_a;
    uint64_t* d_b;
    size_t buf_bytes = n_padded * FP_LIMBS * sizeof(uint64_t);
    CUDA_OK(cudaMalloc(&d_a, buf_bytes));
    CUDA_OK(cudaMalloc(&d_b, buf_bytes));

    // load the raw leaves into d_a
    CUDA_OK(cudaMemcpy(d_a, leaves_canon,
                       n_leaves * FP_LIMBS * sizeof(uint64_t),
                       cudaMemcpyHostToDevice));

    // hash each real leaf into d_b, leave the padding slots as 0
    const int threads = 128;
    int blocks = (int)((n_leaves + threads - 1) / threads);
    poseidon_hash_kernel<<<blocks, threads>>>(d_a, 1, n_leaves, d_b);
    CUDA_OK(cudaGetLastError());

    if (n_padded > n_leaves) {
        CUDA_OK(cudaMemset(
            d_b + n_leaves * FP_LIMBS, 0,
            (n_padded - n_leaves) * FP_LIMBS * sizeof(uint64_t)));
    }

    // build up the tree, each level reads one buffer and writes the other
    uint64_t* in_buf  = d_b;
    uint64_t* out_buf = d_a;
    size_t count = n_padded;
    while (count > 1) {
        size_t pairs = count >> 1;
        int blocks2 = (int)((pairs + threads - 1) / threads);
        merkle_level_kernel<<<blocks2, threads>>>(in_buf, out_buf, pairs);
        CUDA_OK(cudaGetLastError());
        uint64_t* tmp = in_buf; in_buf = out_buf; out_buf = tmp;
        count = pairs;
    }
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(root_canon, in_buf,
                       FP_LIMBS * sizeof(uint64_t),
                       cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));
}
