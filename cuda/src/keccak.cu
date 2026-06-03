// the extern "C" functions are the ones python loads through ctypes
//   keccak_hash_batch   : hash a batch of equal-length inputs in parallel
//   merkle_build_keccak : build a sha-3 merkle tree, return the 32-byte root
#include "keccak.cuh"

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


#define CUDA_OK(call) do {                                                  \
    cudaError_t _e = (call);                                                \
    if (_e != cudaSuccess) {                                                \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                           \
                __FILE__, __LINE__, cudaGetErrorString(_e));                \
        std::exit(1);                                                       \
    }                                                                       \
} while (0)


// one thread per hash
__global__ void keccak_hash_kernel(const uint8_t* __restrict__ inputs,
                                   size_t input_len,
                                   size_t n,
                                   uint8_t* __restrict__ outputs) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const uint8_t* in_ptr = inputs + tid * input_len;
    uint8_t* out_ptr = outputs + tid * KECCAK_OUTPUT_BYTES;
    sha3_256_device(in_ptr, input_len, out_ptr);
}

// one thread per parent node: hashes two 32-byte children into the parent
__global__ void merkle_keccak_level_kernel(const uint8_t* __restrict__ in_level,
                                           uint8_t* __restrict__ out_level,
                                           size_t pairs) {
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= pairs) return;

    const uint8_t* left = in_level + (2 * tid) * KECCAK_OUTPUT_BYTES;
    sha3_256_device(left, 2 * KECCAK_OUTPUT_BYTES, out_level + tid * KECCAK_OUTPUT_BYTES);
}


// hash a batch of equal-length inputs on the gpu and copy the results back
extern "C" void keccak_hash_batch(const uint8_t* inputs,
                                  size_t input_len,
                                  size_t n,
                                  uint8_t* outputs) {
    if (n == 0) return;

    uint8_t* d_in;
    uint8_t* d_out;
    size_t in_bytes  = n * input_len;
    size_t out_bytes = n * KECCAK_OUTPUT_BYTES;
    CUDA_OK(cudaMalloc(&d_in,  in_bytes));
    CUDA_OK(cudaMalloc(&d_out, out_bytes));
    CUDA_OK(cudaMemcpy(d_in, inputs, in_bytes, cudaMemcpyHostToDevice));

    const int threads = 128;
    int blocks = (int)((n + threads - 1) / threads);
    keccak_hash_kernel<<<blocks, threads>>>(d_in, input_len, n, d_out);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(outputs, d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_in));
    CUDA_OK(cudaFree(d_out));
}


// build the whole merkle tree on the gpu and return the root
extern "C" void merkle_build_keccak(const uint8_t* leaves,
                                    size_t leaf_len,
                                    size_t n_leaves,
                                    uint8_t* root) {
    if (n_leaves == 0) return;

    // round the leaf count up to a power of two, pad slots stay zeroed
    size_t n_padded = 1;
    while (n_padded < n_leaves) n_padded <<= 1;

    // two buffers, ping-pong between them one level at a time
    uint8_t* d_a;
    uint8_t* d_b;
    size_t buf_bytes = n_padded * KECCAK_OUTPUT_BYTES;
    CUDA_OK(cudaMalloc(&d_a, buf_bytes));
    CUDA_OK(cudaMalloc(&d_b, buf_bytes));

    uint8_t* d_leaves;
    size_t leaves_bytes = n_leaves * leaf_len;
    CUDA_OK(cudaMalloc(&d_leaves, leaves_bytes));
    CUDA_OK(cudaMemcpy(d_leaves, leaves, leaves_bytes, cudaMemcpyHostToDevice));

    // hash each real leaf into d_a, leave padding slots as 0
    const int threads = 128;
    int blocks = (int)((n_leaves + threads - 1) / threads);
    keccak_hash_kernel<<<blocks, threads>>>(d_leaves, leaf_len, n_leaves, d_a);
    CUDA_OK(cudaGetLastError());

    if (n_padded > n_leaves) {
        CUDA_OK(cudaMemset(
            d_a + n_leaves * KECCAK_OUTPUT_BYTES, 0,
            (n_padded - n_leaves) * KECCAK_OUTPUT_BYTES));
    }

    // build up the tree, each level reads one buffer and writes the other
    uint8_t* in_buf  = d_a;
    uint8_t* out_buf = d_b;
    size_t count = n_padded;
    while (count > 1) {
        size_t pairs = count >> 1;
        int blocks2 = (int)((pairs + threads - 1) / threads);
        merkle_keccak_level_kernel<<<blocks2, threads>>>(in_buf, out_buf, pairs);
        CUDA_OK(cudaGetLastError());
        uint8_t* tmp = in_buf; in_buf = out_buf; out_buf = tmp;
        count = pairs;
    }
    CUDA_OK(cudaDeviceSynchronize());

    CUDA_OK(cudaMemcpy(root, in_buf, KECCAK_OUTPUT_BYTES, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_a));
    CUDA_OK(cudaFree(d_b));
    CUDA_OK(cudaFree(d_leaves));
}
