// the sha-3 (keccak) hash, gpu version. same algorithm as
// python/hashes/keccak.py, with the same outputs as the python reference.
//
// params (sha3-256):
//     rate   = 136 bytes (1088 bits)
//     output = 32 bytes  (256 bits)
//     suffix = 0x06      (FIPS 202 sha-3 domain separation)
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define KECCAK_RATE_BYTES   136
#define KECCAK_OUTPUT_BYTES 32
#define KECCAK_ROUNDS       24


// round constants, xored into lane [0][0] each round to break symmetry
__constant__ static uint64_t d_KECCAK_RC[KECCAK_ROUNDS] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL,
};

// rotation offsets for the rho step
__constant__ static int d_KECCAK_RHO[5][5] = {
    { 0, 36,  3, 41, 18},
    { 1, 44, 10, 45,  2},
    {62,  6, 43, 15, 61},
    {28, 55, 25, 21, 56},
    {27, 20, 39,  8, 14},
};


// rotate a 64-bit value left by n bits. (64-n)&63 keeps n=0 safe in C.
__device__ __forceinline__
uint64_t keccak_rot64(uint64_t x, int n) {
    return (x << n) | (x >> ((64 - n) & 63));
}


// the keccak-f[1600] permutation: 24 rounds of theta, rho+pi, chi, iota
__device__ __forceinline__
void keccak_f1600(uint64_t state[5][5]) {
    #pragma unroll
    for (int round = 0; round < KECCAK_ROUNDS; round++) {
        // theta
        uint64_t C[5], D[5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            C[x] = state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4];
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ keccak_rot64(C[(x + 1) % 5], 1);
        }
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                state[x][y] ^= D[x];
            }
        }

        // rho + pi
        uint64_t B[5][5];
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                B[y][(2 * x + 3 * y) % 5] = keccak_rot64(state[x][y], d_KECCAK_RHO[x][y]);
            }
        }

        // chi (the only non-linear step)
        #pragma unroll
        for (int x = 0; x < 5; x++) {
            #pragma unroll
            for (int y = 0; y < 5; y++) {
                state[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y]);
            }
        }

        // iota
        state[0][0] ^= d_KECCAK_RC[round];
    }
}


// sha3-256 hash of one input: absorb input_len bytes, squeeze 32 bytes out
__device__ __forceinline__
void sha3_256_device(const uint8_t* input, size_t input_len, uint8_t* output) {
    uint64_t state[5][5];
    #pragma unroll
    for (int i = 0; i < 5; i++) {
        #pragma unroll
        for (int j = 0; j < 5; j++) {
            state[i][j] = 0;
        }
    }

    // absorb full rate-sized blocks, one byte at a time into the lane it
    // belongs to. avoids needing a scratch buffer.
    size_t pos = 0;
    while (input_len - pos >= KECCAK_RATE_BYTES) {
        for (int i = 0; i < KECCAK_RATE_BYTES; i++) {
            int lane_idx = i / 8;
            int x = lane_idx % 5;
            int y = lane_idx / 5;
            int byte_idx = i % 8;
            state[x][y] ^= ((uint64_t)input[pos + i]) << (byte_idx * 8);
        }
        keccak_f1600(state);
        pos += KECCAK_RATE_BYTES;
    }

    // absorb the leftover bytes
    size_t remaining = input_len - pos;
    for (size_t i = 0; i < remaining; i++) {
        int lane_idx = i / 8;
        int x = lane_idx % 5;
        int y = lane_idx / 5;
        int byte_idx = i % 8;
        state[x][y] ^= ((uint64_t)input[pos + i]) << (byte_idx * 8);
    }

    // pad: 0x06 right after the input, 0x80 in the last byte of the rate
    {
        int lane_idx = (int)(remaining / 8);
        int x = lane_idx % 5;
        int y = lane_idx / 5;
        int byte_idx = (int)(remaining % 8);
        state[x][y] ^= ((uint64_t)0x06) << (byte_idx * 8);
    }
    {
        int last = KECCAK_RATE_BYTES - 1;
        int lane_idx = last / 8;
        int x = lane_idx % 5;
        int y = lane_idx / 5;
        int byte_idx = last % 8;
        state[x][y] ^= ((uint64_t)0x80) << (byte_idx * 8);
    }
    keccak_f1600(state);

    // squeeze 32 bytes (4 lanes) out, little-endian
    for (int i = 0; i < KECCAK_OUTPUT_BYTES; i++) {
        int lane_idx = i / 8;
        int x = lane_idx % 5;
        int y = lane_idx / 5;
        int byte_idx = i % 8;
        output[i] = (uint8_t)(state[x][y] >> (byte_idx * 8));
    }
}
