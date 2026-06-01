// the poseidon hash, gpu version. this is the same algorithm as
// python/hashes/poseidon.py, written for the gpu, and it produces the exact
// same outputs as the python reference.
//
// everything here works in Montgomery form (see fp_bn254.cuh). the round
// constants and MDS matrix are converted to Montgomery form on the cpu and
// copied into __constant__ memory once at startup. the hash converts its
// inputs into Montgomery form on the way in and back out on the way out, so
// callers only ever deal with normal field elements.
//
// params (same as the python file):
//     t     = 3    state width
//     alpha = 5    sbox is x^5
//     R_F   = 8    full rounds, 4 at the start and 4 at the end
//     R_P   = 57   partial rounds in the middle
#pragma once

#include "fp_bn254.cuh"

#define POSEIDON_T        3
#define POSEIDON_R_F      8
#define POSEIDON_R_P      57
#define POSEIDON_RATE     (POSEIDON_T - 1)
#define POSEIDON_TOTAL_RC ((POSEIDON_R_F + POSEIDON_R_P) * POSEIDON_T)

// round constants and MDS matrix, uploaded by the cpu, already in Montgomery
// form. d_RC is one flat list, d_MDS is a flat 3x3 matrix.
__constant__ uint64_t d_RC[POSEIDON_TOTAL_RC * FP_LIMBS];
__constant__ uint64_t d_MDS[POSEIDON_T * POSEIDON_T * FP_LIMBS];


// add this round's constants to each state element
__device__ __forceinline__
void poseidon_add_round_constants(uint64_t state[POSEIDON_T][FP_LIMBS], int round_idx) {
    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) {
        uint64_t tmp[FP_LIMBS];
        const uint64_t* rc = &d_RC[(round_idx * POSEIDON_T + i) * FP_LIMBS];
        fp_add(state[i], rc, tmp);
        fp_copy(tmp, state[i]);
    }
}

// run the x^5 sbox on every state element
__device__ __forceinline__
void poseidon_full_sbox(uint64_t state[POSEIDON_T][FP_LIMBS]) {
    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) {
        uint64_t tmp[FP_LIMBS];
        fp_pow5(state[i], tmp);
        fp_copy(tmp, state[i]);
    }
}

// run the x^5 sbox on just element 0
__device__ __forceinline__
void poseidon_partial_sbox(uint64_t state[POSEIDON_T][FP_LIMBS]) {
    uint64_t tmp[FP_LIMBS];
    fp_pow5(state[0], tmp);
    fp_copy(tmp, state[0]);
}

// mix the state by multiplying it with the MDS matrix
__device__ __forceinline__
void poseidon_mix(uint64_t state[POSEIDON_T][FP_LIMBS]) {
    uint64_t out[POSEIDON_T][FP_LIMBS];
    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) {
        #pragma unroll
        for (int k = 0; k < FP_LIMBS; k++) out[i][k] = 0;
        #pragma unroll
        for (int j = 0; j < POSEIDON_T; j++) {
            const uint64_t* m = &d_MDS[(i * POSEIDON_T + j) * FP_LIMBS];
            uint64_t prod[FP_LIMBS];
            fp_mont_mul(m, state[j], prod);
            uint64_t sum[FP_LIMBS];
            fp_add(out[i], prod, sum);
            fp_copy(sum, out[i]);
        }
    }
    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) {
        fp_copy(out[i], state[i]);
    }
}

// the full permutation: full rounds, then partial rounds, then full rounds
__device__ __forceinline__
void poseidon_permutation(uint64_t state[POSEIDON_T][FP_LIMBS]) {
    const int half = POSEIDON_R_F / 2;
    int round = 0;

    #pragma unroll
    for (int r = 0; r < half; r++) {
        poseidon_add_round_constants(state, round++);
        poseidon_full_sbox(state);
        poseidon_mix(state);
    }
    for (int r = 0; r < POSEIDON_R_P; r++) {
        poseidon_add_round_constants(state, round++);
        poseidon_partial_sbox(state);
        poseidon_mix(state);
    }
    #pragma unroll
    for (int r = 0; r < half; r++) {
        poseidon_add_round_constants(state, round++);
        poseidon_full_sbox(state);
        poseidon_mix(state);
    }
}

// hash a list of field elements down to one. the input count goes in the
// capacity slot so inputs of different lengths cant collide. inputs and
// output are normal field elements; Montgomery conversion happens here.
__device__ __forceinline__
void poseidon_hash_device(const uint64_t* inputs_canon,
                          int n_inputs,
                          uint64_t* output_canon) {
    uint64_t state[POSEIDON_T][FP_LIMBS];
    #pragma unroll
    for (int i = 0; i < POSEIDON_T; i++) {
        #pragma unroll
        for (int k = 0; k < FP_LIMBS; k++) state[i][k] = 0;
    }
    // capacity slot = number of inputs, converted to Montgomery form
    uint64_t len_canon[FP_LIMBS] = {(uint64_t)n_inputs, 0, 0, 0};
    fp_to_mont(len_canon, state[POSEIDON_RATE]);

    if (n_inputs == 0) {
        poseidon_permutation(state);
        fp_from_mont(state[0], output_canon);
        return;
    }

    // absorb RATE inputs at a time, run the permutation between blocks
    for (int off = 0; off < n_inputs; off += POSEIDON_RATE) {
        for (int i = 0; i < POSEIDON_RATE && (off + i) < n_inputs; i++) {
            uint64_t in_mont[FP_LIMBS];
            fp_to_mont(&inputs_canon[(off + i) * FP_LIMBS], in_mont);
            uint64_t sum[FP_LIMBS];
            fp_add(state[i], in_mont, sum);
            fp_copy(sum, state[i]);
        }
        poseidon_permutation(state);
    }

    fp_from_mont(state[0], output_canon);
}
