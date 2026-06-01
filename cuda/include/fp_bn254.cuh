// all the number crunching for the BN254 field lives here. a "field element"
// is just a number mod a big 254-bit prime p. its way too big to fit in one
// int so we keep it as 4 chunks of 64 bits stuck together. the slow part of
// modular math is the divide-to-wrap-around step, so we use montgomery form
// which is a trick that sneaks the wrapping into the multiply so we never
// actually divide. the poseidon constants get flipped into montgomery form
// on the cpu before they ever show up here.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#define FP_LIMBS 4

// the prime p, plus two nums montgomery needs. all chopped into 4
// little chunkfirst pieces. R is just 2^256.
__constant__ static uint64_t d_P[FP_LIMBS] = {
    0x43E1F593F0000001ULL,
    0x2833E84879B97091ULL,
    0xB85045B68181585DULL,
    0x30644E72E131A029ULL,
};

__constant__ static uint64_t d_R2[FP_LIMBS] = {
    0x1BB8E645AE216DA7ULL,
    0x53FE3AB1E35C59E3ULL,
    0x8C49833D53BB8085ULL,
    0x0216D0B17F4E44A5ULL,
};

__constant__ static uint64_t d_N_INV = 0xC2E1F593EFFFFFFFULL;


// multiply two 64-bit numbers, hand back the top and bottom halves
__device__ __forceinline__
void fp_mul64(uint64_t a, uint64_t b, uint64_t* hi, uint64_t* lo) {
    *lo = a * b;
    *hi = __umul64hi(a, b);
}

// add two numbers plus a carry bit, also tells you the new carry
__device__ __forceinline__
uint64_t fp_adc(uint64_t a, uint64_t b, uint64_t cin, uint64_t* cout) {
    uint64_t s = a + b;
    uint64_t c1 = (s < a) ? 1ULL : 0ULL;
    uint64_t s2 = s + cin;
    uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
    *cout = c1 + c2;
    return s2;
}

// subtract two numbers with a borrow bit, also tells you the new borrow
__device__ __forceinline__
uint64_t fp_sbb(uint64_t a, uint64_t b, uint64_t bin, uint64_t* bout) {
    uint64_t d = a - b;
    uint64_t b1 = (a < b) ? 1ULL : 0ULL;
    uint64_t d2 = d - bin;
    uint64_t b2 = (d < bin) ? 1ULL : 0ULL;
    *bout = b1 + b2;
    return d2;
}

// is this number >= p? check from the top chunk down
__device__ __forceinline__
bool fp_ge_p(const uint64_t t[FP_LIMBS]) {
    for (int i = FP_LIMBS - 1; i >= 0; i--) {
        if (t[i] > d_P[i]) return true;
        if (t[i] < d_P[i]) return false;
    }
    return true;  // exactly equal
}

// take p away from the number
__device__ __forceinline__
void fp_sub_p(const uint64_t t[FP_LIMBS], uint64_t r[FP_LIMBS]) {
    uint64_t borrow = 0;
    for (int i = 0; i < FP_LIMBS; i++) {
        r[i] = fp_sbb(t[i], d_P[i], borrow, &borrow);
    }
}

// copy one number into another
__device__ __forceinline__
void fp_copy(const uint64_t a[FP_LIMBS], uint64_t r[FP_LIMBS]) {
    #pragma unroll
    for (int i = 0; i < FP_LIMBS; i++) r[i] = a[i];
}

// add two field numbers, then wrap back down if it spilled over p
__device__ __forceinline__
void fp_add(const uint64_t a[FP_LIMBS], const uint64_t b[FP_LIMBS], uint64_t r[FP_LIMBS]) {
    uint64_t t[FP_LIMBS];
    uint64_t carry = 0;
    #pragma unroll
    for (int i = 0; i < FP_LIMBS; i++) {
        t[i] = fp_adc(a[i], b[i], carry, &carry);
    }
    // even with no carry the sum might still be too big so still check
    if (carry || fp_ge_p(t)) {
        fp_sub_p(t, r);
    } else {
        fp_copy(t, r);
    }
}

// multiply two field numbers the montgomery way. for each chunk of b we add
// a*chunk into a running total, then add a multiple of p that makes
// the bottom chunk zero so we can shift it off. After 4 rounds the total is
// the answer (maybe one p too big) and one subtract cleans it up
__device__ __forceinline__
void fp_mont_mul(const uint64_t a[FP_LIMBS],
                 const uint64_t b[FP_LIMBS],
                 uint64_t r[FP_LIMBS]) {
    uint64_t t[FP_LIMBS + 2] = {0, 0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < FP_LIMBS; i++) {
        // t += a * b[i]
        uint64_t carry = 0;
        #pragma unroll
        for (int j = 0; j < FP_LIMBS; j++) {
            uint64_t hi, lo;
            fp_mul64(a[j], b[i], &hi, &lo);
            uint64_t c1;
            uint64_t s = fp_adc(t[j], lo, carry, &c1);
            t[j] = s;
            uint64_t c2 = 0;
            uint64_t hi_new = fp_adc(hi, c1, 0, &c2);
            carry = hi_new;
        }
        {
            uint64_t c_out;
            t[FP_LIMBS] = fp_adc(t[FP_LIMBS], carry, 0, &c_out);
            t[FP_LIMBS + 1] += c_out;
        }

        // pick the multiple of p that zeroes out the bottom chunk
        uint64_t m = t[0] * d_N_INV;

        // t += m * p, this makes t[0] become 0
        carry = 0;
        #pragma unroll
        for (int j = 0; j < FP_LIMBS; j++) {
            uint64_t hi, lo;
            fp_mul64(m, d_P[j], &hi, &lo);
            uint64_t c1;
            uint64_t s = fp_adc(t[j], lo, carry, &c1);
            t[j] = s;
            uint64_t c2 = 0;
            uint64_t hi_new = fp_adc(hi, c1, 0, &c2);
            carry = hi_new;
        }
        {
            uint64_t c_out;
            t[FP_LIMBS] = fp_adc(t[FP_LIMBS], carry, 0, &c_out);
            t[FP_LIMBS + 1] += c_out;
        }

        // shift the now-zero bottom chunk off
        #pragma unroll
        for (int j = 0; j < FP_LIMBS + 1; j++) {
            t[j] = t[j + 1];
        }
        t[FP_LIMBS + 1] = 0;
    }

    // might be one p too big, fix it
    if (t[FP_LIMBS] || fp_ge_p(t)) {
        fp_sub_p(t, r);
    } else {
        fp_copy(t, r);
    }
}

// push a normal number into montgomery form
__device__ __forceinline__
void fp_to_mont(const uint64_t a[FP_LIMBS], uint64_t r[FP_LIMBS]) {
    fp_mont_mul(a, d_R2, r);
}

// pull a number back out of montgomery form into a normal one
__device__ __forceinline__
void fp_from_mont(const uint64_t a[FP_LIMBS], uint64_t r[FP_LIMBS]) {
    uint64_t one[FP_LIMBS] = {1ULL, 0ULL, 0ULL, 0ULL};
    fp_mont_mul(a, one, r);
}

// raise a number to the 5th power (this is the poseidon sbox)
__device__ __forceinline__
void fp_pow5(const uint64_t x[FP_LIMBS], uint64_t r[FP_LIMBS]) {
    uint64_t x2[FP_LIMBS], x4[FP_LIMBS];
    fp_mont_mul(x, x, x2);
    fp_mont_mul(x2, x2, x4);
    fp_mont_mul(x4, x, r);
}
