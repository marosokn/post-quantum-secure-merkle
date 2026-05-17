"""Poseidon hash function over the BN254 scalar field.

Implementation follows the Poseidon paper (Grassi et. al.). Round constants are generated via the Grain 
LFSR as specified in paper's reference parameter generation script. The mixing layer uses a 
Cauchy MDS matrix over the field — Cauchy matrices are provably MDS for any choice of distinct {x_i} and 
{y_j} with nonzero pairwise sums, which is the property Poseidon's security argument relies on.

Parameters (instance: rate=2, capacity=1, BN254 scalar field):
    t      = 3       state width
    alpha  = 5       S-box exponent
    R_F    = 8       full rounds (4 before / 4 after the partial section)
    R_P    = 57      partial rounds
    |F_p|  = 254 bit BN254 scalar field
"""
from __future__ import annotations

# BN254 scalar field modulus (https://docs.rs/ark-bn254/latest/ark_bn254/)
FIELD_P = 21888242871839275222246405745257275088548364400416034343698204186575808495617

T = 3
ALPHA = 5
R_F = 8
R_P = 57
FIELD_BITS = 254
RATE = T - 1


#=========grain LFSR========== 

def to_bits(value: int, width: int) -> list[int]:
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


class _Grain:
    """80-bit LFSR used by the Poseidon reference to derive round constants

    Init vector layout:
        2 bits  : field type    (1 = GF(p))
        4 bits  : S-box type    (0 = x^alpha)
        12 bits : log2(p)       (254 here)
        12 bits : state width t
        10 bits : R_F
        10 bits : R_P
        30 bits : all ones
    """

    def __init__(self, field: int = 1, sbox: int = 0):
        bits: list[int] = []
        bits += to_bits(field, 2)
        bits += to_bits(sbox, 4)
        bits += to_bits(FIELD_BITS, 12)
        bits += to_bits(T, 12)
        bits += to_bits(R_F, 10)
        bits += to_bits(R_P, 10)
        bits += [1] * 30
        assert len(bits) == 80
        self.state = bits
        for _ in range(160):  # warm-up; output discarded
            self.step()

    def step(self) -> int:
        s = self.state
        new_bit = s[0] ^ s[13] ^ s[23] ^ s[38] ^ s[51] ^ s[62]
        self.state = s[1:] + [new_bit]
        return new_bit

    def output_bit(self) -> int:
        # Rejectionsample one output bit from the LFSR by stepping twice and returning the second output.
        while True:
            sel = self.step()
            cand = self.step()
            if sel == 1:
                return cand

    def next_field_element(self) -> int:
        # Rejection sample one field element from F_p by stepping the LFSR.
        while True:
            v = 0
            for _ in range(FIELD_BITS):
                v = (v << 1) | self.output_bit()
            if v < FIELD_P:
                return v


#========Round constants and MDS matrix==========

_ROUND_CONSTANTS: list[int] | None = None
_MDS: list[list[int]] | None = None


def _build_round_constants() -> list[int]:
    grain = _Grain()
    return [grain.next_field_element() for _ in range((R_F + R_P) * T)]


def _build_mds() -> list[list[int]]:
    """Cauchy MDS matrix: M[i][j] = (x_i + y_j)^{-1} mod p.

    Disjoint integer sets {x_i} and {y_j} guarantee nonzero pairwise sums in
    F_p (each sum is a small positive integer well below p), so every entry
    and every submatrix determinant is well-defined and nonzero 
    """
    xs = [i for i in range(T)]
    ys = [T + i for i in range(T)]
    return [
        [pow((xs[i] + ys[j]) % FIELD_P, -1, FIELD_P) for j in range(T)]
        for i in range(T)
    ]


def ensure() -> tuple[list[int], list[list[int]]]:
    global _ROUND_CONSTANTS, _MDS
    if _ROUND_CONSTANTS is None:
        _ROUND_CONSTANTS = _build_round_constants()
        _MDS = _build_mds()
    return _ROUND_CONSTANTS, _MDS  # type: ignore[return-value]


#========Permutation==========

def sbox(x: int) -> int:
    return pow(x, ALPHA, FIELD_P)


def add_round_constants(state: list[int], rc: list[int], offset: int) -> list[int]:
    return [(state[i] + rc[offset + i]) % FIELD_P for i in range(T)]


def mix(state: list[int], mds: list[list[int]]) -> list[int]:
    return [
        sum(mds[i][j] * state[j] for j in range(T)) % FIELD_P
        for i in range(T)
    ]


def poseidon_permutation(state: list[int]) -> list[int]:
    """Apply the Poseidon permutation to a width-`T` state.

    Structure: R_F/2 full rounds -> R_P partial rounds -> R_F/2 full rounds.
    A full round applies the S-box to every element; a partial round applies
    it only to element 0.
    """
    if len(state) != T:
        raise ValueError(f"state must have length {T}, got {len(state)}")
    rc, mds = ensure()
    s = [x % FIELD_P for x in state]
    half = R_F // 2
    offset = 0

    for _ in range(half):
        s = add_round_constants(s, rc, offset); offset += T
        s = [sbox(x) for x in s]
        s = mix(s, mds)

    for _ in range(R_P):
        s = add_round_constants(s, rc, offset); offset += T
        s[0] = sbox(s[0])
        s =     mix(s, mds)

    for _ in range(half):
        s = add_round_constants(s, rc, offset); offset += T
        s = [sbox(x) for x in s]
        s = mix(s, mds)

    return s


#========Sponge hash==========

def poseidon_hash(inputs: list[int]) -> int:
    """Sponge-based hash producing one field element.

    The capacity slot is preloaded with `len(inputs)` so that inputs of
    different lengths cannot collide trivially (length domain separation).
    Inputs are absorbed `RATE` elements at a time by field addition into
    the rate slots, with a permutation between blocks.
    """
    state = [0] * T
    state[RATE] = len(inputs)
    if not inputs:
        return poseidon_permutation(state)[0]

    for off in range(0, len(inputs), RATE):
        chunk = inputs[off:off + RATE]
        for i, x in enumerate(chunk):
            state[i] = (state[i] + (x % FIELD_P)) % FIELD_P
        state = poseidon_permutation(state)

    return state[0]
