"""poseidon hash over the BN254 scalar field.

poseidon is an algebraic hash. instead of bit ops like sha-3 it does add and
multiply on field elements mod a big prime. its built from rounds: add the
round constants, run the x^5 sbox, then mix the state with a matrix.

round constants come from the Grain LFSR (the poseidon paper specifies this).
the mix step uses a Cauchy MDS matrix.

params:
    t      = 3       state width
    alpha  = 5       sbox exponent (x^5)
    R_F    = 8       full rounds, 4 before and 4 after the partial rounds
    R_P    = 57      partial rounds
    field  = 254-bit BN254 prime
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
    # write a number out as a list of bits, most significant bit first
    return [(value >> (width - 1 - i)) & 1 for i in range(width)]


class _Grain:
    """80-bit LFSR used by the Poseidon reference to derive round constants

    the starting state packs the poseidon params into 80 bits:
        2 bits  : field type    (1 = GF(p))
        4 bits  : S-box type    (0 = x^alpha)
        12 bits : log2(p)       (254 here)
        12 bits : state width t
        10 bits : R_F
        10 bits : R_P
        30 bits : all ones
    """

    # pack the params into the 80-bit state and run 160 warm-up steps
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
        for _ in range(160):  # warm-up, these bits get thrown away
            self.step()

    def step(self) -> int:
        # advance the LFSR by one bit (xor of 6 taps)
        s = self.state
        new_bit = s[0] ^ s[13] ^ s[23] ^ s[38] ^ s[51] ^ s[62]
        self.state = s[1:] + [new_bit]
        return new_bit

    def output_bit(self) -> int:
        # step twice, the first bit decides whether we keep the second
        while True:
            sel = self.step()
            cand = self.step()
            if sel == 1:
                return cand

    def next_field_element(self) -> int:
        # pull 254 bits into a number, retry if it came out >= p
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


# public aliases used by bindings/cuda_poseidon.py
build_round_constants = _build_round_constants


def _build_mds() -> list[list[int]]:
    """Cauchy MDS matrix: M[i][j] = (x_i + y_j)^{-1} mod p.

    Disjoint integer sets {x_i} and {y_j} guarantee nonzero pairwise sums in
    F_p (each sum is a small positive integer well below p), so every entry
    and every submatrix determinant is well-defined and nonzero.
    """
    xs = [i for i in range(T)]
    ys = [T + i for i in range(T)]
    return [
        [pow((xs[i] + ys[j]) % FIELD_P, -1, FIELD_P) for j in range(T)]
        for i in range(T)
    ]


# public alias used by bindings/cuda_poseidon.py
build_mds = _build_mds


def ensure() -> tuple[list[int], list[list[int]]]:
    # build the constants on first use, then reuse them after that
    global _ROUND_CONSTANTS, _MDS
    if _ROUND_CONSTANTS is None:
        _ROUND_CONSTANTS = _build_round_constants()
        _MDS = _build_mds()
    return _ROUND_CONSTANTS, _MDS  # type: ignore[return-value]


#========permutation==========

def sbox(x: int) -> int:
    # the sbox: x^5 mod p
    return pow(x, ALPHA, FIELD_P)


def add_round_constants(state: list[int], rc: list[int], offset: int) -> list[int]:
    # add this round's constants to each state element
    return [(state[i] + rc[offset + i]) % FIELD_P for i in range(T)]


def mix(state: list[int], mds: list[list[int]]) -> list[int]:
    # multiply the state by the MDS matrix
    return [
        sum(mds[i][j] * state[j] for j in range(T)) % FIELD_P
        for i in range(T)
    ]


def poseidon_permutation(state: list[int]) -> list[int]:
    # the full permutation: full rounds, then partial rounds, then full rounds.
    # full rounds sbox every element, partial rounds only sbox element 0
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


#========sponge hash==========

def poseidon_hash(inputs: list[int]) -> int:
    # hash a list of field elements down to one. the input length goes in the
    # capacity slot so inputs of different lengths cant collide. absorb RATE
    # elements at a time, run the permutation between blocks
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
