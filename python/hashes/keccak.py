"""SHA-3 (Keccak) hash function implemented from scratch.

Follows the NIST FIPS 202 standard. The core is the Keccak-f[1600]
permutation applied inside a sponge construction.
"""
from __future__ import annotations

_RHO_OFFSETS = [
    [0,  36,  3, 41, 18],
    [1,  44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14],
]

_ROUND_CONSTANTS = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
]

MASK64 = 0xFFFFFFFFFFFFFFFF


def _rot64(x: int, n: int) -> int:
    """Rotate a 64-bit integer left by n bits."""
    return ((x << n) | (x >> (64 - n))) & MASK64


def _keccak_f(state: list[list[int]]) -> list[list[int]]:
    """Apply 24 rounds of the Keccak-f[1600] permutation to a 5x5 state."""
    for rc in _ROUND_CONSTANTS:
        # theta: XOR each lane with parity of two neighbouring columns
        C = [state[x][0] ^ state[x][1] ^ state[x][2] ^ state[x][3] ^ state[x][4]
             for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rot64(C[(x + 1) % 5], 1) for x in range(5)]
        state = [[state[x][y] ^ D[x] for y in range(5)] for x in range(5)]

        # rho + pi: rotate each lane then move it to a new position
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rot64(state[x][y], _RHO_OFFSETS[x][y])

        # chi: non-linear step mixing each row
        state = [[B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])
                  for y in range(5)] for x in range(5)]

        # iota: XOR round constant into lane [0][0] to break symmetry
        state[0][0] ^= rc

    return state


def _absorb(rate_bytes: int, suffix: int, message: bytes) -> list[list[int]]:
    """Pad the message and absorb it into a fresh Keccak state."""
    # pad10*1 padding: append suffix byte, zero pad, set last bit
    msg = bytearray(message)
    msg.append(suffix)
    while len(msg) % rate_bytes != 0:
        msg.append(0x00)
    msg[-1] |= 0x80

    # initialise 5x5 state of 64-bit lanes to zero
    state = [[0] * 5 for _ in range(5)]

    # absorb each rate-sized block by XORing into the state
    for block_start in range(0, len(msg), rate_bytes):
        block = msg[block_start:block_start + rate_bytes]
        # convert bytes to 64-bit lanes and XOR into state
        for i in range(rate_bytes // 8):
            x, y = i % 5, i // 5
            lane = int.from_bytes(block[i*8:(i+1)*8], 'little')
            state[x][y] ^= lane
        state = _keccak_f(state)

    return state


def _squeeze(state: list[list[int]], output_bytes: int) -> bytes:
    """Read output_bytes from the state after absorption."""
    out = bytearray()
    for i in range(output_bytes // 8):
        x, y = i % 5, i // 5
        out += state[x][y].to_bytes(8, 'little')
    return bytes(out[:output_bytes])


def sha3_256(message: bytes) -> bytes:
    """Compute the SHA3-256 hash of message. Returns 32 bytes.
    
    Rate = 1088 bits = 136 bytes (1600 - 2*256 capacity bits)
    Output = 256 bits = 32 bytes
    Suffix = 0x06 per FIPS 202 domain separation for SHA-3
    """
    state = _absorb(rate_bytes=136, suffix=0x06, message=message)
    return _squeeze(state, output_bytes=32)


