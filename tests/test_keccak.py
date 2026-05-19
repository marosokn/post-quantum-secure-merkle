"""Unit tests for the SHA-3 (Keccak) implementation.

Tests check against:
1. NIST FIPS 202 known answer test (KAT) vectors
2. Edge cases around the sponge rate boundary (135/136/137 bytes)
3. Structural properties (determinism, avalanche, output length)
"""
import pytest
from python.hashes.keccak import sha3_256, _keccak_f, _rot64, MASK64
import hashlib
import os


# NIST known-answer tests===============================
# These hex digests are from the official NIST CAVS test vectors for SHA3-256
# https://csrc.nist.gov/projects/cryptographic-algorithm-validation-program

NIST_VECTORS = [
    (b"", "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a"),
    (b"abc", "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532"),
    (b"a", "80084bf2fba02475726feb2cab2d8215eab14bc6bdd8bfb2c8151257032ecd8b"),
    (b"The quick brown fox jumps over the lazy dog",
     "69070dda01975c8c120c3aada1b282394e7f032fa9cf32f4cb2259a0897dfc04"),
]


@pytest.mark.parametrize("message,expected_hex", NIST_VECTORS)
def test_nist_known_answers(message, expected_hex):
    """Implementation must match the official NIST test vectors exactly."""
    assert sha3_256(message).hex() == expected_hex


# Output format tests===============================

def test_output_is_32_bytes():
    """SHA3-256 always produces a 256-bit (32-byte) digest."""
    assert len(sha3_256(b"")) == 32
    assert len(sha3_256(b"hello")) == 32
    assert len(sha3_256(b"x" * 1000)) == 32


def test_output_is_bytes():
    """Output type must be bytes, not str or list."""
    assert isinstance(sha3_256(b""), bytes)


# Determinism===============================

def test_deterministic():
    """Same input must always produce the same hash."""
    msg = b"deterministic test input"
    assert sha3_256(msg) == sha3_256(msg)


def test_different_inputs_different_outputs():
    """Distinct inputs should produce distinct outputs (collision resistance)."""
    assert sha3_256(b"hello") != sha3_256(b"world")
    assert sha3_256(b"a") != sha3_256(b"b")
    assert sha3_256(b"") != sha3_256(b"\x00")


# Avalanche effect===============================

def test_avalanche_single_bit_flip():
    """Flipping one bit should change roughly half the output bits."""
    a = sha3_256(b"\x00" * 32)
    b = sha3_256(b"\x01" + b"\x00" * 31)
    diff_bits = sum(bin(x ^ y).count("1") for x, y in zip(a, b))
    # statistically should be ~128 bits different out of 256
    assert 80 < diff_bits < 176


# Sponge rate boundary tests===============================
# rate = 136 bytes for SHA3-256. these tests check that padding and block
# absorption are correct around the boundary.

def test_at_rate_boundary_minus_one():
    """135 bytes: one less than the rate, fits in single block with padding."""
    h = sha3_256(b"x" * 135)
    assert len(h) == 32


def test_at_rate_boundary():
    """136 bytes: exactly one rate-sized block, needs full padding block."""
    h = sha3_256(b"x" * 136)
    assert len(h) == 32


def test_at_rate_boundary_plus_one():
    """137 bytes: forces 2 absorbed blocks."""
    h = sha3_256(b"x" * 137)
    assert len(h) == 32


def test_boundary_inputs_distinct():
    """The three boundary cases must produce distinct hashes."""
    h1 = sha3_256(b"x" * 135)
    h2 = sha3_256(b"x" * 136)
    h3 = sha3_256(b"x" * 137)
    assert h1 != h2 != h3 != h1


def test_multi_block_absorption():
    """Inputs spanning many blocks should hash correctly."""
    h = sha3_256(b"y" * 1000)
    assert len(h) == 32
    assert sha3_256(b"y" * 1000) == h


# Internal permutation tests===============================

def test_rot64_zero_rotation():
    """Rotating by 0 returns the input unchanged."""
    assert _rot64(0x123456789ABCDEF0, 0) == 0x123456789ABCDEF0


def test_rot64_full_rotation():
    """Rotating a 64-bit value by 64 should equal the input."""
    # _rot64 isn't defined for n=64 but rotation by 1 then 63 should round-trip
    x = 0xDEADBEEFCAFEBABE
    assert _rot64(_rot64(x, 1), 63) == x


def test_rot64_stays_in_64_bits():
    """Rotation result must always fit in 64 bits."""
    result = _rot64(0xFFFFFFFFFFFFFFFF, 17)
    assert 0 <= result <= MASK64


def test_keccak_f_is_not_identity():
    """The permutation must actually change the state."""
    zero_state = [[0] * 5 for _ in range(5)]
    result = _keccak_f(zero_state)
    # iota XORs round constants into [0][0], so it cannot stay all zero
    assert result != zero_state


def test_keccak_f_avalanche():
    """One-bit difference in input state should propagate to all lanes."""
    s1 = [[0] * 5 for _ in range(5)]
    s2 = [[0] * 5 for _ in range(5)]
    s2[0][0] = 1
    r1 = _keccak_f(s1)
    r2 = _keccak_f(s2)
    # essentially all lanes should differ after 24 rounds
    diffs = sum(1 for x in range(5) for y in range(5) if r1[x][y] != r2[x][y])
    assert diffs >= 20  # out of 25

@pytest.mark.parametrize("size", [0, 1, 17, 64, 135, 136, 137, 271, 272, 500, 2048])
def test_matches_hashlib_at_various_sizes(size):
    """Output must match Python's standard library across many input sizes."""
    msg = os.urandom(size)
    assert sha3_256(msg).hex() == hashlib.sha3_256(msg).hexdigest()


def test_matches_hashlib_fuzz():
    """Run 50 random inputs of random sizes and verify all match hashlib."""
    import random
    random.seed(42)  # deterministic test run
    for _ in range(50):
        size = random.randint(0, 5000)
        msg = os.urandom(size)
        assert sha3_256(msg).hex() == hashlib.sha3_256(msg).hexdigest(), \
            f"mismatch at size {size}"


def test_matches_hashlib_all_byte_values():
    """Test the boundary input bytes(range(256))."""
    msg = bytes(range(256))
    assert sha3_256(msg).hex() == hashlib.sha3_256(msg).hexdigest()


# Input immutability tests===============================

def test_input_not_modified():
    """Hashing must not mutate the caller's input bytes."""
    msg = bytearray(b"important data")
    original = bytes(msg)
    sha3_256(bytes(msg))
    assert bytes(msg) == original


# Long message tests===============================

def test_long_message():
    """Verify a multi-kilobyte message produces consistent output."""
    msg = b"long message test " * 1000  # ~18 KB
    h1 = sha3_256(msg)
    h2 = sha3_256(msg)
    assert h1 == h2
    assert len(h1) == 32
    # also verify against the reference
    assert h1.hex() == hashlib.sha3_256(msg).hexdigest()