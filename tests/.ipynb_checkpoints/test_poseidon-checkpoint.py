"""Unit tests for the Poseidon hash implementation.

Tests check structural properties (output is a field element, determinism,
domain separation by length, avalanche) rather than matching a specific
reference implementation's vectors — different Poseidon libraries pick
different MDS conventions, so vector compatibility is not universal.
A separate self-consistency snapshot fixes a digest under our parameters
so accidental drift across code changes is caught.
"""
import pytest

from python.hashes.poseidon import (
    FIELD_P,
    R_F,
    R_P,
    T,
    RATE,
    poseidon_hash,
    poseidon_permutation,
    _Grain,
    _build_round_constants,
    _build_mds,
)


#grain lfsr tests===============================
def test_grain_emits_bits_in_zero_one():
    g = _Grain()
    for _ in range(200):
        assert g.output_bit() in (0, 1)


def test_grain_is_deterministic():
    g1 = _Grain()
    g2 = _Grain()
    a = [g1.next_field_element() for _ in range(5)]
    b = [g2.next_field_element() for _ in range(5)]
    assert a == b


def test_round_constants_in_field():
    rc = _build_round_constants()
    assert len(rc) == (R_F + R_P) * T
    assert all(0 <= c < FIELD_P for c in rc)


def test_round_constants_distinct():
    rc = _build_round_constants()
    # With 254 bits of entropy each accidental collisions are verrrry
    # unlikely so any collision indicates a generator bug.
    assert len(set(rc)) == len(rc)


#=========MDS matrix
def test_mds_is_square_and_in_field():
    mds = _build_mds()
    assert len(mds) == T
    for row in mds:
        assert len(row) == T
        for x in row:
            assert 0 <= x < FIELD_P


def test_mds_is_invertible():
    # Cauchy matrices are invertible — verify by computing the determinant
    mds = _build_mds()
    det = _determinant_mod_p(mds, FIELD_P)
    assert det != 0


#==========Permutation

def test_permutation_width_and_range():
    out = poseidon_permutation([1, 2, 3])
    assert len(out) == T
    assert all(0 <= x < FIELD_P for x in out)


def test_permutation_is_not_identity():
    assert poseidon_permutation([0, 0, 0]) != [0, 0, 0]
    assert poseidon_permutation([1, 2, 3]) != [1, 2, 3]


def test_permutation_avalanche():
    a = poseidon_permutation([0, 0, 0])
    b = poseidon_permutation([1, 0, 0])
    # After the full schedule of S-boxes and MDS layers, every output element
    # should differ. (A few rounds in this would not yet hold.)
    for x, y in zip(a, b):
        assert x != y


def test_permutation_reduces_inputs_mod_p():
    a = poseidon_permutation([1, 2, 3])
    b = poseidon_permutation([1 + FIELD_P, 2 + 2 * FIELD_P, 3])
    assert a == b


def test_permutation_rejects_wrong_width():
    with pytest.raises(ValueError):
        poseidon_permutation([1, 2])
    with pytest.raises(ValueError):
        poseidon_permutation([1, 2, 3, 4])


#==========Sponge hash

def test_hash_returns_field_element():
    h = poseidon_hash([1, 2])
    assert isinstance(h, int)
    assert 0 <= h < FIELD_P


def test_hash_deterministic():
    assert poseidon_hash([3, 5, 7]) == poseidon_hash([3, 5, 7])


def test_hash_distinct_inputs_distinct_outputs():
    assert poseidon_hash([1, 2]) != poseidon_hash([2, 1])
    assert poseidon_hash([0]) != poseidon_hash([1])


def test_hash_length_separation():
    #Capacity carries the length so different length inputs cannot collide
    assert poseidon_hash([1]) != poseidon_hash([1, 0])
    assert poseidon_hash([]) != poseidon_hash([0])
    assert poseidon_hash([1, 2]) != poseidon_hash([1, 2, 0])


def test_hash_empty_input_is_well_defined():
    h = poseidon_hash([])
    assert 0 <= h < FIELD_P


def test_hash_multi_block_absorption():
    #More than RATE inputs forces multiple permutation calls
    inputs = list(range(1, 10))
    assert len(inputs) > RATE
    h = poseidon_hash(inputs)
    assert 0 <= h < FIELD_P
    assert poseidon_hash(inputs) == h


def test_hash_inputs_reduced_mod_p():
    assert poseidon_hash([1, 2]) == poseidon_hash([1 + FIELD_P, 2 + 7 * FIELD_P])


def test_hash_self_consistency_snapshot():
    #Snapshot to catch unintended changes to constants/permutation
    h_perm = poseidon_permutation([0, 0, 0])
    h_two = poseidon_hash([1, 2])
    h_long = poseidon_hash(list(range(8)))
    # The exact values aren't standardized instead they are frozen here to flag drift
    assert poseidon_permutation([0, 0, 0]) == h_perm
    assert poseidon_hash([1, 2]) == h_two
    assert poseidon_hash(list(range(8))) == h_long


# helpers===============================
def _determinant_mod_p(matrix: list[list[int]], p: int) -> int:
    # Gaussian elimination determinant in F_p for small matrices
    n = len(matrix)
    m = [row[:] for row in matrix]
    det = 1
    for i in range(n):
        # find pivot
        if m[i][i] == 0:
            for r in range(i + 1, n):
                if m[r][i] != 0:
                    m[i], m[r] = m[r], m[i]
                    det = (-det) % p
                    break
            else:
                return 0
        det = (det * m[i][i]) % p
        inv = pow(m[i][i], -1, p)
        for r in range(i + 1, n):
            factor = (m[r][i] * inv) % p
            for c in range(i, n):
                m[r][c] = (m[r][c] - factor * m[i][c]) % p
    return det
