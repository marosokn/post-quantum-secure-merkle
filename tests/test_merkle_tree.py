import pytest
from python.merkle.merkle_tree import MerkleTree

def fake_hash_leaf(leaf: bytes) -> bytes:
    """Return a deterministic fake leaf hash for MerkleTree unit tests."""
    return b"H(" + leaf + b")"


def fake_hash_pair(left: bytes, right: bytes) -> bytes:
    """Return a deterministic fake parent hash from two child hashes."""
    return b"H(" + left + b" and " + right + b")"

even_leaves = [b"a", b"b", b"c", b"d"]
odd_leaves = [b"a", b"b", b"c", b"d", b"e"]


def test_build_tree_even_number_of_leaves():
    """check that an even number of leaves builds levels with sizes 4, 2, 1"""
    tree = MerkleTree(
        leaves=even_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    assert len(tree.levels) == 3
    assert len(tree.levels[0]) == 4
    assert len(tree.levels[1]) == 2
    assert len(tree.levels[2]) == 1

def test_build_tree_odd_number_of_leaves():
    """check that an odd number of leaves duplicate the las node with itself"""
    tree = MerkleTree(
        leaves=odd_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    assert len(tree.levels) == 4
    assert len(tree.levels[0]) == 5
    assert len(tree.levels[1]) == 3
    assert len(tree.levels[2]) == 2
    assert len(tree.levels[3]) == 1

def test_empty_leaves():
    """check that an empty leaf list is rejected"""
    with pytest.raises(ValueError):
        MerkleTree(
            leaves=[],
            hash_leaf=fake_hash_leaf,
            hash_pair=fake_hash_pair,
        )

def test_root():
    """Check that root() returns the single node at the top of the tree."""
    even_tree = MerkleTree(
        leaves=even_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )
    odd_tree = MerkleTree(
        leaves=odd_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    assert even_tree.root() == b"H(H(H(a) and H(b)) and H(H(c) and H(d)))"
    assert odd_tree.root() == (
        b"H(H(H(H(a) and H(b)) and H(H(c) and H(d))) and "
        b"H(H(H(e) and H(e)) and H(H(e) and H(e))))"
    )

def test_generate_proof_even_tree():
    """check that generate_proof returns the correct list"""
    tree = MerkleTree(
        leaves=even_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    proof = tree.generate_proof(2)

    assert proof == [
        ("r", b"H(d)"),
        ("l", b"H(H(a) and H(b))"),
    ]

def test_verify_proof_accepts_valid_proof():
    """check that verify_proof accepts a vaild proof for the target leaf"""

    proof = [
        ("r", b"H(d)"),
        ("l", b"H(H(a) and H(b))"),
    ]

    root = b"H(H(H(a) and H(b)) and H(H(c) and H(d)))"

    assert MerkleTree.verify_proof(
        leaf=b"c",
        proof=proof,
        root=root,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

def test_verify_proof_rejects_wrong_leaf():
    """check that verify_proof does not accept a vaild proof for the target leaf"""
    proof = [
        ("r", b"H(d)"),
        ("l", b"H(H(a) and H(b))"),
    ]

    root = b"H(H(H(a) and H(b)) and H(H(c) and H(d)))"

    assert not MerkleTree.verify_proof(
        leaf=b"x",
        proof=proof,
        root=root,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )