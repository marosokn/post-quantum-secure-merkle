import pytest
from python.merkle.merkle_tree import MerkleTree

def fake_hash_leaf(leaf):
    return f"H({leaf})"

def fake_hash_pair(left, right):
    return f"H({left} and {right})"

even_leaves = ["a", "b", "c", "d"]
odd_leaves = ["a", "b", "c", "d", "e"]


def test_build_tree_even_number_of_leaves():
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
    with pytest.raises(ValueError):
        MerkleTree(
            leaves=[],
            hash_leaf=fake_hash_leaf,
            hash_pair=fake_hash_pair,
        )

def test_root():
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

    assert even_tree.root() == "H(H(H(a) and H(b)) and H(H(c) and H(d)))"
    assert odd_tree.root() == "H(H(H(H(a) and H(b)) and H(H(c) and H(d))) and H(H(H(e) and H(e)) and H(H(e) and H(e))))"


def test_generate_proof_even_tree():
    tree = MerkleTree(
        leaves=even_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    proof = tree.generate_proof(2)

    assert proof == [
        ("r", "H(d)"),
        ("l", "H(H(a) and H(b))"),
    ]

def test_verify_proof_accepts_valid_proof():
    tree = MerkleTree(
        leaves=even_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    proof = [
        ("r", "H(d)"),
        ("l", "H(H(a) and H(b))"),
    ]

    root = "H(H(H(a) and H(b)) and H(H(c) and H(d)))"

    assert MerkleTree.verify_proof(
        leaf="c",
        proof=proof,
        root=root,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )
def test_verify_proof_rejects_wrong_leaf():
    tree = MerkleTree(
        leaves=even_leaves,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )

    proof = [
        ("r", "H(d)"),
        ("l", "H(H(a) and H(b))"),
    ]

    root = "H(H(H(a) and H(b)) and H(H(c) and H(d)))"

    assert not MerkleTree.verify_proof(
        leaf="x",
        proof=proof,
        root=root,
        hash_leaf=fake_hash_leaf,
        hash_pair=fake_hash_pair,
    )