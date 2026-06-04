"""merkle tree built with the poseidon hash.

a merkle tree hashes leaves in pairs, then hashes those results in pairs, and
so on until one value is left - the root. it lets you prove a leaf is in the
tree without revealing the whole tree.

each leaf is hashed on its own (poseidon_hash([leaf])) before it goes in, so a
leaf value cant be confused with an internal node. if the leaf count isnt a
power of two the rest is padded with zeros - thats safe because finding a real
leaf that hashes to zero would mean breaking poseidon.
"""
from __future__ import annotations
from typing import TypeVar

from ..hashes.poseidon import FIELD_P, poseidon_hash
from ..hashes.keccak import sha3_256

HashNode = TypeVar("HashNode")


def _generate_proof_from_levels(
    levels: list[list[HashNode]],
    leaf_count: int,
    index: int,
) -> list[tuple[str, HashNode]]:
    if index < 0 or index >= leaf_count:
        raise IndexError("leaf index out of range")

    proof: list[tuple[str, HashNode]] = []
    current_index = index

    for level in levels[:-1]:
        if current_index % 2 == 0:
            sibling_index = current_index + 1
            direction = "r"
        else:
            sibling_index = current_index - 1
            direction = "l"
        proof.append((direction, level[sibling_index]))
        current_index //= 2

    return proof


class Keccak256MerkleTree:
    # build the tree from the leaves up to the root
    def __init__(self, leaves: list[bytes]):
        if not leaves:
            raise ValueError("at least one leaf required")

        self.original_leaves: list[bytes] = list(leaves)

        # round up to a power of two, pad with 32 zero bytes
        n = len(self.original_leaves)
        target = 1
        while target < n:
            target <<= 1

        level0 = [sha3_256(x) for x in self.original_leaves]
        level0 += [bytes(32)] * (target - n)

        # hash pairs level by level until only the root is left
        self.levels: list[list[bytes]] = [level0]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = [
                sha3_256(cur[2 * i] + cur[2 * i + 1])
                for i in range(len(cur) // 2)
            ]
            self.levels.append(nxt)

    @property
    def root(self) -> bytes:
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        return len(self.levels) - 1

    def leaf_count(self) -> int:
        return len(self.original_leaves)

    def generate_proof(self, index: int) -> list[tuple[str, bytes]]:
        return _generate_proof_from_levels(
            self.levels,
            len(self.original_leaves),
            index,
        )

    @staticmethod
    def verify_proof(
        leaf: bytes,
        proof: list[tuple[str, bytes]],
        root: bytes,
    ) -> bool:
        current = sha3_256(leaf)

        for direction, sibling in proof:
            if direction == "r":
                current = sha3_256(current + sibling)
            elif direction == "l":
                current = sha3_256(sibling + current)
            else:
                raise ValueError("direction must be 'l' or 'r'")

        return current == root


class PoseidonMerkleTree:
    # build the tree from the leaves up to the root
    def __init__(self, leaves: list[int]):
        if not leaves:
            raise ValueError("at least one leaf required")

        self.original_leaves: list[int] = [int(x) % FIELD_P for x in leaves]

        # round the leaf count up to a power of two, pad the rest with zeros
        n = len(self.original_leaves)
        target = 1
        while target < n:
            target <<= 1

        level0 = [poseidon_hash([x]) for x in self.original_leaves]
        level0 += [0] * (target - n)

        # hash pairs level by level until only the root is left
        self.levels: list[list[int]] = [level0]
        while len(self.levels[-1]) > 1:
            cur = self.levels[-1]
            nxt = [
                poseidon_hash([cur[2 * i], cur[2 * i + 1]])
                for i in range(len(cur) // 2)
            ]
            self.levels.append(nxt)

    @property
    def root(self) -> int:
        # the single value at the top of the tree
        return self.levels[-1][0]

    @property
    def depth(self) -> int:
        # number of levels from the leaves up to the root
        return len(self.levels) - 1

    def leaf_count(self) -> int:
        # number of real leaves, not counting padding
        return len(self.original_leaves)

    def generate_proof(self, index: int) -> list[tuple[str, int]]:
        return _generate_proof_from_levels(
            self.levels,
            len(self.original_leaves),
            index,
        )

    @staticmethod
    def verify_proof(
        leaf: int,
        proof: list[tuple[str, int]],
        root: int,
    ) -> bool:
        current = poseidon_hash([int(leaf) % FIELD_P])

        for direction, sibling in proof:
            if direction == "r":
                current = poseidon_hash([current, sibling])
            elif direction == "l":
                current = poseidon_hash([sibling, current])
            else:
                raise ValueError("direction must be 'l' or 'r'")

        return current == root
