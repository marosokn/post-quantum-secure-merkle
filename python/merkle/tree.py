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

from ..hashes.poseidon import FIELD_P, poseidon_hash


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
