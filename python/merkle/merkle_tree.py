from __future__ import annotations

class MerkleTree:
    """
    Builds a binary Merkle tree from a list of leaves.

    The tree itself is stored as a list of levels.
    levels[0] contains the hashed leaves, and levels[-1][0] is the Merkle root.
    If the number of nodes in a level is odd, the last node is paired with itself.

    Args:
        leaves: Original leaf values.
        hash_leaf: Function that converts a leaf value into a hash.
        hash_pair: Function that combines two child nodes into one parent node.
    """
    def __init__(self, leaves: list[bytes], hash_leaf, hash_pair):
        self.leaves = leaves
        self.hash_leaf = hash_leaf
        self.hash_pair = hash_pair

        if len(leaves) == 0:
            raise ValueError("leaves must not be empty")

        leaf_hashes = []
        for leaf in leaves:
            leaf_hashes.append(self.hash_leaf(leaf))

        self.levels = self._build_tree(leaf_hashes=leaf_hashes)

    def _build_tree(self, leaf_hashes: list[bytes]) -> list[list[bytes]]:
        """
        Build tree levels from leaf hashes up to the root
        
        Args:
            leaf_hashes: list of leaves that are already hashed
            
        Returns:
            A list of levels, where levels[0] contains hashed leaves and 
            levels[-1] contains the root hash.
        """
        levels = []
        levels.append(leaf_hashes)
        current_level = leaf_hashes

        while (len(current_level) > 1):
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                if i+1 < len(current_level):
                    right = current_level[i+1]
                else:
                    right = left
                
                parent = self.hash_pair(left, right)
                next_level.append(parent)
            levels.append(next_level)
            current_level = next_level
        
        return levels

    def root(self) -> bytes:
        """
        Returns the root.
        """
        return self.levels[-1][0]

    def generate_proof(self, index: int) -> list[tuple[str, bytes]]:
        """
        Generate a Merkle proof for a leaf at index leaf
        
        Args:
            index: Index of the leaf to prove
            
        Returns:
            A list of sibling hashes with their direction.
            "l" means the sibling is on the left,
            "r" means the sibling is on the right.
        """
        proof = []
        current_index = index

        for level in self.levels[:-1]:
            if current_index%2 == 0:
                direction = "r"
                sibling_idx = current_index + 1

                if sibling_idx < len(level):
                    sibling = level[sibling_idx]
                else:
                    sibling = level[current_index]
            else:
                direction = "l"
                sibling = level[current_index-1]
            
            proof.append((direction, sibling))
            current_index = current_index // 2
            
        return proof

    @staticmethod
    def verify_proof(
        leaf: bytes, 
        proof: list[tuple[str, bytes]], 
        root: bytes, 
        hash_leaf, 
        hash_pair
    ) -> bool:
        """
        Verifies whether the leaf is in the tree given the root.

        Args: 
            leaf: A leaf value before hashed.
            proof: Sibling hashes and directions from leaf level to root
            root: Expected Merkle root
            hash_leaf: Function that converts a leaf value into a hash.
            hash_pair: Function that combines two child nodes into one parent node.

        Returns:
            True if proof reconstructs the given root, otherwise False.
        """
        current = hash_leaf(leaf)
        for direction, sibling in proof:
            if direction == "r":
                current = hash_pair(current, sibling)
            else:
                current = hash_pair(sibling, current)

        return current == root