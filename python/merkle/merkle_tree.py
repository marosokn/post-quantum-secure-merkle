from __future__ import annotations

class MerkleTree:
    def __init__(self, leaves, hash_leaf, hash_pair):
        self.leaves = leaves
        self.hash_leaf = hash_leaf
        self.hash_pair = hash_pair

        if len(leaves) == 0:
            raise ValueError("leaves must not be empty")

        leaf_hashes = []
        for leaf in leaves:
            leaf_hashes.append(self.hash_leaf(leaf))

        self.levels = self._build_tree(leaf_hashes=leaf_hashes)

    def _build_tree(self, leaf_hashes):
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

    def root(self):
        return self.levels[-1][0]

    def generate_proof(self, index):
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
    def verify_proof(leaf, proof, root, hash_leaf, hash_pair):
        current = hash_leaf(leaf)
        for direction, sibling in proof:
            if direction == "r":
                current = hash_pair(current, sibling)
            else:
                current = hash_pair(sibling, current)

        return current == root