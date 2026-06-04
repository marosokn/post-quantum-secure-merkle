"""python wrapper around the gpu keccak code in cuda/libpqmerkle.so.

the .so exposes two functions:
    keccak_hash_batch(inputs, input_len, n, outputs)
    merkle_build_keccak(leaves, leaf_len, n_leaves, root)

"""
from __future__ import annotations

import ctypes
from pathlib import Path


LIB_PATH = Path(__file__).resolve().parent.parent / "cuda" / "libpqmerkle.so"
HASH_BYTES = 32


class CudaKeccak:
    """handle for the gpu keccak library. no constants to upload."""

    def __init__(self, lib_path: Path | str | None = None):
        path = Path(lib_path) if lib_path else LIB_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"CUDA library not found at {path} — run `make -C cuda` first"
            )
        self.lib = ctypes.CDLL(str(path))

        u8p = ctypes.POINTER(ctypes.c_uint8)

        self.lib.keccak_hash_batch.argtypes = [
            u8p, ctypes.c_size_t, ctypes.c_size_t, u8p,
        ]
        self.lib.keccak_hash_batch.restype = None

        self.lib.merkle_build_keccak.argtypes = [
            u8p, ctypes.c_size_t, ctypes.c_size_t, u8p,
        ]
        self.lib.merkle_build_keccak.restype = None

        self.lib.merkle_generate_proof_keccak.argtypes = [
            u8p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, u8p, u8p, ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.merkle_generate_proof_keccak.restype = None

        self.lib.merkle_verify_proof_keccak.argtypes = [
            u8p, ctypes.c_size_t, u8p, u8p, ctypes.c_size_t, u8p, u8p,
        ]
        self.lib.merkle_verify_proof_keccak.restype = None

    # hash a batch of equal-length byte strings on the gpu
    def hash_batch(self, inputs: list[bytes]) -> list[bytes]:
        if not inputs:
            return []
        n = len(inputs)
        input_len = len(inputs[0])
        if any(len(b) != input_len for b in inputs):
            raise ValueError("all inputs must have the same length")

        flat = b"".join(inputs)
        in_buf = (ctypes.c_uint8 * (n * input_len)).from_buffer_copy(flat)
        out_buf = (ctypes.c_uint8 * (n * HASH_BYTES))()

        self.lib.keccak_hash_batch(in_buf, input_len, n, out_buf)
        return [
            bytes(out_buf[i * HASH_BYTES:(i + 1) * HASH_BYTES])
            for i in range(n)
        ]

    # build a sha-3 merkle tree, return the 32-byte root
    def merkle_root(self, leaves: list[bytes]) -> bytes:
        if not leaves:
            raise ValueError("at least one leaf required")
        n = len(leaves)
        leaf_len = len(leaves[0])
        if any(len(b) != leaf_len for b in leaves):
            raise ValueError("all leaves must have the same length")

        flat = b"".join(leaves)
        in_buf = (ctypes.c_uint8 * (n * leaf_len)).from_buffer_copy(flat)
        out_buf = (ctypes.c_uint8 * HASH_BYTES)()

        self.lib.merkle_build_keccak(in_buf, leaf_len, n, out_buf)
        return bytes(out_buf)
    
    def merkle_proof(self, leaves: list[bytes], index: int) -> list[tuple[str, bytes]]:
        if not leaves:
            raise ValueError("at lease one leaf required")
        if index < 0 or index >= len(leaves):
            raise IndexError("index out of range")
        
        n = len(leaves)
        leaf_len = len(leaves[0])
        if any(len(b) != leaf_len for b in leaves):
            raise ValueError("leaves must have same length")
        
        depth = 0
        target = 1
        while target < n:
            target <<= 1
            depth += 1

        flat = b"".join(leaves)
        in_buf = (ctypes.c_uint8 * (n * leaf_len)).from_buffer_copy(flat)
        proof_buf = (ctypes.c_uint8 * (depth*HASH_BYTES))()
        direction_buf = (ctypes.c_uint8 * depth)()
        proof_len = ctypes.c_size_t(0)

        self.lib.merkle_generate_proof_keccak(
            in_buf, leaf_len, n, index, proof_buf, direction_buf, ctypes.byref(proof_len)
        )

        proof: list[tuple[str, bytes]] = []
        for i in range(proof_len.value):
            sibling = bytes(proof_buf[i * HASH_BYTES:(i+1) * HASH_BYTES])
            direction = "r" if direction_buf[i] == 1 else "l"
            proof.append((direction, sibling))

        return proof
    
    def verify_merkle_proof(self, leaf: bytes, proof: list[tuple[str, bytes]], root: bytes) -> bool:
        if len(root) != HASH_BYTES:
            raise ValueError("root must be same with hash bytes")

        leaf_buf = (ctypes.c_uint8 * len(leaf)).from_buffer_copy(leaf)

        proof_bytes = b"".join(sibling for _, sibling in proof)
        proof_buf = (ctypes.c_uint8 * len(proof_bytes)).from_buffer_copy(proof_bytes)

        directions_buf = (ctypes.c_uint8* len(proof))()
        for i , (direction, sibling) in enumerate(proof):
            if len(sibling) != HASH_BYTES:
                raise ValueError("each proof sibling must be hash bytes")
            if direction == "r":
                directions_buf[i] = 1
            else:
                directions_buf[i] = 0

        root_buf = (ctypes.c_uint8 * HASH_BYTES).from_buffer_copy(root)
        ok = ctypes.c_uint8(0)

        self.lib.merkle_verify_proof_keccak(
            leaf_buf, len(leaf), proof_buf, directions_buf, len(proof), root_buf, ctypes.byref(ok),
        )

        return bool(ok.value)
    