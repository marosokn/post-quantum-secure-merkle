"""python wrapper around the gpu code in cuda/libpqmerkle.so.

the .so exposes three functions:
    poseidon_load_constants(rc_mont, mds_mont)
    poseidon_hash_batch(inputs, inputs_per_hash, n, outputs)
    merkle_build_poseidon(leaves, n_leaves, root)

field elements are too big for one int, so each one crosses the boundary as 4
limbs of 64 bits. the round constants and MDS are converted to Montgomery form
here before being uploaded so the gpu doesnt have to. normal inputs and
outputs stay as regular integers.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

from python.hashes.poseidon import (
    FIELD_P,
    T,
    R_F,
    R_P,
    build_mds,
    build_round_constants,
)


LIB_PATH = Path(__file__).resolve().parent.parent / "cuda" / "libpqmerkle.so"

FP_LIMBS = 4
R = 1 << 256


def _to_mont(x: int) -> int:
    # convert a normal field element into Montgomery form
    return (x * R) % FIELD_P


def _to_limbs(x: int) -> tuple[int, int, int, int]:
    # split a big number into 4 limbs of 64 bits, least significant first
    mask = (1 << 64) - 1
    return (x & mask, (x >> 64) & mask, (x >> 128) & mask, (x >> 192) & mask)


def _from_limbs(limbs) -> int:
    # rebuild a big number from its 4 limbs
    return limbs[0] | (limbs[1] << 64) | (limbs[2] << 128) | (limbs[3] << 192)


def _pack_field_elements(values, mont: bool):
    # flatten a list of field elements into one ctypes uint64 array for the gpu
    n = len(values)
    buf = (ctypes.c_uint64 * (n * FP_LIMBS))()
    for i, v in enumerate(values):
        v_int = v % FIELD_P
        if mont:
            v_int = _to_mont(v_int)
        for k, limb in enumerate(_to_limbs(v_int)):
            buf[i * FP_LIMBS + k] = limb
    return buf


def _unpack_field_elements(buf, n: int) -> list[int]:
    # turn the gpu's flat uint64 array back into a list of field elements
    return [
        _from_limbs([buf[i * FP_LIMBS + k] for k in range(FP_LIMBS)])
        for i in range(n)
    ]


class CudaPoseidon:
    """handle for the gpu poseidon library.

    constructing one loads the .so and uploads the round constants + MDS to the
    gpu. after that hash_batch and merkle_root are cheap to call.
    """

    # load the .so, set up the function signatures, upload the constants
    def __init__(self, lib_path: Path | str | None = None):
        path = Path(lib_path) if lib_path else LIB_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"CUDA library not found at {path} — run `make -C cuda` first"
            )
        self.lib = ctypes.CDLL(str(path))

        u64p = ctypes.POINTER(ctypes.c_uint64)

        self.lib.poseidon_load_constants.argtypes = [u64p, u64p]
        self.lib.poseidon_load_constants.restype = None

        self.lib.poseidon_hash_batch.argtypes = [
            u64p, ctypes.c_int, ctypes.c_size_t, u64p,
        ]
        self.lib.poseidon_hash_batch.restype = None

        self.lib.merkle_build_poseidon.argtypes = [
            u64p, ctypes.c_size_t, u64p,
        ]
        self.lib.merkle_build_poseidon.restype = None

        self.lib.merkle_generate_proof_poseidon.argtypes = [
            u64p, ctypes.c_size_t, ctypes.c_size_t, u64p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        self.lib.merkle_generate_proof_poseidon.restype = None

        self.lib.merkle_verify_proof_poseidon.argtypes = [
            u64p, u64p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            u64p,
            ctypes.POINTER(ctypes.c_uint8),
        ]
        self.lib.merkle_verify_proof_poseidon.restype = None

        self._upload_constants()

    # build the round constants + MDS and upload them in Montgomery form
    def _upload_constants(self) -> None:
        rc = build_round_constants()
        mds = build_mds()
        assert len(rc) == (R_F + R_P) * T
        assert len(mds) == T and all(len(row) == T for row in mds)

        rc_buf = _pack_field_elements(rc, mont=True)
        mds_flat = [mds[i][j] for i in range(T) for j in range(T)]
        mds_buf = _pack_field_elements(mds_flat, mont=True)

        self.lib.poseidon_load_constants(rc_buf, mds_buf)

    # hash a batch of input groups on the gpu, every group must be the same length
    def hash_batch(self, input_groups: list[list[int]]) -> list[int]:
        if not input_groups:
            return []
        n = len(input_groups)
        inputs_per_hash = len(input_groups[0])
        if any(len(g) != inputs_per_hash for g in input_groups):
            raise ValueError("all input groups must have the same length")

        flat_inputs = [x for group in input_groups for x in group]
        in_buf = _pack_field_elements(flat_inputs, mont=False)
        out_buf = (ctypes.c_uint64 * (n * FP_LIMBS))()

        self.lib.poseidon_hash_batch(in_buf, inputs_per_hash, n, out_buf)
        return _unpack_field_elements(out_buf, n)

    # build a poseidon merkle tree from the leaves on the gpu, return the root
    def merkle_root(self, leaves: list[int]) -> int:
        if not leaves:
            raise ValueError("at least one leaf required")
        n = len(leaves)
        in_buf = _pack_field_elements(leaves, mont=False)
        out_buf = (ctypes.c_uint64 * FP_LIMBS)()
        self.lib.merkle_build_poseidon(in_buf, n, out_buf)
        return _from_limbs([out_buf[k] for k in range(FP_LIMBS)])

    def merkle_proof(self, leaves: list[int], index: int) -> list[tuple[str, int]]:
        if not leaves:
            raise ValueError("at least one leaf required")
        if index < 0 or index >= len(leaves):
            raise IndexError("leaf index out of range")

        n = len(leaves)
        depth = 0
        target = 1
        while target < n:
            target <<= 1
            depth += 1

        in_buf = _pack_field_elements(leaves, mont=False)
        proof_buf = (ctypes.c_uint64 * (depth * FP_LIMBS))()
        directions_buf = (ctypes.c_uint8 * depth)()
        proof_len = ctypes.c_size_t(0)

        self.lib.merkle_generate_proof_poseidon(
            in_buf,
            n,
            index,
            proof_buf,
            directions_buf,
            ctypes.byref(proof_len),
        )

        proof_values = _unpack_field_elements(proof_buf, proof_len.value)
        proof: list[tuple[str, int]] = []
        for i, sibling in enumerate(proof_values):
            direction = "r" if directions_buf[i] == 1 else "l"
            proof.append((direction, sibling))

        return proof

    def verify_merkle_proof(
        self,
        leaf: int,
        proof: list[tuple[str, int]],
        root: int,
    ) -> bool:
        leaf_buf = _pack_field_elements([leaf], mont=False)
        proof_values = [sibling for _, sibling in proof]
        proof_buf = _pack_field_elements(proof_values, mont=False)

        directions_buf = (ctypes.c_uint8 * len(proof))()
        for i, (direction, _) in enumerate(proof):
            if direction == "r":
                directions_buf[i] = 1
            elif direction == "l":
                directions_buf[i] = 0
            else:
                raise ValueError("direction must be 'l' or 'r'")

        root_buf = _pack_field_elements([root], mont=False)
        ok = ctypes.c_uint8(0)

        self.lib.merkle_verify_proof_poseidon(
            leaf_buf,
            proof_buf,
            directions_buf,
            len(proof),
            root_buf,
            ctypes.byref(ok),
        )

        return bool(ok.value)
