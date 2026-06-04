"""runs the cpu and gpu versions side by side and checks they agree.

for each size it:
1. generates the same random leaves for both sides
2. hashes / builds the tree on the cpu (python/hashes/poseidon.py)
3. hashes / builds the tree on the gpu (cuda/libpqmerkle.so)
4. prints the timings and checks the outputs are exactly equal

run `make -C cuda` first. the timings include the host->device copy and the
python packing. the gpu wins big once theres enough work, but for tiny inputs
the cpu can be faster since the gpu has fixed per-call overhead.
"""
from __future__ import annotations

import random
import sys
import time

from python.hashes.poseidon import FIELD_P, poseidon_hash
from python.merkle.tree import PoseidonMerkleTree
from bindings.cuda_poseidon import CudaPoseidon
from python.merkle.tree import Keccak256MerkleTree
from python.hashes.keccak import sha3_256
from bindings.cuda_keccak import CudaKeccak


def random_field_elements(n: int, seed: int = 0) -> list[int]:
    # generate n random field elements, same seed gives the same values
    rng = random.Random(seed)
    return [rng.randrange(FIELD_P) for _ in range(n)]

def random_bytes_list(n: int, size: int = 32, seed: int = 0) -> list[bytes]:
    rng = random.Random(seed)
    return [bytes(rng.getrandbits(8) for _ in range(size)) for _ in range(n)]


# ---------------------------------------------------------------------------

def compare_single_hash(cuda: CudaPoseidon) -> None:
    # hash a few hand-picked inputs and check cpu and gpu agree
    print("=" * 72)
    print("Single-hash sanity check (Poseidon over BN254)")
    print("=" * 72)
    # the gpu needs every input in a batch to be the same length, so group
    # them by length. the cpu result is the source of truth.
    grouped = {
        1: [[0], [42], [FIELD_P - 1]],
        2: [[1, 2], [FIELD_P - 1, FIELD_P - 2]],
        4: [[42, 7, 1337, 9001]],
    }
    for length, cases in grouped.items():
        cpu = [poseidon_hash(c) for c in cases]
        gpu = cuda.hash_batch(cases)
        for inp, c, g in zip(cases, cpu, gpu):
            ok = "OK " if c == g else "FAIL"
            print(f"  [{ok}] inputs={inp}")
            print(f"        CPU = 0x{c:064x}")
            print(f"        GPU = 0x{g:064x}")
        assert cpu == gpu, f"single-hash mismatch at length {length}"


def compare_hash_batch(cuda: CudaPoseidon, n: int, inputs_per_hash: int = 2) -> None:
    # hash n inputs on both sides, time them, check they match
    print("=" * 72)
    print(f"Hash batch: N={n}, inputs_per_hash={inputs_per_hash}")
    print("=" * 72)
    groups = [random_field_elements(inputs_per_hash, seed=i) for i in range(n)]

    t0 = time.perf_counter()
    cpu = [poseidon_hash(g) for g in groups]
    cpu_dt = time.perf_counter() - t0

    # warm-up call first so the one-time first-launch cost isnt in the timing
    cuda.hash_batch(groups[:1])
    t0 = time.perf_counter()
    gpu = cuda.hash_batch(groups)
    gpu_dt = time.perf_counter() - t0

    match = cpu == gpu
    speedup = cpu_dt / gpu_dt if gpu_dt > 0 else float("inf")
    print(f"  CPU : {cpu_dt*1000:8.2f} ms  ({cpu_dt*1e6/n:7.2f} us/hash)")
    print(f"  GPU : {gpu_dt*1000:8.2f} ms  ({gpu_dt*1e6/n:7.2f} us/hash)")
    print(f"  speedup x{speedup:6.2f}    match: {match}")
    if not match:
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            if c != g:
                print(f"  mismatch at i={i}: cpu=0x{c:x} gpu=0x{g:x}")
                break
    assert match, "hash batch mismatch"


def compare_merkle(cuda: CudaPoseidon, n_leaves: int) -> None:
    # build a merkle tree on both sides, time them, check the roots match
    print("=" * 72)
    print(f"Merkle tree: {n_leaves} leaves")
    print("=" * 72)
    leaves = random_field_elements(n_leaves, seed=n_leaves * 13 + 1)

    t0 = time.perf_counter()
    cpu_root = PoseidonMerkleTree(leaves).root
    cpu_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_root = cuda.merkle_root(leaves)
    gpu_dt = time.perf_counter() - t0

    match = cpu_root == gpu_root
    speedup = cpu_dt / gpu_dt if gpu_dt > 0 else float("inf")
    print(f"  CPU : {cpu_dt*1000:8.2f} ms")
    print(f"  GPU : {gpu_dt*1000:8.2f} ms")
    print(f"  speedup x{speedup:6.2f}    match: {match}")
    print(f"  root: 0x{cpu_root:064x}")
    if not match:
        print(f"  GPU root differs: 0x{gpu_root:064x}")
    assert match, "merkle root mismatch"

def compare_poseidon_proof(cuda: CudaPoseidon, n_leaves: int, index: int) -> None:
    print("=" * 72)
    print(f"Merkle proof (Poseidon): {n_leaves} leaves, index={index}")
    print("=" * 72)
    leaves = random_field_elements(n_leaves, seed=n_leaves * 17 + 3)

    t0 = time.perf_counter()
    cpu_tree = PoseidonMerkleTree(leaves)
    cpu_proof = cpu_tree.generate_proof(index)
    cpu_gen_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_proof = cuda.merkle_proof(leaves, index)
    gpu_gen_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpu_ok = PoseidonMerkleTree.verify_proof(
        leaves[index],
        cpu_proof,
        cpu_tree.root,
    )
    cpu_verify_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_ok = cuda.verify_merkle_proof(
        leaves[index],
        gpu_proof,
        cpu_tree.root,
    )
    gpu_verify_dt = time.perf_counter() - t0

    wrong_leaf = (leaves[index] + 1) % FIELD_P
    gpu_wrong_leaf_ok = cuda.verify_merkle_proof(
        wrong_leaf,
        gpu_proof,
        cpu_tree.root,
    )

    proof_match = cpu_proof == gpu_proof

    print(f"  proof length          : {len(cpu_proof)}")
    print(f"  CPU proof verifies    : {cpu_ok}")
    print(f"  GPU proof verifies    : {gpu_ok}")
    print(f"  GPU wrong leaf passes : {gpu_wrong_leaf_ok}")
    print(f"  CPU/GPU proof match   : {proof_match}")
    print(f"  CPU generation        : {cpu_gen_dt*1000:8.2f} ms")
    print(f"  GPU generation        : {gpu_gen_dt*1000:8.2f} ms")
    print(f"  CPU verification      : {cpu_verify_dt*1000:8.2f} ms")
    print(f"  GPU verification      : {gpu_verify_dt*1000:8.2f} ms")
    print(f"  root                  : 0x{cpu_tree.root:064x}")

    assert cpu_ok, "Poseidon CPU proof verification failed"
    assert gpu_ok, "Poseidon GPU proof verification failed"
    assert not gpu_wrong_leaf_ok, "Poseidon GPU proof accepted the wrong leaf"
    assert proof_match, "Poseidon CPU/GPU proofs differ"

# ---------------------------------------------------------------------------

def compare_keccak_single_hash(cuda: CudaKeccak) -> None:
    # hash a few hand-picked inputs and check cpu and gpu agree
    print("=" * 72)
    print("Single-hash sanity check (SHA-3 / Keccak-256)")
    print("=" * 72)
    # gpu hash_batch needs equal-length inputs, so group by length
    grouped = {
        0:  [b""],
        3:  [b"abc"],
        32: [bytes(32), bytes(range(32))],
        43: [b"The quick brown fox jumps over the lazy dog"],
    }
    for length, cases in grouped.items():
        cpu = [sha3_256(c) for c in cases]
        gpu = cuda.hash_batch(cases)
        for inp, c, g in zip(cases, cpu, gpu):
            ok = "OK " if c == g else "FAIL"
            disp = (inp[:24] + b"...").hex() if len(inp) > 24 else inp.hex()
            print(f"  [{ok}] input_len={length}  input=0x{disp}")
            print(f"        CPU = 0x{c.hex()}")
            print(f"        GPU = 0x{g.hex()}")
        assert cpu == gpu, f"sha-3 single-hash mismatch at length {length}"


def compare_keccak_hash_batch(cuda: CudaKeccak, n: int, input_len: int = 64) -> None:
    # hash n inputs on both sides, time them, check they match
    print("=" * 72)
    print(f"SHA-3 hash batch: N={n}, input_len={input_len}")
    print("=" * 72)
    inputs = random_bytes_list(n, size=input_len, seed=n)

    t0 = time.perf_counter()
    cpu = [sha3_256(b) for b in inputs]
    cpu_dt = time.perf_counter() - t0

    # warmup so the first-launch cost isnt in the timing
    cuda.hash_batch(inputs[:1])
    t0 = time.perf_counter()
    gpu = cuda.hash_batch(inputs)
    gpu_dt = time.perf_counter() - t0

    match = cpu == gpu
    speedup = cpu_dt / gpu_dt if gpu_dt > 0 else float("inf")
    print(f"  CPU : {cpu_dt*1000:8.2f} ms  ({cpu_dt*1e6/n:7.2f} us/hash)")
    print(f"  GPU : {gpu_dt*1000:8.2f} ms  ({gpu_dt*1e6/n:7.2f} us/hash)")
    print(f"  speedup x{speedup:6.2f}    match: {match}")
    if not match:
        for i, (c, g) in enumerate(zip(cpu, gpu)):
            if c != g:
                print(f"  mismatch at i={i}: cpu=0x{c.hex()} gpu=0x{g.hex()}")
                break
    assert match, "sha-3 hash batch mismatch"


def compare_keccak_merkle(cuda: CudaKeccak, n_leaves: int) -> None:
    # build a merkle tree on both sides, time them, check the roots match
    print("=" * 72)
    print(f"SHA-3 Merkle tree: {n_leaves} leaves")
    print("=" * 72)
    leaves = random_bytes_list(n_leaves, size=32, seed=n_leaves * 13 + 2)

    t0 = time.perf_counter()
    cpu_root = Keccak256MerkleTree(leaves).root
    cpu_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_root = cuda.merkle_root(leaves)
    gpu_dt = time.perf_counter() - t0

    match = cpu_root == gpu_root
    speedup = cpu_dt / gpu_dt if gpu_dt > 0 else float("inf")
    print(f"  CPU : {cpu_dt*1000:8.2f} ms")
    print(f"  GPU : {gpu_dt*1000:8.2f} ms")
    print(f"  speedup x{speedup:6.2f}    match: {match}")
    print(f"  root: 0x{cpu_root.hex()}")
    if not match:
        print(f"  GPU root differs: 0x{gpu_root.hex()}")
    assert match, "sha-3 merkle root mismatch"

def compare_keccak_proof(cuda: CudaKeccak, n_leaves: int, index: int) -> None:
    print("=" * 72)
    print(f"CPU SHA-3 proof: {n_leaves} leaves, index={index}")
    print("=" * 72)
    leaves = random_bytes_list(n_leaves, size=32, seed=n_leaves * 17 + 4)

    t0 = time.perf_counter()
    cpu_tree = Keccak256MerkleTree(leaves)
    cpu_proof = cpu_tree.generate_proof(index)
    cpu_gen_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_proof = cuda.merkle_proof(leaves, index)
    gpu_gen_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    cpu_ok = Keccak256MerkleTree.verify_proof(leaves[index], cpu_proof, cpu_tree.root)
    cpu_verify_dt = time.perf_counter() - t0

    t0 = time.perf_counter()
    gpu_ok = cuda.verify_merkle_proof(leaves[index], gpu_proof, cpu_tree.root)
    gpu_verify_dt = time.perf_counter() - t0

    wrong_leaf = bytes([leaves[index][0] ^ 1]) + leaves[index][1:]
    gpu_wrong_leaf_ok = cuda.verify_merkle_proof(wrong_leaf, gpu_proof, cpu_tree.root)

    proof_match = cpu_proof == gpu_proof

    print(f"  proof length          : {len(cpu_proof)}")
    print(f"  CPU proof verifies    : {cpu_ok}")
    print(f"  GPU proof verifies    : {gpu_ok}")
    print(f"  GPU wrong leaf passes : {gpu_wrong_leaf_ok}")
    print(f"  CPU/GPU proof match   : {proof_match}")
    print(f"  CPU generation        : {cpu_gen_dt*1000:8.2f} ms")
    print(f"  GPU generation        : {gpu_gen_dt*1000:8.2f} ms")
    print(f"  CPU verification      : {cpu_verify_dt*1000:8.2f} ms")
    print(f"  GPU verification      : {gpu_verify_dt*1000:8.2f} ms")
    print(f"  root                  : 0x{cpu_tree.root.hex()}")

    assert cpu_ok, "SHA-3 CPU proof verification failed"
    assert gpu_ok, "SHA-3 GPU proof verification failed"
    assert not gpu_wrong_leaf_ok, "SHA-3 GPU proof accepted the wrong leaf"
    assert proof_match, "SHA-3 CPU/GPU proofs differ"

# ---------------------------------------------------------------------------

def main() -> int:
    # --- Poseidon ---
    print("Initializing GPU Poseidon (uploading constants)...")
    t0 = time.perf_counter()
    cuda_p = CudaPoseidon()
    print(f"  done in {(time.perf_counter()-t0)*1000:.1f} ms\n")

    compare_single_hash(cuda_p)
    print()
    for n in (16, 256, 4096, 65536):
        compare_hash_batch(cuda_p, n)
        print()
    for n_leaves in (1, 5, 16, 1024, 65536):
        compare_merkle(cuda_p, n_leaves)
        print()

        index = min(n_leaves - 1, n_leaves // 2)
        compare_poseidon_proof(cuda_p, n_leaves, index)
        print()

    # --- SHA-3 / Keccak ---
    print("\nInitializing GPU Keccak...")
    t0 = time.perf_counter()
    cuda_k = CudaKeccak()
    print(f"  done in {(time.perf_counter()-t0)*1000:.1f} ms\n")

    compare_keccak_single_hash(cuda_k)
    print()
    for n in (16, 256, 4096, 65536):
        compare_keccak_hash_batch(cuda_k, n)
        print()
    for n_leaves in (1, 5, 16, 1024, 65536):
        compare_keccak_merkle(cuda_k, n_leaves)
        print()
        index = min(n_leaves - 1, n_leaves // 2)
        compare_keccak_proof(cuda_k, n_leaves, index)
        print()


    print("All comparisons passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
