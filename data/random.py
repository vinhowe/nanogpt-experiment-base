"""
Random token dataset generator.

Generates randint-sampled tokens (uniform in [0, vocab_size)) stored as raw uint16
binaries compatible with the training loader (no header):
- train.bin
- val.bin

The output directory is named based on parameters, e.g.:
  data/random_V2048_L64_N65536/

CLI parameters:
- --vocab (default 2048)
- --seq_len (default 64)
- --num_seqs (default 2**16)

Notes:
- Tokens are stored as uint16. vocab must be <= 65536.
- A fixed 10% of sequences are used for validation.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict

import numpy as np


def write_raw_uint16(filename: str, tokens: np.ndarray) -> None:
    """Write tokens (uint16) to a flat binary file with no header."""
    if not isinstance(tokens, np.ndarray):
        raise TypeError("tokens must be a numpy array")
    if tokens.dtype != np.uint16:
        raise TypeError("tokens must have dtype uint16")
    with open(filename, "wb") as f:
        f.write(tokens.tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate random token dataset")
    parser.add_argument(
        "--vocab", type=int, default=2048, help="Vocabulary size (<= 65536)"
    )
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument(
        "--num_seqs", type=int, default=2**16, help="Number of sequences"
    )
    args = parser.parse_args()

    if args.vocab <= 0 or args.seq_len <= 0 or args.num_seqs <= 0:
        raise ValueError("All parameters must be positive integers.")
    if args.vocab > 2**16:
        raise ValueError("vocab must be <= 65536 to fit uint16")

    # Name output directory based on parameters
    out_dir_name = f"random_V{args.vocab}_L{args.seq_len}_N{args.num_seqs}"
    out_dir = os.path.join(os.path.dirname(__file__), out_dir_name)
    os.makedirs(out_dir, exist_ok=True)

    # Split sequences into train/val (10% val, at least 1 sequence)
    val_fraction = 0.10
    num_val_seqs = max(1, int(args.num_seqs * val_fraction))
    num_train_seqs = args.num_seqs - num_val_seqs

    # Fixed seed for reproducibility
    rng = np.random.default_rng(1337)

    def sample_tokens(num_sequences: int) -> np.ndarray:
        token_count = num_sequences * args.seq_len
        tokens = rng.integers(low=0, high=args.vocab, size=token_count, dtype=np.uint16)
        return tokens

    # Generate and write
    train_tokens = sample_tokens(num_train_seqs)
    val_tokens = sample_tokens(num_val_seqs)

    write_raw_uint16(os.path.join(out_dir, "train.bin"), train_tokens)
    write_raw_uint16(os.path.join(out_dir, "val.bin"), val_tokens)

    # Store parameters and basic metadata in JSON
    meta: Dict[str, object] = {
        "vocab_size": int(args.vocab),
        "seq_len": int(args.seq_len),
        "num_seqs": int(args.num_seqs),
        "num_train_seqs": int(num_train_seqs),
        "num_val_seqs": int(num_val_seqs),
        "tokens_train": int(train_tokens.size),
        "tokens_val": int(val_tokens.size),
        "dtype": "uint16",
        "seed": 1337,
        "val_fraction": val_fraction,
        "file_format": "raw-uint16",
        "output_dir": out_dir_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": "data/random.py",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote dataset to {out_dir}")
    print(f"  train.bin: {train_tokens.size:,} tokens")
    print(f"  val.bin:   {val_tokens.size:,} tokens")


if __name__ == "__main__":
    main()
