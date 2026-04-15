#!/usr/bin/env python3
"""Convert GloVe text embeddings to VECTRO+STREAM1 binary format.

The GloVe text format is: <word> <f1> <f2> ... <fd>
The STREAM1 wire format is:
  [15-byte header: "VECTRO+STREAM1\\n"]
  repeated: [u32 LE payload_len][bincode 1.3 Embedding payload]

bincode 1.3 Embedding layout:
  [u64 LE id_byte_len][id UTF-8 bytes][u64 LE vec_len][f32 LE values ...]

Usage:
    python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1
    python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1 --max-vectors 50000
    python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1 --quiet

Exit codes:
    0  success
    1  user / input error
    2  runtime / I/O error
"""

import argparse
import os
import struct
import sys

STREAM1_HEADER = b"VECTRO+STREAM1\n"


def _bincode_embedding(id_str: str, vector: list) -> bytes:
    """Serialize an Embedding to bincode 1.3 wire format.

    bincode 1.3 encodes sequences with a u64 LE length prefix, then payload:
      - String  → [u64 LE byte_len][UTF-8 bytes]
      - Vec<f32> → [u64 LE element_count][f32 LE values ...]
    """
    id_bytes = id_str.encode("utf-8")
    n = len(vector)
    return (
        struct.pack("<Q", len(id_bytes))       # u64 LE: byte length of id
        + id_bytes                              # UTF-8 id
        + struct.pack("<Q", n)                  # u64 LE: number of f32 elements
        + struct.pack(f"<{n}f", *vector)        # f32 LE values
    )


def convert(input_path: str, output_path: str, max_vectors: int | None, quiet: bool) -> int:
    """Convert a GloVe text file to VECTRO+STREAM1 binary.

    Returns an exit code: 0 = success, 1 = input error, 2 = runtime error.
    """
    if not os.path.exists(input_path):
        print(f"error: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        with (
            open(input_path, "r", encoding="utf-8", errors="replace") as fin,
            open(output_path, "wb") as fout,
        ):
            fout.write(STREAM1_HEADER)
            count = 0
            skipped = 0

            for line_no, line in enumerate(fin, start=1):
                if max_vectors is not None and count >= max_vectors:
                    break

                line = line.rstrip("\n")
                if not line:
                    continue

                # GloVe format: word sp f1 sp f2 ... sp fd
                space = line.index(" ") if " " in line else -1
                if space == -1:
                    skipped += 1
                    continue

                id_str = line[:space]
                raw_floats = line[space + 1 :]

                try:
                    vector = list(map(float, raw_floats.split()))
                except ValueError:
                    if not quiet:
                        print(
                            f"warning: skipping line {line_no} (parse error)", file=sys.stderr
                        )
                    skipped += 1
                    continue

                if not vector:
                    skipped += 1
                    continue

                payload = _bincode_embedding(id_str, vector)
                fout.write(struct.pack("<I", len(payload)))  # u32 LE outer length prefix
                fout.write(payload)
                count += 1

                if not quiet and count % 100_000 == 0:
                    print(f"  converted {count:,} vectors ...", flush=True)

    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not quiet:
        print(f"✓ wrote {count:,} vectors → {output_path}")
        if skipped:
            print(f"  ({skipped} lines skipped)")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="convert_glove_to_stream1",
        description="Convert GloVe text embeddings to VECTRO+STREAM1 binary format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1
  python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1 --max-vectors 50000
  python scripts/convert_glove_to_stream1.py glove.6B.100d.txt glove100d.stream1 --quiet

Download GloVe-100d:
  https://nlp.stanford.edu/data/glove.6B.zip
  unzip glove.6B.zip glove.6B.100d.txt
""",
    )
    parser.add_argument("input", help="Path to GloVe text file (e.g. glove.6B.100d.txt)")
    parser.add_argument("output", help="Output path for STREAM1 binary (e.g. glove100d.stream1)")
    parser.add_argument(
        "--max-vectors",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of vectors to convert (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output (errors still go to stderr)",
    )

    args = parser.parse_args()
    sys.exit(convert(args.input, args.output, args.max_vectors, args.quiet))


if __name__ == "__main__":
    main()
