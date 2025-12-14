import argparse
from pathlib import Path
import tarfile

import zstandard as zstd


def main():
    ap = argparse.ArgumentParser(
        description="Pack compressed data for zfw projects.",
    )
    ap.add_argument("input", type=Path, help="Directory containing data to be packed")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output parent directory for the packed data (defaults to cwd)",
        default=".",
    )
    args = ap.parse_args()

    pack(input_path=args.input, output_dir=args.output)


def pack(input_path: Path, output_dir: Path):
    if not input_path.is_dir():
        raise ValueError(f"Input path {input_path} is not a directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / (input_path.with_suffix(".tar.zstd").name)

    with open(output_path, "wb") as f_out:
        with zstd.ZstdCompressor(level=22).stream_writer(f_out) as compressor:
            with tarfile.open(fileobj=compressor, mode="w") as tar:
                tar.add(input_path, arcname=input_path.name)
