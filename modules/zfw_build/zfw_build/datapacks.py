"""
This module helps with packing and unpacking tar.zst compressed data archives.

The packed `.tar.zstd` files are checked into version control. They are unpacked at
build time into packages' `bundled_data` directories. We also write a SHA-256 checksum
file adjacent to each unpacked directory to avoid unnecessary re-extraction if the
source archive hasn't changed.
"""

__all__ = [
    "build_datapacks",
]

from pathlib import Path
import tarfile
import hashlib
import shutil

import zstandard as zstd


def build_datapacks(file_path_list: list[Path], dst_parent_dir: Path):
    """
    Extracts a tar.zst archive to a specified destination directory.

    Args:
        file_path (str): The path to the .tar.zst file.
        destination_dir (str): The directory to extract files into.
    """

    for file_path in file_path_list:
        if str(file_path).endswith(".tar.zstd"):
            unpack_compressed_data_file(file_path, dst_parent_dir)
        elif file_path.is_dir():
            copy_data_directory(file_path, dst_parent_dir)
        else:
            raise ValueError(
                f"Data pack path must be a .tar.zstd file or directory: {file_path}"
            )


def unpack_compressed_data_file(file_path: Path, dst_parent_dir: Path):
    # Check if we can skip extraction using a checksum adjacent to the output dir:
    _, dst_sum_path = get_output_dir(file_path, dst_parent_dir)
    current_src_sha256 = read_file_shasum(dst_sum_path)
    new_src_sha256 = compute_file_sha256(file_path)
    if new_src_sha256 == current_src_sha256:
        return

    # Perform extraction
    extract_tar_zstd_eagerly(file_path, dst_parent_dir)

    # Write .src-sha256 file with current SHA-256
    write_file_shasum(dst_sum_path, new_src_sha256)


def copy_data_directory(src_dir: Path, dst_parent_dir: Path):
    output_dir = dst_parent_dir / src_dir.name

    for src_dir_path, _, src_file_names in src_dir.walk():
        dir_path_rel = Path(src_dir_path).relative_to(src_dir)
        dst_dir_path = output_dir / dir_path_rel

        dst_dir_path.mkdir(parents=True, exist_ok=True)

        for file_name in src_file_names:
            src_file_path = Path(src_dir_path) / file_name
            dst_file_path = dst_dir_path / file_name

            if not should_copy_file(src_file_path, dst_file_path):
                continue

            shutil.copy2(src_file_path, dst_file_path)


def should_copy_file(src_file_path: Path, dst_file_path: Path) -> bool:
    """Determine if a file should be copied based on modification times."""
    if not dst_file_path.exists():
        return True
    if src_file_path.stat().st_mtime_ns > dst_file_path.stat().st_mtime_ns:
        return True
    return False


def get_output_dir(file_path: Path, dst_parent_dir: Path) -> tuple[Path, Path]:
    """Get the output directory for an extracted archive."""
    assert str(file_path).endswith(".tar.zstd")
    dst_dir_name = str(file_path.name.replace(".tar.zstd", ""))
    dst_dir = dst_parent_dir / dst_dir_name
    dst_sum_path = dst_dir.with_suffix(".src-sha256")
    return dst_dir, dst_sum_path


def compute_file_sha256(file_path: Path) -> str:
    """Compute the SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def read_file_shasum(file_path: Path) -> str | None:
    """Read the entire contents of a file and return it as bytes."""

    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None


def write_file_shasum(file_path: Path, shasum: str):
    """Write a SHA-256 hash to a file."""
    with open(file_path, "w") as f:
        print(shasum, file=f)


def extract_tar_zstd_eagerly(file_path: Path, dst_parent_dir: Path):
    # Open the compressed file in binary read mode
    # NOTE: Mode "r|" means open for reading without compression.
    with open(file_path, "rb") as in_file:
        with zstd.ZstdDecompressor().stream_reader(in_file) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar_file:
                tar_file.extractall(path=dst_parent_dir)

    # Verify extraction by checking if the destination directory exists
    dst_dir, _ = get_output_dir(file_path, dst_parent_dir)
    if not dst_dir.exists():
        raise RuntimeError(f"Extraction failed, {dst_dir} does not exist.")
