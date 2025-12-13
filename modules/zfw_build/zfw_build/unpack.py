from pathlib import Path
import tarfile
import hashlib

import zstandard as zstd


def extract_tar_zstd(file_path: Path, dst_parent_dir: Path):
    """
    Extracts a tar.zst archive to a specified destination directory.

    Args:
        file_path (str): The path to the .tar.zst file.
        destination_dir (str): The directory to extract files into.
    """

    assert str(file_path).endswith(".tar.zstd")
    assert file_path.exists() and file_path.is_file()

    # Check if we can skip extraction: if the SHA-256 matches the DONE file
    dst_dir = get_output_dir(file_path, dst_parent_dir)
    if compute_file_sha256(file_path) == read_file_shasum(dst_dir / "DONE"):
        return

    # Perform extraction
    extract_tar_zstd_eagerly(file_path, dst_parent_dir)

    # Write DONE file with current SHA-256
    write_file_shasum(dst_dir / "DONE", compute_file_sha256(file_path))


def get_output_dir(file_path: Path, dst_parent_dir: Path) -> Path:
    """Get the output directory for an extracted archive."""
    assert str(file_path).endswith(".tar.zstd")
    dst_dir_name = str(file_path.name.replace(".tar.zstd", ""))
    return dst_parent_dir / dst_dir_name


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
    dst_dir = get_output_dir(file_path, dst_parent_dir)
    if not dst_dir.exists():
        raise RuntimeError(f"Extraction failed, {dst_dir} does not exist.")
