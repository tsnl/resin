"""Vulkan SDK download and management utilities."""

import hashlib
import platform
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Literal
import requests
from requests.exceptions import RequestException
from rich.progress import (
    Progress,
    BarColumn,
    DownloadColumn,
    TextColumn,
    TimeElapsedColumn,
)
from setup.log_util import get_logger

logger = get_logger(__name__)


BUILD_DIR = Path("build") / "vulkan-sdk"

PlatformType = Literal["linux", "mac", "windows"]


def _get_platform() -> PlatformType:
    """Get the current platform identifier."""
    system = platform.system().lower()
    if system == "linux":
        return "linux"
    elif system == "darwin":
        return "mac"
    elif system == "windows":
        return "windows"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def _get_download_url(
    vulkan_version: str,
    platform_name: PlatformType,
) -> tuple[str, str]:
    """
    Get the download URL and file extension for the Vulkan SDK.

    Returns:
        tuple[str, str]: (url, extension) where extension is "tar.xz", "zip"
    """
    base_url = "https://sdk.lunarg.com/sdk/download"

    if platform_name == "linux":
        # Linux uses tar.xz
        url = f"{base_url}/{vulkan_version}/linux/vulkan_sdk.tar.xz"
        return (url, "tar.xz")
    elif platform_name == "mac":
        # macOS uses zip
        url = f"{base_url}/{vulkan_version}/mac/vulkan_sdk.zip"
        return (url, "zip")
    elif platform_name == "windows":
        # Windows uses zip
        url = f"{base_url}/{vulkan_version}/windows/vulkan_sdk.exe"
        return (url, "zip")
    else:
        raise ValueError(f"Unknown platform: {platform_name}")


def _get_checksum_url(vulkan_version: str, platform_name: PlatformType) -> str:
    """Get the checksum URL for the Vulkan SDK."""
    base_url = "https://sdk.lunarg.com/sdk/sha"

    if platform_name == "linux":
        return f"{base_url}/{vulkan_version}/linux/vulkan_sdk.tar.xz.txt"
    elif platform_name == "mac":
        return f"{base_url}/{vulkan_version}/mac/vulkan_sdk.zip.txt"
    elif platform_name == "windows":
        return f"{base_url}/{vulkan_version}/windows/vulkan_sdk.exe.txt"
    else:
        raise ValueError(f"Unknown platform: {platform_name}")


def _download_file(url: str, dest_path: Path) -> None:
    """Download a file from URL to destination path with progress indication."""
    logger.info(f"Downloading {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            total_size = int(response.headers.get("Content-Length", 0) or 0)

            with open(dest_path, "wb") as f:
                if total_size:
                    # Use rich.progress for a nicer download bar
                    with Progress(
                        TextColumn("{task.description}"),
                        DownloadColumn(),
                        BarColumn(),
                        "[progress.percentage]{task.percentage:>3.1f}%",
                        TimeElapsedColumn(),
                        transient=True,
                    ) as progress:
                        task = progress.add_task("Downloading", total=total_size)
                        for chunk in response.iter_content(chunk_size=8192):
                            if not chunk:
                                continue
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))
                else:
                    # Unknown size, stream without progress bar
                    for chunk in response.iter_content(chunk_size=8192):
                        if not chunk:
                            continue
                        f.write(chunk)

    except RequestException as exc:
        logger.error(f"Download failed: {exc}")
        raise

    logger.info(f"Downloaded to {dest_path}")


def _verify_checksum(file_path: Path, checksum_path: Path) -> bool:
    """Verify the SHA256 checksum of a file."""
    logger.info(f"Verifying checksum for {file_path.name}...")

    # Read the expected checksum (format: "hash  filename")
    with open(checksum_path, "r") as f:
        checksum_line = f.read().strip()
        # Handle both "hash filename" and just "hash" formats
        expected_checksum = checksum_line.split()[0] if checksum_line else ""

    # Calculate actual checksum
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)

    actual_checksum = sha256.hexdigest()

    if actual_checksum.lower() == expected_checksum.lower():
        logger.info("Checksum verified")
        return True
    else:
        logger.error("✗ Checksum mismatch")
        logger.error(f"  Expected: {expected_checksum}")
        logger.error(f"  Actual:   {actual_checksum}")
        return False


def _extract_archive(archive_path: Path, dest_dir: Path, extension: str) -> None:
    """Extract a tar.xz, zip, or exe archive."""
    logger.info(f"Extracting {archive_path.name}...")
    dest_dir.mkdir(parents=True, exist_ok=True)

    if extension == "tar.xz":
        with tarfile.open(archive_path, "r:xz") as tar:
            tar.extractall(dest_dir)
    elif extension == "zip":
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive extension: {extension}")

    logger.info(f"Extracted to {dest_dir}")


def fetch(vulkan_version: str) -> None:
    """
    Download and extract the Vulkan SDK for the current platform.

    Args:
        vulkan_version: Version string like "1.4.238.1"
    """
    platform_name = _get_platform()

    # Get download URLs and extension
    download_url, extension = _get_download_url(vulkan_version, platform_name)
    checksum_url = _get_checksum_url(vulkan_version, platform_name)

    # Set up paths
    archive_filename = f"{vulkan_version}.{extension}"
    archive_path = BUILD_DIR / archive_filename
    checksum_path = BUILD_DIR / f"{archive_filename}.sha256.txt"
    download_done_marker = BUILD_DIR / f"{archive_filename}.DONE"

    # Archive contains a top-level directory with the version, so extract to platform_dir
    extract_dir = BUILD_DIR / vulkan_version
    extract_done_marker = BUILD_DIR / f"{vulkan_version}.DONE"

    # Step 1: Download archive and checksum if not already done
    if download_done_marker.exists():
        logger.info(f"Archive already downloaded: {archive_path}")
    else:
        # Download archive
        _download_file(download_url, archive_path)

        # Download checksum
        _download_file(checksum_url, checksum_path)

        # Verify checksum
        if not _verify_checksum(archive_path, checksum_path):
            logger.error("Checksum verification failed")
            sys.exit(1)

        # Mark download as complete
        download_done_marker.touch()

    # Step 2: Extract archive if not already done
    if extract_done_marker.exists():
        logger.info(f"Archive already extracted: {extract_dir}")
    else:
        # Extract to BUILD_DIR
        _extract_archive(archive_path, BUILD_DIR, extension)

        # Mark extraction as complete
        extract_done_marker.touch()
        logger.info("Setup complete")
    logger.info(f"Vulkan SDK {vulkan_version} is ready at: {extract_dir}")


def clean(vulkan_version: str) -> None:
    """
    Clean the extracted Vulkan SDK (but keep the downloaded archive).

    Args:
        vulkan_version: Version string like "1.4.238.1"
    """
    platform_name = _get_platform()
    platform_dir = BUILD_DIR / platform_name

    extract_dir = platform_dir / vulkan_version
    extract_done_marker = platform_dir / f"{vulkan_version}.DONE"

    # Remove extraction directory
    if extract_dir.exists():
        logger.info(f"Removing {extract_dir}...")
        shutil.rmtree(extract_dir)

    # Remove extraction done marker
    if extract_done_marker.exists():
        extract_done_marker.unlink()
