"""Helper utilities for image reference testing."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from resin import compute_psnr, logger

LOG = logger(__name__)


def assert_image_matches_reference(
    actual_image: np.ndarray,
    test_name: str,
    psnr_threshold: float,
    test_subdir: str,
) -> None:
    """
    Compare an actual rendered image against an expected reference image.

    If no reference exists, creates it and passes the test.
    If reference exists, compares using PSNR matching.
    On mismatch, saves the actual image as <name>.actual.png and raises AssertionError.

    Args:
        actual_image: The rendered image as a numpy array (H, W, C).
        test_name: Name of the test (used for filename).
        psnr_threshold: Minimum PSNR value required to pass.
        test_subdir: Subdirectory under tests/expect/resin/ and output/resin/ for organizing test images.
    """

    expect_dir = Path("tests/expect/resin") / test_subdir
    expect_dir.mkdir(parents=True, exist_ok=True)

    expect_path = expect_dir / f"{test_name}.png"
    actual_path = expect_dir / f"{test_name}.actual.png"

    # Also save to output directory for convenience
    output_path = Path("output/resin") / test_subdir / f"{test_name}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(output_path, actual_image)

    if not expect_path.exists():
        # No reference exists - create it and pass
        iio.imwrite(expect_path, actual_image)
        LOG.warning(f"{test_name}: no expect found: image created: {expect_path}")
        return

    # Load expected image
    expected_image = iio.imread(expect_path)

    # Check shapes match
    if actual_image.shape != expected_image.shape:
        iio.imwrite(actual_path, actual_image)
        raise AssertionError(
            f"Image shape mismatch for {test_name}: "
            f"expected {expected_image.shape}, got {actual_image.shape}. "
            f"Actual image saved to {actual_path}"
        )

    # Compare images
    psnr = compute_psnr(actual_image, expected_image)
    if psnr < psnr_threshold:
        iio.imwrite(actual_path, actual_image)
        raise AssertionError(
            f"PSNR match failed for {test_name}: "
            f"PSNR {psnr:.2f} < {psnr_threshold}. "
            f"Actual image saved to {actual_path}"
        )

    # If we reach here, the images matched
    # Clean up any previous actual image
    if actual_path.exists():
        actual_path.unlink()
