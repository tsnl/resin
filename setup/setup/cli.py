#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""Setup script for managing Vulkan SDK and other dependencies."""

import argparse
import sys

import setup.vulkan_sdk as vulkan_sdk
from setup.log_util import get_logger

logger = get_logger(__name__)


def cmd_setup(args: argparse.Namespace) -> int:
    """Run the setup command."""
    logger.info(f"Setting up Vulkan SDK version {args.vulkan_sdk_version}...")
    vulkan_sdk.fetch(args.vulkan_sdk_version)
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Run the clean command."""
    logger.info(f"Cleaning Vulkan SDK version {args.vulkan_sdk_version}...")
    vulkan_sdk.clean(args.vulkan_sdk_version)
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Setup and manage project dependencies"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Setup command
    setup_parser = subparsers.add_parser(
        "setup", help="Download and extract dependencies"
    )
    setup_parser.add_argument(
        "--vulkan-sdk-version",
        required=True,
        help="Vulkan SDK version to download (e.g., 1.4.238.1)",
    )
    setup_parser.set_defaults(func=cmd_setup)

    # Clean command
    clean_parser = subparsers.add_parser(
        "clean", help="Clean extracted dependencies (keep downloaded archives)"
    )
    clean_parser.add_argument(
        "--vulkan-sdk-version",
        required=True,
        help="Vulkan SDK version to clean (e.g., 1.4.238.1)",
    )
    clean_parser.set_defaults(func=cmd_clean)

    # Parse arguments
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # Execute the command
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
