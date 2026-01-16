"""
Resin Editor CLI - launches the GLTF viewer with immediate-mode GUI.
"""

import argparse
import logging
import time
from pathlib import Path

import wgpu
import resin

from .gltf_viewer import GltfViewer


LOG = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Resin Editor - GLTF Viewer")
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (verbose logging)",
    )
    args = ap.parse_args()

    # Setup logging
    resin.setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        console=True,
        file=Path("resin.log"),
    )

    LOG.info(f"Starting Resin Editor: debug={args.debug}")

    # Create WebGPU device
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = resin.help_request_wgpu_device(adapter)

    # Create window context (manages GLFW)
    window_context = resin.WindowContext()

    # Create window
    window = resin.Window(
        device=device,
        window_context=window_context,
        width_dip=1280,
        height_dip=720,
        title="Resin Editor - GLTF Viewer",
    )

    # Get paths for models and environments
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    models_path = workspace_root / "tests" / "data" / "glTF-Sample-Assets" / "Models"
    environments_path = workspace_root / "tests" / "data" / "glTF-Sample-Environments"

    # Create viewer
    viewer = GltfViewer(
        window=window,
        device=device,
        models_path=models_path,
        environments_path=environments_path,
    )

    window.show()

    # Main loop with frame timing
    last_time = time.perf_counter()
    while not window.should_close():
        current_time = time.perf_counter()
        dt = current_time - last_time
        last_time = current_time

        viewer.run(dt)

    # Cleanup
    viewer.dispose()
    window.dispose()
    window_context.dispose()


if __name__ == "__main__":
    main()
