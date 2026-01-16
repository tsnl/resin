"""
Offline render script for profiling and benchmarking.

This script renders a scene without a display window, useful for:
- Performance profiling with Chromium traces
- Headless rendering for CI/CD
- Batch rendering jobs

Usage:
    uv run python -m resin_sandbox.offline_render --frames 100 --trace trace.json
"""

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import wgpu

import resin


LOG = logging.getLogger(__name__)


def create_headless_device() -> tuple[wgpu.GPUAdapter, wgpu.GPUDevice]:
    """Create a WebGPU device for headless rendering."""
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = resin.help_request_wgpu_device(adapter)
    return adapter, device


def load_all_models(
    models_path: Path,
    renderer: resin.Draw3dRenderer,
) -> dict[tuple[resin.Draw3dGeometry, resin.Draw3dMaterial], np.ndarray]:
    """Load all GLTF sample models and arrange them in a row."""
    models = [
        ("Avocado", "Avocado/glTF-Binary/Avocado.glb"),
        ("Damaged Helmet", "DamagedHelmet/glTF/DamagedHelmet.gltf"),
        ("Flight Helmet", "FlightHelmet/glTF/FlightHelmet.gltf"),
        ("Water Bottle", "WaterBottle/glTF-Binary/WaterBottle.glb"),
        ("Sci-Fi Helmet", "SciFiHelmet/glTF/SciFiHelmet.gltf"),
        ("Lantern", "Lantern/glTF-Binary/Lantern.glb"),
        ("Antique Camera", "AntiqueCamera/glTF-Binary/AntiqueCamera.glb"),
        ("Boom Box", "BoomBox/glTF-Binary/BoomBox.glb"),
        ("Corset", "Corset/glTF-Binary/Corset.glb"),
        ("Duck", "Duck/glTF-Binary/Duck.glb"),
    ]

    meshes: dict[tuple[resin.Draw3dGeometry, resin.Draw3dMaterial], np.ndarray] = {}
    spacing = 3.0
    x_offset = -spacing * (len(models) - 1) / 2

    for i, (model_name, model_file) in enumerate(models):
        model_path = models_path / model_file

        if not model_path.exists():
            LOG.warning(f"Model not found: {model_path}")
            continue

        LOG.info(f"Loading model {i + 1}/{len(models)}: {model_name}")

        with resin.trace.span(f"load_model/{model_name}", "loading"):
            resource_meshes = resin.load_gltf(model_path)

        # Create translation matrix
        translation = np.eye(4, dtype=np.float32)
        translation[0, 3] = x_offset + i * spacing

        for (geom_res, mat_res), transforms in resource_meshes.items():
            geometry = resin.Draw3dGeometry.from_resource(geom_res, renderer)
            material = resin.Draw3dMaterial.from_resource(mat_res, renderer)

            # Apply translation
            translated_transforms = np.array(
                [translation @ t for t in transforms], dtype=np.float32
            )

            key = (geometry, material)
            if key in meshes:
                meshes[key] = np.concatenate([meshes[key], translated_transforms])
            else:
                meshes[key] = translated_transforms

    total_instances = sum(len(t) for t in meshes.values())
    LOG.info(f"Loaded {len(models)} models with {total_instances} total instances")
    return meshes


def create_camera(frame_index: int, total_frames: int) -> resin.Draw3dCamera:
    """Create a camera that orbits around the scene."""
    # Orbit parameters
    radius = 8.0
    height = 2.0
    angle = (frame_index / total_frames) * 2 * math.pi

    # Camera position
    x = radius * math.sin(angle)
    y = radius * math.cos(angle)
    z = height

    # Look at origin
    forward = np.array([-x, -y, -z], dtype=np.float32)
    forward = forward / np.linalg.norm(forward)

    # Compute up and right
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    # Build transform
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 0] = right
    transform[:3, 1] = forward
    transform[:3, 2] = up
    transform[:3, 3] = [x, y, z]

    return resin.Draw3dCamera(
        transform=transform,
        fov_y_rad=math.radians(60),
        aspect_ratio=16 / 9,
        max_distance=20.0,
    )


def run_offline_render(
    num_frames: int,
    output_trace: Path | None,
    resolution: tuple[int, int],
    models_path: Path,
    environments_path: Path,
) -> None:
    """Run offline rendering with profiling."""

    # Enable tracing if requested
    if output_trace:
        resin.enable_chromium_trace(output_trace)
        LOG.info(f"Chromium tracing enabled, will save to {output_trace}")

    # Create device
    LOG.info("Creating WebGPU device...")
    with resin.trace.span("create_device", "setup"):
        adapter, device = create_headless_device()

    LOG.info(f"Using adapter: {adapter.info}")

    # Create renderer
    LOG.info(f"Creating renderer at {resolution[0]}x{resolution[1]}...")
    with resin.trace.span("create_renderer", "setup"):
        renderer = resin.Draw3dRenderer(
            device=device,
            queue=device.queue,
            target_size_wh_px=resolution,
            samples_per_pixel=1,
            accumulator_frame_count=1,
            render_scale=1.0,
        )

    # Load models
    LOG.info("Loading models...")
    with resin.trace.span("load_all_models", "loading"):
        meshes = load_all_models(models_path, renderer)

    if not meshes:
        LOG.error("No models loaded, exiting")
        return

    # Load environment
    env_path = environments_path / "neutral.hdr"
    environment_texture: resin.Draw3dTexture | None = None
    if env_path.exists():
        LOG.info(f"Loading environment: {env_path}")
        with resin.trace.span("load_environment", "loading"):
            env_image = resin.load_image(env_path)
            environment_texture = resin.Draw3dTexture(
                renderer, data=env_image, usage="environment"
            )
    else:
        LOG.warning(f"Environment not found: {env_path}")

    # Create initial camera for scene construction
    initial_camera = create_camera(0, num_frames)

    # Create scene
    scene = resin.Draw3dScene(
        camera=initial_camera,
        meshes=meshes,
        environment_map=environment_texture,
    )

    # Render frames
    LOG.info(f"Rendering {num_frames} frames...")
    frame_times: list[float] = []

    for frame_idx in range(num_frames):
        frame_start = time.perf_counter()

        with resin.trace.span(f"frame/{frame_idx}", "frame", args={"frame": frame_idx}):
            # Update camera
            scene.camera = create_camera(frame_idx, num_frames)

            # Create command encoder
            encoder = device.create_command_encoder()

            # Record render commands
            renderer.record(scene, encoder)

            # Submit
            with resin.trace.span("gpu_submit", "gpu"):
                device.queue.submit([encoder.finish()])

            # Wait for GPU (for accurate timing)
            with resin.trace.span("gpu_wait", "gpu"):
                device.queue.submit([])  # Flush
                # Note: wgpu-py doesn't have explicit sync, this is approximate

        frame_end = time.perf_counter()
        frame_time = frame_end - frame_start
        frame_times.append(frame_time)

        # Log progress every 10 frames
        if (frame_idx + 1) % 10 == 0:
            fps = 1.0 / frame_time if frame_time > 0 else 0
            LOG.info(
                f"Frame {frame_idx + 1}/{num_frames}: {frame_time * 1000:.2f}ms ({fps:.1f} FPS)"
            )

    # Report statistics
    frame_times_arr = np.array(frame_times)
    LOG.info("=" * 60)
    LOG.info("Render Statistics:")
    LOG.info(f"  Total frames: {num_frames}")
    LOG.info(f"  Mean frame time: {frame_times_arr.mean() * 1000:.2f}ms")
    LOG.info(f"  Median frame time: {np.median(frame_times_arr) * 1000:.2f}ms")
    LOG.info(f"  P5 frame time: {np.percentile(frame_times_arr, 5) * 1000:.2f}ms")
    LOG.info(f"  P95 frame time: {np.percentile(frame_times_arr, 95) * 1000:.2f}ms")
    LOG.info(f"  Min frame time: {frame_times_arr.min() * 1000:.2f}ms")
    LOG.info(f"  Max frame time: {frame_times_arr.max() * 1000:.2f}ms")
    LOG.info(f"  Effective FPS: {1.0 / frame_times_arr.mean():.1f}")
    LOG.info("=" * 60)

    # Log trace statistics
    resin.trace.log(truncate_window_sec=None)

    # Save trace (will also happen on exit if atexit was registered)
    if output_trace:
        resin.save_chromium_trace(output_trace)

    # Cleanup
    renderer.dispose()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline render for profiling and benchmarking"
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=100,
        help="Number of frames to render (default: 100)",
    )
    parser.add_argument(
        "--trace",
        type=Path,
        default=None,
        help="Output path for Chromium trace file (e.g., trace.json)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Render width in pixels (default: 1920)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Render height in pixels (default: 1080)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Setup logging
    resin.setup_logging(
        level=logging.DEBUG if args.debug else logging.INFO,
        console=True,
    )

    # Find data paths
    workspace_root = Path(__file__).parent.parent.parent.parent.parent
    models_path = workspace_root / "tests" / "data" / "glTF-Sample-Assets" / "Models"
    environments_path = workspace_root / "tests" / "data" / "glTF-Sample-Environments"

    if not models_path.exists():
        LOG.error(f"Models path not found: {models_path}")
        LOG.error("Please ensure glTF sample assets are available in tests/data/")
        return

    # Run render
    run_offline_render(
        num_frames=args.frames,
        output_trace=args.trace,
        resolution=(args.width, args.height),
        models_path=models_path,
        environments_path=environments_path,
    )


if __name__ == "__main__":
    main()
