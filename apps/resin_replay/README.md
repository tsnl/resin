# zfw-replay

CLI tool for rendering glTF scenes with camera trajectories using the resin (zfw Draw3d) renderer or Mitsuba.

## Usage

```bash
# Render with resin renderer (default)
echo "camera-1,1,0,0,0,0,1,0,-3,0,0,1,0,0,0,0,1" | zfw-replay model.gltf

# Render with mitsuba renderer
echo "camera-1,1,0,0,0,0,1,0,-3,0,0,1,0,0,0,0,1" | zfw-replay model.gltf --renderers mitsuba

# Render with both renderers for comparison
echo "camera-1,1,0,0,0,0,1,0,-3,0,0,1,0,0,0,0,1" | zfw-replay model.gltf --renderers both

# With environment map
echo "camera-1,1,0,0,0,0,1,0,-3,0,0,1,0,0,0,0,1" | zfw-replay model.gltf --environment-map env.hdr
```

## CSV Format

Each row of the CSV input contains:
- Column 1: Camera ID (kebab-case string, e.g., `camera-front`)
- Columns 2-17: 4x4 camera transform matrix in row-major order

Camera IDs can be repeated across rows for multiple frames in a trajectory.

## Output

Images are saved to `<output>/<camera_id>/color-<frame_idx>-<renderer>.png`.
