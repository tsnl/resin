# `TODO`

Goal: build a Soulslike game-engine using ML policies to accelerate game-dev.

Components:
- Renderer
- Physics
- Editor

Tasks:
- [ ] Renderer 
  - [ ] Clean up existing `draw_3d` renderer implementation.
    - [x] Switch to single buffer accumulator, always increment.
    - [ ] Separate postprocessing pipeline into separate compute shader entry point.
    - [ ] Continue factoring megakernel into smaller kernels, e.g. wavefront.
  - [ ] Refactor: break into small, composable functors.
  - [x] Add TLAS to ray-trace operator to support larger scenes.
  - [ ] Train de-noiser to improve render quality.
- [ ] Physics
  - [ ] Constrained rigid-body dynamics solver.
  - [ ] Train walking policy to operate in game-engine.
- [ ] Editor
  - [ ] MCP core
  - [ ] GUI around MCP core

