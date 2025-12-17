- Use `make` targets to do anything if possible
  - `make sync` runs codgen via `make build` and then runs `uv sync`.
  - `make tests` runs all tests and writes test outputs (e.g. image renders).
  - `make sandbox` launches a windowed interactive application.
- Ensure you use `uv` to activate a Python environment with the right dependencies.
- (on macOS) Run `source ~/VulkanSDK/1.4.328.1/setup-env.sh` before running any `make` targets.
- Always use Python3.12+ type statements instead of type aliases.
  ```
  # DO NOT DO THIS:
  PrimaryColor = Literal["red", "green", "blue"]

  # DO THIS INSTEAD:
  type PrimaryColor = Literal["red", "green", "blue"]
  ```
