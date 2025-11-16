# `zero`

A game engine for fun and profit.

```bash
$ VENV_PATH=path/to/your/venv
$ VULKAN_SDK=/path/to/vulkan/sdk    # use setup-env.sh in the SDK tarball

$ source "${VENV_PATH}/bin/activate"
$ pip install .
$ python3 examples/demo.py
```

```python
import zero
engine = zero.init(zero.Config())
zero.run(engine)
```

## NixOS Development

```bash
# Setup the Vulkan SDK environment.
$ source ~/VulkanSDK/1.4.328.1/setup-env.sh

# Activate the Nix shell
$ nix-shell

# Launch zed, BUT KEEP THE NIX SHELL OPEN.
$ zeditor .
```
