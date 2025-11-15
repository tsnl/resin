# `zero`

A game engine for fun and profit.

```bash
$ VENV_PATH=path/to/your/venv
$ VULKAN_SDK=/path/to/vulkan/sdk

$ source "${VENV_PATH}/bin/activate"
$ source "${VULKAN_SDK}/setup-env.sh"
$ pip install .
$ python3 examples/demo.py
```

```python
import zero
engine = zero.init(zero.Config())
zero.run(engine)
```
