__all__ = [
    "load_shader",
]

import functools
import importlib.resources
import re
from typing import Literal

type ShaderName = Literal[
    "elementwise-binary",
    "elementwise-unary",
    "matmul",
    "test-shader",
]


def load_shader(name: ShaderName, template_consts: dict[str, str]) -> str:
    # Load the shader text:
    text = load_shader_template(name)

    # Perform text substitution for each template const:
    for pp_name, pp_value in template_consts.items():
        text = re.sub(
            rf"""
            \/\* \s* template \s* \*\/  \s*     # /* template */
            const \s+ {pp_name}         \s*     # const <NAME>
            : \s* (?P<tyspec> .+?)      \s*     # : <TYPE>
            =                           \s*     # =
            \s* (?P<value> .+?)         \s*     # <VALUE_PLACEHOLDER>
            ;                                   # ;
            """,
            rf"/* template */ const {pp_name}: \g<tyspec> = {pp_value};",
            text,
            flags=re.VERBOSE,
        )

    # Done:
    return text


@functools.cache
def load_shader_template(name: ShaderName) -> str:
    folder_traversable = importlib.resources.files()
    file_path = folder_traversable.joinpath(f"{name}.wgsl")
    return file_path.read_text()
