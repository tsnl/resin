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
    text = _load_shader_template(name)

    # Perform text substitution for each template const:
    text = _render_shader_template(text, template_consts)

    # Done:
    return text


@functools.cache
def _load_shader_template(name: ShaderName) -> str:
    folder_traversable = importlib.resources.files()
    file_path = folder_traversable.joinpath(f"{name}.wgsl")
    return file_path.read_text()


def _render_shader_template(template_text: str, template_consts: dict[str, str]) -> str:
    text = template_text

    for const_name, const_value in template_consts.items():
        text = re.sub(
            rf"""
            \/\* \s* template \s* \*\/  \s*     # /* template */
            const \s+ {const_name}      \s*     # const <NAME>
            : \s* (?P<tyspec> .+?)      \s*     # : <TYPE>
            =                           \s*     # =
            (?P<value> .+?)             \s*     # <VALUE_PLACEHOLDER>
            ;                                   # ;
            """,
            rf"/* template */ const {const_name}: \g<tyspec> = {const_value};",
            text,
            flags=re.VERBOSE,
        )

    return text
