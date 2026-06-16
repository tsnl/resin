__all__ = [
    "Kernel",
]

import functools
import importlib.resources
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

from ..scalar import (
    BinaryCompareOperator,
    BinaryScalarOperator,
    ScalarType,
    UnaryScalarOperator,
    spell_stype_in_wgsl,
)

#
# Kernel
#


@dataclass(frozen=True, kw_only=True)
class Kernel(ABC):
    """
    Kernels are GPU programs that can be dispatched for evaluation.

    They can be reified as WGSL for dispatch or analyzed and replaced during
    optimizations like kernel fusion.
    """

    @abstractmethod
    def _wgsl_template(self) -> tuple[Template, TemplateParams]: ...

    def wgsl(self) -> str:
        """
        Renders the shader into a WGSL string by loading the appropriate template and
        rendering it with the provided template parameters.
        """

        template_name, template_params = self._wgsl_template()
        template = _load_template_string(template_name)
        return _render_shader_template_with_params(template, template_params)


@dataclass(frozen=True, kw_only=True)
class ElementwiseUnaryKernel(Kernel):
    selected_uop: UnaryScalarOperator
    n: int
    t: ScalarType

    UNARY_UOP_CODE_DICT: dict[UnaryScalarOperator, int] = field(
        # Mapping defined in WGSL template file "elementwise-unary.wgsl"
        default_factory=lambda: {
            "neg": 0,
            "exp": 1,
            "log": 2,
            "not": 3,
            "sin": 4,
            "cos": 5,
        },
        init=False,
        repr=False,
    )

    def _wgsl_template(self) -> tuple[Template, TemplateParams]:
        return (
            "elementwise-unary",
            TemplateParams(
                consts={
                    "SELECTED_UOP": f"{self.UNARY_UOP_CODE_DICT[self.selected_uop]}u",
                    "N": f"{self.n}u",
                },
                types={
                    "T": spell_stype_in_wgsl(self.t),
                },
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ElementwiseBinaryKernel(Kernel):
    selected_bop: BinaryScalarOperator | BinaryCompareOperator
    n: int
    t: ScalarType

    SCALAR_BOP_CODE_DICT: dict[BinaryScalarOperator | BinaryCompareOperator, int] = (
        field(
            # Mapping defined in WGSL template file "elementwise-binary.wgsl"
            default_factory=lambda: {
                "pow": 0,
                "div": 1,
                "sub": 2,
                "mul": 3,
                "add": 4,
                "max": 5,
                "min": 6,
                "eq": 7,
                "ne": 8,
                "gt": 9,
                "lt": 10,
                "ge": 11,
                "le": 12,
            },
            init=False,
            repr=False,
        )
    )

    def _wgsl_template(self) -> tuple[Template, TemplateParams]:
        return (
            "elementwise-binary",
            TemplateParams(
                consts={
                    "SELECTED_BOP": f"u32({self.SCALAR_BOP_CODE_DICT[self.selected_bop]})",
                    "N": f"u32({self.n})",
                },
                types={
                    "T": spell_stype_in_wgsl(self.t),
                },
            ),
        )


@dataclass(frozen=True, kw_only=True)
class MatmulKernel(Kernel):
    m: int
    n: int
    k: int
    t: ScalarType

    def _wgsl_template(self) -> tuple[Template, TemplateParams]:
        return (
            "matmul",
            TemplateParams(
                consts={
                    "M": f"u32({self.m})",
                    "N": f"u32({self.n})",
                    "K": f"u32({self.k})",
                },
                types={
                    "T": spell_stype_in_wgsl(self.t),
                },
            ),
        )


@dataclass(frozen=True, kw_only=True)
class Test1Kernel(Kernel):
    test_constant: int

    def _wgsl_template(self) -> tuple[Template, TemplateParams]:
        return (
            "test1",
            TemplateParams(
                consts={"TEST_CONSTANT": f"u32({self.test_constant})"},
                types={},
            ),
        )


#
# ShaderTemplate
#


type Template = Literal[
    "elementwise-binary",
    "elementwise-unary",
    "matmul",
    "test1",
]
"""
Names of shader templates that can be loaded from disk.
Should reflect the on-disk WGSL files bundled alongside this file.
"""


@dataclass
class TemplateParams:
    consts: dict[str, str]
    types: dict[str, str]


@functools.cache
def _load_template_string(template_name: Template) -> str:
    folder_traversable = importlib.resources.files()
    file_path = folder_traversable.joinpath(f"{template_name}.wgsl")
    return file_path.read_text()


def _render_shader_template_with_params(template: str, params: TemplateParams) -> str:
    # Initialize the text to be rendered:
    text = template

    # Perform text substitution for each template const:
    for const_name, const_value in params.consts.items():
        text = re.sub(
            rf"""
            \/\* \s* template \s* \*\/  \s*     # /* template */
            const \s+ {const_name}      \s*     # const <NAME>
            =                           \s*     # =
            (?P<value> .+?)             \s*     # <VALUE_PLACEHOLDER>
            ;                                   # ;
            """,
            rf"/* template */ const {const_name} = {const_value};",
            text,
            flags=re.VERBOSE,
        )

    # Perform text substitution for each template alias:
    for alias_name, alias_value in params.types.items():
        text = re.sub(
            rf"""
            \/\* \s* template \s* \*\/  \s*     # /* template */
            alias \s+ {alias_name}      \s*     # alias <NAME>
            =                           \s*     # =
            (?P<value> .+?)             \s*     # <VALUE_PLACEHOLDER>
            ;                                   # ;
            """,
            rf"/* template */ type {alias_name} = {alias_value};",
            text,
            flags=re.VERBOSE,
        )

    # Done:
    return text
