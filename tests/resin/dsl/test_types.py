import pytest

from resin.dsl.dsl import View
from resin.dsl.types import (
    PodSpec,
    Tensor,
    TensorMeta,
    TensorSpec,
    annotation_to_spec,
    parse_signature,
)
from .fixtures import LinearParams, mlp_step


class TestAnnotationToSpec:
    def test_tensor(self) -> None:
        hint = Tensor["f4", (3, 4)]
        spec = annotation_to_spec(hint)
        assert spec == TensorSpec("f4", (3, 4))

    def test_tensor_meta_directly(self) -> None:
        from typing import Annotated

        hint = Annotated[View, TensorMeta("f4", (2,))]
        spec = annotation_to_spec(hint)
        assert spec == TensorSpec("f4", (2,))

    def test_typed_dict(self) -> None:
        spec = annotation_to_spec(LinearParams)
        assert spec == {
            "weight": TensorSpec("f4", (10, 784)),
            "bias": TensorSpec("f4", (10,)),
        }

    def test_tuple(self) -> None:
        hint = tuple[Tensor["f4", (2,)], Tensor["f4", (3,)]]
        spec = annotation_to_spec(hint)
        assert spec == (
            TensorSpec("f4", (2,)),
            TensorSpec("f4", (3,)),
        )

    def test_list(self) -> None:
        hint = list[Tensor["f4", (4,)]]
        spec = annotation_to_spec(hint)
        assert spec == [TensorSpec("f4", (4,))]

    def test_pod_int(self) -> None:
        assert annotation_to_spec(int) == PodSpec(int)

    def test_bare_dict_rejected(self) -> None:
        with pytest.raises(ValueError, match="TypedDict"):
            annotation_to_spec(dict)


class TestParseSignature:
    def test_mlp_step(self) -> None:
        spec = parse_signature(mlp_step)
        assert set(spec.args.keys()) == {"x", "y", "params"}
        assert spec.return_spec == TensorSpec("f4", ())

    def test_unannotated_param_raises(self) -> None:
        def bad(x) -> Tensor["f4", ()]:  # type: ignore[valid-type]
            return x

        with pytest.raises(ValueError, match="unannotated parameter"):
            parse_signature(bad)

    def test_unannotated_return_raises(self) -> None:
        def bad(x: Tensor["f4", ()]):  # noqa: ANN202
            return x

        with pytest.raises(ValueError, match="unannotated return"):
            parse_signature(bad)

    def test_varargs_rejected(self) -> None:
        def bad(*xs: Tensor["f4", ()]) -> Tensor["f4", ()]:  # type: ignore[valid-type]
            return xs[0]

        with pytest.raises(ValueError, match="\\*args"):
            parse_signature(bad)