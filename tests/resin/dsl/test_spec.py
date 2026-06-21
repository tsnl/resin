import pytest

from resin.dsl.spec import (
    SignatureSpec,
    TensorSpec,
    annotation_to_spec,
    bind_call_args,
    materialize_spec,
    parse_signature,
    typecheck_against_spec,
    validate_kwargs,
)
from resin.core.etype import F4
from resin.dsl import TensorMeta, View, param
from .fixtures import LinearParams, mlp_step


class TestAnnotationToSpec:
    def test_view_subscript(self) -> None:
        hint = View[F4, (3, 4)]
        spec = annotation_to_spec(hint)
        assert spec == TensorSpec("f4", (3, 4))

    def test_tensor_meta_directly(self) -> None:
        from typing import Annotated

        hint = Annotated[View, TensorMeta("f4", (2,))]
        spec = annotation_to_spec(hint)
        assert spec == TensorSpec("f4", (2,))

    def test_rank_zero_subscript(self) -> None:
        hint = View[F4, ()]
        spec = annotation_to_spec(hint)
        assert spec == TensorSpec("f4", ())

    def test_typed_dict(self) -> None:
        spec = annotation_to_spec(LinearParams)
        assert spec == {
            "weight": TensorSpec("f4", (10, 784)),
            "bias": TensorSpec("f4", (10,)),
        }

    def test_tuple(self) -> None:
        hint = tuple[View[F4, (2,)], View[F4, (3,)]]
        spec = annotation_to_spec(hint)
        assert spec == (
            TensorSpec("f4", (2,)),
            TensorSpec("f4", (3,)),
        )

    def test_list(self) -> None:
        hint = list[View[F4, (4,)]]
        spec = annotation_to_spec(hint)
        assert spec == [TensorSpec("f4", (4,))]

    def test_pod_int_rejected(self) -> None:
        with pytest.raises(ValueError, match="unsupported annotation"):
            annotation_to_spec(int)

    def test_bare_dict_rejected(self) -> None:
        with pytest.raises(ValueError, match="TypedDict"):
            annotation_to_spec(dict)

    def test_view_scalar_union_rejected(self) -> None:
        from resin.core.etype import Scalar

        with pytest.raises(ValueError, match="View \\| Scalar"):
            annotation_to_spec(View | Scalar)


def _view(shape: tuple[int, ...], *, etype: str = "f4") -> View:
    return param(shape=shape, etype=etype, label="v")


class TestParseSignature:
    def test_mlp_step(self) -> None:
        spec = parse_signature(mlp_step)
        assert set(spec.args.keys()) == {"x", "y", "params"}
        assert spec.return_spec == TensorSpec("f4", ())

    def test_unannotated_param_raises(self) -> None:
        def bad(x) -> View[F4, ()]:  # type: ignore[valid-type]
            return x

        with pytest.raises(ValueError, match="unannotated parameter"):
            parse_signature(bad)

    def test_unannotated_return_raises(self) -> None:
        def bad(x: View[F4, ()]):  # noqa: ANN202
            return x

        with pytest.raises(ValueError, match="unannotated return"):
            parse_signature(bad)

    def test_varargs_rejected(self) -> None:
        def bad(*xs: View[F4, ()]) -> View[F4, ()]:  # type: ignore[valid-type]
            return xs[0]

        with pytest.raises(ValueError, match="\\*args"):
            parse_signature(bad)


class TestTypecheckAgainstSpec:
    def test_tensor_view(self) -> None:
        typecheck_against_spec(_view((2, 3)), TensorSpec("f4", (2, 3)))

    def test_tensor_accepts_scalar_for_rank_zero(self) -> None:
        typecheck_against_spec(42, TensorSpec("f4", ()))
        typecheck_against_spec(1.5, TensorSpec("f8", ()))

    def test_tensor_rejects_scalar_for_non_rank_zero(self) -> None:
        with pytest.raises(TypeError, match="expected View with shape"):
            typecheck_against_spec(42, TensorSpec("f4", (2,)))

    def test_tensor_rejects_non_view(self) -> None:
        with pytest.raises(TypeError, match="expected View"):
            typecheck_against_spec("x", TensorSpec("f4", (2,)))

    def test_tensor_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="expected shape"):
            typecheck_against_spec(_view((2,)), TensorSpec("f4", (3,)))

    def test_tensor_rejects_wrong_etype(self) -> None:
        with pytest.raises(ValueError, match="expected etype"):
            typecheck_against_spec(
                param(shape=(2,), etype="f8", label="v"),
                TensorSpec("f4", (2,)),
            )

    def test_typed_dict(self) -> None:
        spec = {
            "weight": TensorSpec("f4", (2, 3)),
            "bias": TensorSpec("f4", (2,)),
        }
        typecheck_against_spec(
            {"weight": _view((2, 3)), "bias": _view((2,))},
            spec,
        )

    def test_typed_dict_rejects_missing_key(self) -> None:
        spec = {"weight": TensorSpec("f4", (2, 3))}
        with pytest.raises(ValueError, match="expected keys"):
            typecheck_against_spec({"bias": _view((2,))}, spec)


class TestValidateKwargs:
    def test_accepts_matching_kwargs(self) -> None:
        spec = SignatureSpec(
            args={"x": TensorSpec("f4", (2,))},
            return_spec=TensorSpec("f4", ()),
        )
        validate_kwargs({"x": _view((2,))}, spec)

    def test_rejects_extra_kwargs(self) -> None:
        spec = SignatureSpec(
            args={"x": TensorSpec("f4", (2,))},
            return_spec=TensorSpec("f4", ()),
        )
        with pytest.raises(ValueError, match="expected kwargs"):
            validate_kwargs({"x": _view((2,)), "y": _view((2,))}, spec)


class TestBindCallArgs:
    def test_positional_and_keyword(self) -> None:
        spec = SignatureSpec(
            args={
                "x": TensorSpec("f4", (2,)),
                "y": TensorSpec("f4", (2,)),
            },
            return_spec=TensorSpec("f4", (2,)),
        )
        x = _view((2,))
        y = _view((2,))
        assert bind_call_args(spec, (x,), {"y": y}) == {"x": x, "y": y}

    def test_rejects_duplicate_argument(self) -> None:
        spec = SignatureSpec(
            args={"x": TensorSpec("f4", (2,))},
            return_spec=TensorSpec("f4", (2,)),
        )
        with pytest.raises(TypeError, match="multiple values"):
            bind_call_args(spec, (_view((2,)),), {"x": _view((2,))})


class TestMaterializeSpec:
    def test_tensor_spec(self) -> None:
        view = materialize_spec(
            TensorSpec("f4", (2, 3)),
            lambda tensor_spec, label: param(
                shape=tensor_spec.shape,
                etype=tensor_spec.etype,
                label=label,
            ),
            path="x",
        )
        assert view.shape == (2, 3)
        assert view.etype == "f4"

    def test_list_spec_unsupported(self) -> None:
        with pytest.raises(TypeError, match="unsupported spec"):
            materialize_spec(
                [TensorSpec("f4", (2,))],
                lambda tensor_spec, label: param(
                    shape=tensor_spec.shape,
                    etype=tensor_spec.etype,
                    label=label,
                ),
            )