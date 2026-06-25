import subprocess
import sys
import textwrap
from pathlib import Path
from tempfile import NamedTemporaryFile

_REPO_ROOT = Path(__file__).resolve().parents[2].absolute()


def _run_pyright(snippet: str) -> subprocess.CompletedProcess[str]:
    with NamedTemporaryFile("w+", suffix=".py", dir=_REPO_ROOT) as tmp:
        _ = tmp.write(textwrap.dedent(snippet))
        tmp.flush()
        return subprocess.run(
            [sys.executable, "-m", "basedpyright", tmp.name, "--level", "error"],
            capture_output=True,
            text=True,
            check=False,
        )


def test_module_repr_subtypes_pytree() -> None:
    result = _run_pyright(
        """
        from resin.core.pytree import PyTree
        from resin.nn import Linear, linear_new

        layer: Linear = linear_new(2, 3, bias=True)
        model: list[Linear] = [layer]
        tree: PyTree[object] = model
        _ = tree
        """
    )
    assert result.returncode == 0


def test_typeddict_module_repr_does_not_subtype_pytree() -> None:
    result = _run_pyright(
        """
        from typing import NotRequired, TypedDict
        from resin.core.pytree import PyTree
        from resin.dsl import View, param
        from resin.core.etype import F4

        class Linear(TypedDict):
            weight: View
            bias: NotRequired[View]

        layer: Linear = {"weight": param(shape=(3, 2), etype=F4)}
        model: list[Linear] = [layer]
        tree: PyTree[View] = model
        _ = tree
        """
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "reportAssignmentType" in output


def test_pytree_accepts_literal_keyed_dict() -> None:
    # The dict arm is pytree.Mapping (items()-only, covariant in key and value), so a
    # narrowed-key repr subtypes PyTree[View] without a cast — unlike collections.abc.Mapping,
    # whose key-typed __getitem__ would force key invariance and reject it.
    result = _run_pyright(
        """
        from typing import Literal
        from resin.core.pytree import PyTree
        from resin.dsl import View, param
        from resin.core.etype import F4

        type Mlp = dict[Literal["weights", "bias"], View]

        layer: Mlp = {
            "weights": param(shape=(3, 2), etype=F4),
            "bias": param(shape=(3,), etype=F4),
        }
        tree: PyTree[View] = layer
        model: PyTree[View] = [layer]
        _ = (tree, model)
        """
    )
    assert result.returncode == 0


def test_pytree_rejects_str() -> None:
    result = _run_pyright(
        """
        from resin.core.pytree import PyTree

        tree: PyTree[int] = "nope"
        _ = tree
        """
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "reportAssignmentType" in output


def test_pytree_accepts_homogeneous_tuple_container() -> None:
    result = _run_pyright(
        """
        from resin.core.pytree import PyTree
        from resin.dsl import View, param
        from resin.core.etype import F4

        leaf = param(shape=(2,), etype=F4)
        pair: tuple[View, View] = (leaf, leaf)
        tree: PyTree[View] = pair
        _ = tree
        """
    )
    assert result.returncode == 0


def test_map_pytree_keeps_leaf_inference() -> None:
    result = _run_pyright(
        """
        from resin.core.pytree import PyTree, map_pytree

        tree: PyTree[int] = {"a": [1, 2], "b": 3}

        def double(x: int) -> int:
            return x * 2

        _ = map_pytree(tree, double)
        """
    )
    assert result.returncode == 0


def test_zip_pytree_leaf_is_pytree_zipped_namedtuple() -> None:
    result = _run_pyright(
        """
        from resin.core.pytree import PyTree, PyTreeZipped, map_pytree, zip_pytree

        params: PyTree[int] = {"w": 10, "b": 2}
        grads: PyTree[int] = {"w": 3, "b": 1}
        zipped: PyTree[PyTreeZipped[int, int]] = zip_pytree(params, grads)

        def update(pair: PyTreeZipped[int, int]) -> int:
            return pair.a - pair.b

        _ = map_pytree(zipped, update)
        """
    )
    assert result.returncode == 0
