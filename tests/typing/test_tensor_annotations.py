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


def test_pyright_rejects_non_view_for_tensor_annotation() -> None:
    result = _run_pyright(
        """
        from resin.core.etype import F4
        from resin.dsl import View

        def accepts_tensor(x: View[F4, (2,)]) -> None:
            _ = x

        accepts_tensor(42)
        """
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "reportArgumentType" in output


def test_pyright_accepts_view_for_tensor_annotation() -> None:
    result = _run_pyright(
        """
        from resin.core.etype import F4
        from resin.dsl import View, param

        def accepts_tensor(x: View[F4, (2,)]) -> None:
            _ = x

        accepts_tensor(param(shape=(2,), etype=F4, name="x"))
        """
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_pyright_accepts_tensor_methods_in_function_body() -> None:
    result = _run_pyright(
        """
        from resin.core.etype import F4
        from resin.dsl import View, const, param

        def reduce_scalar(x: View[F4, (2,)]) -> View[F4, ()]:
            return x.sum().squeeze(axes=(0,))

        _ = reduce_scalar(param(shape=(2,), etype=F4, name="x"))
        _ = reduce_scalar(const(1.0, etype=F4).broadcast((2,)))
        """
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_pyright_accepts_scalar_for_rank_zero_tensor() -> None:
    result = _run_pyright(
        """
        from resin.core.etype import Scalar, F4
        from resin.dsl import View, const


        def accepts_scalar_tensor(x: View | Scalar) -> None:
            _ = x


        accepts_scalar_tensor(1.0)
        accepts_scalar_tensor(42)
        accepts_scalar_tensor(const(1.0, etype=F4))
        """
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
