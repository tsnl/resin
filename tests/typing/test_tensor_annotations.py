import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_pyright(path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "basedpyright",
            "--pythonversion",
            "3.14",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
        cwd=path.parent,
    )


def test_pyright_rejects_non_view_for_tensor_annotation() -> None:
    target = Path(__file__).with_name("bad_tensor_call.py")
    result = _run_pyright(target)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "reportArgumentType" in output


def test_pyright_accepts_view_for_tensor_annotation() -> None:
    target = Path(__file__).with_name("good_tensor_call.py")
    result = _run_pyright(target)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_pyright_accepts_tensor_methods_in_function_body() -> None:
    target = Path(__file__).with_name("good_tensor_methods.py")
    result = _run_pyright(target)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_pyright_accepts_scalar_for_rank_zero_tensor() -> None:
    target = Path(__file__).with_name("good_tensor_scalar.py")
    result = _run_pyright(target)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output


def test_pyright_accepts_fixtures_mlp_step() -> None:
    target = _REPO_ROOT / "tests" / "resin" / "dsl" / "fixtures.py"
    result = _run_pyright(target)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output