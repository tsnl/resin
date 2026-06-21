import importlib

prelude = importlib.import_module("resin.dsl.prelude")


class TestPrelude:
    def test_all_exports(self) -> None:
        assert set(prelude.__all__) == {
            "View",
            "const",
            "grad_fn",
            "param",
            "Tensor",
            "trace",
            "zeros",
        }