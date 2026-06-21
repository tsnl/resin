import importlib
from typing import cast

import resin

prelude = importlib.import_module("resin.dsl.prelude")


class TestPrelude:
    def test_import_after_resin(self) -> None:
        import resin.dsl.prelude as prelude_module

        assert prelude_module.View is resin.dsl.View

    def test_all_exports(self) -> None:
        exports = cast(list[str], prelude.__all__)
        assert set(exports) == {
            "F2",
            "F4",
            "Scalar",
            "TensorOperand",
            "U4",
            "View",
            "const",
            "grad_fn",
            "param",
            "trace",
            "typecheck",
            "zeros",
        }
