"""Demo of dsl debug_print output for various node types."""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "resin",
# ]
#
# [tool.uv.sources]
# resin = { path = "..", editable = true }
# ///

import sys

from resin import dsl
from resin.core.etype import F2, F4


def section(title: str) -> None:
    print(f"\n=== {title} ===", file=sys.stderr)


section("Scalar constant")
dsl.const(42, etype=F4).debug_print(out=sys.stderr)

section("1D constant")
dsl.const([1, 2, 3], etype=F4).debug_print(out=sys.stderr)

section("2D constant")
dsl.const([[1, 2], [3, 4]], etype=F4).debug_print(out=sys.stderr)

section("fp16 constant")
dsl.const([1, 2], etype=F2).debug_print(out=sys.stderr)

section("Add two tensors")
t1 = dsl.const([1, 2], etype=F4)
t2 = dsl.const([3, 4], etype=F4)
(t1 + t2).debug_print(out=sys.stderr)

section("Chained ops: (t1 + t2) * t3")
t3 = dsl.const([5, 6], etype=F4)
((t1 + t2) * t3).debug_print(out=sys.stderr)

section("Chained exp/log")
dsl.const([1, 2], etype=F4).exp().log().debug_print(out=sys.stderr)

section("Deep chain: exp.log.exp.log")
dsl.const([1, 2, 3, 4], etype=F4).exp().log().exp().log().debug_print(out=sys.stderr)

section("Add scalar")
t = dsl.const([[1, 2], [3, 4]], etype=F4)
(t + 10).debug_print(out=sys.stderr)

section("Sum reduction axis=1")
dsl.const([[1, 2, 3], [4, 5, 6]], etype=F4).reduce(
    axes=(1,), operator="add"
).debug_print(out=sys.stderr)

section("Integer index")
dsl.const([[1, 2, 3], [4, 5, 6]], etype=F4)[(0, slice(None))].debug_print(
    out=sys.stderr
)

section("Slice with step")
dsl.const([1, 2, 3, 4, 5, 6], etype=F4)[(slice(None, None, 2),)].debug_print(
    out=sys.stderr
)

section("Multi-dim index")
dsl.const([[1, 2, 3], [4, 5, 6]], etype=F4)[(1, slice(1, 3))].debug_print(
    out=sys.stderr
)

section("Permute (transpose)")
dsl.const([[1, 2, 3], [4, 5, 6]], etype=F4).permute((1, 0)).debug_print(out=sys.stderr)

section("Compact after slice")
t = dsl.const([[1, 2, 3], [4, 5, 6]], etype=F4)
t[(slice(None, None, 2), slice(None))].copy().debug_print(out=sys.stderr)

section("Max with scalar")
dsl.const([1, 2, 3], etype=F4).max(0).debug_print(out=sys.stderr)

section("Greater than (method)")
t1 = dsl.const([1, 2], etype=F4)
t2 = dsl.const([2, 1], etype=F4)
t1.gt(t2).debug_print(out=sys.stderr)

section("Shared subexpressions")
t1 = dsl.const([1, 2], etype=F4)
t2 = dsl.const([3, 4], etype=F4)
s = t1 + t2
(s * s).debug_print(out=sys.stderr)

section("Param node")
dsl.param(shape=(4,), etype=F4, label="weights").debug_print(out=sys.stderr)
