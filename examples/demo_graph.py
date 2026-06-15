"""Demo of graph.py debug_print output for various node types."""

import sys

from resin.graph import const, param


def section(title: str) -> None:
    print(f"\n=== {title} ===", file=sys.stderr)


# Constants
section("Scalar constant")
const(42, stype="fp32").debug_print(out=sys.stderr)

section("1D constant")
const([1, 2, 3], stype="fp32").debug_print(out=sys.stderr)

section("2D constant")
const([[1, 2], [3, 4]], stype="fp32").debug_print(out=sys.stderr)

section("fp16 constant")
const([1, 2], stype="fp16").debug_print(out=sys.stderr)

# Elementwise binary ops
section("Add two tensors")
t1 = const([1, 2], stype="fp32")
t2 = const([3, 4], stype="fp32")
(t1 + t2).debug_print(out=sys.stderr)

section("Chained ops: (t1 + t2) * t3")
t3 = const([5, 6], stype="fp32")
((t1 + t2) * t3).debug_print(out=sys.stderr)

# Elementwise unary ops
section("Chained exp/log")
const([1, 2], stype="fp32").exp().log().debug_print(out=sys.stderr)

section("Deep chain: exp.log.exp.log")
const([1, 2, 3, 4], stype="fp32").exp().log().exp().log().debug_print(
    out=sys.stderr
)

# Scalar broadcast in binary op
section("Add scalar")
t = const([[1, 2], [3, 4]], stype="fp32")
(t + 10).debug_print(out=sys.stderr)

# Reduction
section("Sum reduction axis=1")
const([[1, 2, 3], [4, 5, 6]], stype="fp32").reduce(
    axes=(1,), operator="add"
).debug_print(out=sys.stderr)

# Indexing (must use tuple keys)
section("Integer index")
const([[1, 2, 3], [4, 5, 6]], stype="fp32")[(0, slice(None))].debug_print(
    out=sys.stderr
)

section("Slice with step")
const([1, 2, 3, 4, 5, 6], stype="fp32")[(slice(None, None, 2),)].debug_print(
    out=sys.stderr
)

section("Multi-dim index")
const([[1, 2, 3], [4, 5, 6]], stype="fp32")[(1, slice(1, 3))].debug_print(
    out=sys.stderr
)

# Permute
section("Permute (transpose)")
const([[1, 2, 3], [4, 5, 6]], stype="fp32").permute((1, 0)).debug_print(
    out=sys.stderr
)

# Compact
section("Compact after slice")
t = const([[1, 2, 3], [4, 5, 6]], stype="fp32")
t[(slice(None, None, 2), slice(None))].copy().debug_print(out=sys.stderr)

# Max/min
section("Max with scalar")
const([1, 2, 3], stype="fp32").max(0).debug_print(out=sys.stderr)

# Comparisons
section("Greater than (method)")
t1 = const([1, 2], stype="fp32")
t2 = const([2, 1], stype="fp32")
t1.gt(t2).debug_print(out=sys.stderr)

# Shared subexpressions
section("Shared subexpressions")
t1 = const([1, 2], stype="fp32")
t2 = const([3, 4], stype="fp32")
s = t1 + t2
(s * s).debug_print(out=sys.stderr)

# Param node
section("Param node")
param(shape=(4,), stype="fp32", label="weights").debug_print(out=sys.stderr)
