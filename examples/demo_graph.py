"""Demo of graph.py debug_print output for various node types."""

import sys

from resin.graph import ConstNode, ParamNode


def section(title: str) -> None:
    print(f"\n=== {title} ===", file=sys.stderr)


# Constants
section("Scalar constant")
ConstNode.new(42, stype="fp32").debug_print(out=sys.stderr)

section("1D constant")
ConstNode.new([1, 2, 3], stype="fp32").debug_print(out=sys.stderr)

section("2D constant")
ConstNode.new([[1, 2], [3, 4]], stype="fp32").debug_print(out=sys.stderr)

section("fp16 constant")
ConstNode.new([1, 2], stype="fp16").debug_print(out=sys.stderr)

# Elementwise binary ops
section("Add two tensors")
t1 = ConstNode.new([1, 2], stype="fp32")
t2 = ConstNode.new([3, 4], stype="fp32")
(t1 + t2).debug_print(out=sys.stderr)

section("Chained ops: (t1 + t2) * t3")
t3 = ConstNode.new([5, 6], stype="fp32")
((t1 + t2) * t3).debug_print(out=sys.stderr)

# Elementwise unary ops
section("Chained exp/log")
ConstNode.new([1, 2], stype="fp32").exp().log().debug_print(out=sys.stderr)

section("Deep chain: exp.log.exp.log")
ConstNode.new([1, 2, 3, 4], stype="fp32").exp().log().exp().log().debug_print(
    out=sys.stderr
)

# Scalar broadcast in binary op
section("Add scalar")
t = ConstNode.new([[1, 2], [3, 4]], stype="fp32")
(t + 10).debug_print(out=sys.stderr)

# Reduction
section("Sum reduction axis=1")
ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32").reduce(
    axis=1, operator="add"
).debug_print(out=sys.stderr)

# Indexing (must use tuple keys)
section("Integer index")
ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")[(0, slice(None))].debug_print(
    out=sys.stderr
)

section("Slice with step")
ConstNode.new([1, 2, 3, 4, 5, 6], stype="fp32")[(slice(None, None, 2),)].debug_print(
    out=sys.stderr
)

section("Multi-dim index")
ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")[(1, slice(1, 3))].debug_print(
    out=sys.stderr
)

# Permute
section("Permute (transpose)")
ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32").permute((1, 0)).debug_print(
    out=sys.stderr
)

# Compact
section("Compact after slice")
t = ConstNode.new([[1, 2, 3], [4, 5, 6]], stype="fp32")
t[(slice(None, None, 2), slice(None))].compact().debug_print(out=sys.stderr)

# Max/min
section("Max with scalar")
ConstNode.new([1, 2, 3], stype="fp32").max(0).debug_print(out=sys.stderr)

# Comparisons
section("Greater than (method)")
t1 = ConstNode.new([1, 2], stype="fp32")
t2 = ConstNode.new([2, 1], stype="fp32")
t1.gt(t2).debug_print(out=sys.stderr)

# Shared subexpressions
section("Shared subexpressions")
t1 = ConstNode.new([1, 2], stype="fp32")
t2 = ConstNode.new([3, 4], stype="fp32")
s = t1 + t2
(s * s).debug_print(out=sys.stderr)

# Param node
section("Param node")
ParamNode.new(shape=(4,), stype="fp32", label="weights").debug_print(out=sys.stderr)
