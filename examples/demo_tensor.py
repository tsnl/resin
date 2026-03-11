import sys
import resin.tensor as rt
import resin.console as rc

# %%

rc.line("Basic Tensor Operations")
t1 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], stype="fp32")
t2 = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], stype="fp32")
tm = (t1 @ t2 + 42)[::2, ::-1]
tm.debug_print(out=sys.stderr)

# %%

rc.line("Common Subexpressions")
t1 = rt.Tensor.const(value=[1, 2], stype="fp32")
t2 = rt.Tensor.const(value=[3, 4], stype="fp32")
sum_t = t1 + t2
result = sum_t * sum_t
result.debug_print(out=sys.stderr)

rc.line()
