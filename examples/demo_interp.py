import sys
import resin.console as rc
import resin.graph as rg

# %%

rc.line("Storage Computation")
t1 = rg.ConstNode.new(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
t2 = rg.ConstNode.new(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
t3 = rg.ConstNode.new(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
ta = t1 + t2
tb = ta * t3
tb.debug_print(out=sys.stderr)
rc.line()
