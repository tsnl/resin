import sys
import resin.console as rc
import resin.front as rf

# %%

rc.line("Storage Computation")
t1 = rf.const(value=[[1, 2], [3, 4], [5, 6]], stype="fp32")
t2 = rf.const(value=[[1, 2], [3, 4], [5, 6]], stype="fp32")
t3 = rf.const(value=[[1, 2], [3, 4], [5, 6]], stype="fp32")
ta = t1 + t2
tb = ta * t3
tb.debug_print(out=sys.stderr)
rc.line()
