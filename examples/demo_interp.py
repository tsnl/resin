import sys
import rich
import resin.console as rc
import resin.tensor as rt
import resin.interp as ri

# %%

rc.line("Storage Computation")
t1 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
t2 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
t3 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
ta = t1 + t2
tb = ta * t3

storage_map = ri.compute_storage(tb)
expects = {
    t1: ri.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
    t2: ri.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
    t3: ri.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
    ta: ri.Storage(shape=(3, 2), dtype="fp32", mode="rw"),
    tb: ri.Storage(shape=(3, 2), dtype="fp32", mode="rw"),
}
for t, s in expects.items():
    assert storage_map[t] == s

assert storage_map[ta] is storage_map[tb]  # Ensure storage reuse

tb.debug_print(out=sys.stderr)

rc.line()

for tensor, storage in storage_map.items():
    rich.print(
        f"{tensor} -> "
        f"Storage(shape={storage.shape}, dtype={storage.dtype}, mode={storage.mode})"
        f" @ {id(storage):08x}",
    )

rc.line()
