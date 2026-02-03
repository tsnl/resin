import sys
import resin.tensor as rt
import resin.interp as ri


def test_compute_storage_basic():
    t1 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    t2 = rt.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
    tm = t1 @ t2

    tm.debug_print(sys.stderr)

    storage_map = ri.compute_storage(tm)
    expects = {
        t1: ri.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
        t2: ri.Storage(shape=(2, 3), dtype="fp32", mode="ro"),
        tm: ri.Storage(shape=(3, 3), dtype="fp32", mode="rw"),
    }
    for t, s in expects.items():
        assert storage_map[t] == s


def test_compute_storage_reuse():
    t1 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    t2 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    t3 = rt.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    ta = t1 + t2
    tb = ta * t3

    tb.debug_print(sys.stderr)

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
