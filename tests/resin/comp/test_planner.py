import sys
import resin.comp as rc


def test_compute_storage_basic():
    t1 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    t2 = rc.Tensor.const(value=[[1, 2, 3], [4, 5, 6]], dtype="fp32")
    tm = t1 @ t2

    tm.debug_print(sys.stderr)

    _, storage_map = rc.compute_storage(tm)
    expects = {
        t1: rc.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
        t2: rc.Storage(shape=(2, 3), dtype="fp32", mode="ro"),
        tm: rc.Storage(shape=(3, 3), dtype="fp32", mode="rw"),
    }
    for t, s in expects.items():
        assert storage_map[t] == s


def test_compute_storage_reuse():
    t1 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    t2 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    t3 = rc.Tensor.const(value=[[1, 2], [3, 4], [5, 6]], dtype="fp32")
    ta = t1 + t2
    tb = ta * t3

    tb.debug_print(sys.stderr)

    _, storage_map = rc.compute_storage(tb)
    expects = {
        t1: rc.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
        t2: rc.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
        t3: rc.Storage(shape=(3, 2), dtype="fp32", mode="ro"),
        ta: rc.Storage(shape=(3, 2), dtype="fp32", mode="rw"),
        tb: rc.Storage(shape=(3, 2), dtype="fp32", mode="rw"),
    }
    for t, s in expects.items():
        assert storage_map[t] == s

    assert storage_map[ta] is storage_map[tb]  # Ensure storage reuse
