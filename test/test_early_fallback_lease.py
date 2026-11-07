"""Fallback lookup snapshots with the real simulator and request lifetimes."""
from types import SimpleNamespace
import array
import ctypes
import errno
import fcntl
import os
import gc
import uuid

import numpy as np
import pytest

from ucm.integration.vllm.early_lookup import LookupContext, current_lookup
from ucm.store.pipeline.early_connector import EarlyPipelineStore
from ucm.store.pipeline import early_connector as early_store
from ucm.store.pipeline import ucmpipelinestore as native


def test_failed_prefetch_fallback_retains_every_lookup_until_normal_completion(monkeypatch):
    store = EarlyPipelineStore.__new__(EarlyPipelineStore)
    store._native_store = object()
    store._fallback_lookup_leases = {}
    kept = []
    class Lease:
        def __del__(self):
            kept.append("released")
    def lookup(pipeline, ids):
        assert pipeline is store._native_store
        return 0, Lease()
    monkeypatch.setattr(native, "EarlyLookupWithLease", lookup)
    request = SimpleNamespace(request_id="request")
    context = LookupContext(SimpleNamespace(store=store), request,
                            SimpleNamespace(state="FAILED"), 0)
    token = current_lookup.set(context)
    try:
        assert store.lookup_on_prefix([b"a" * 16]) == 0
        assert store.lookup_on_prefix([b"b" * 16]) == 0
    finally:
        current_lookup.reset(token)
    assert len(store._fallback_lookup_leases["request"]) == 2
    store.release_lookup_leases("request", cancelled=True)
    gc.collect()
    assert not kept
    store.release_lookup_leases("request")
    gc.collect()
    assert kept == ["released", "released"]
    store.release_lookup_leases("request")


def test_missing_lookup_keeps_no_lease(monkeypatch):
    store = EarlyPipelineStore.__new__(EarlyPipelineStore)
    store._native_store = object()
    store._fallback_lookup_leases = {}
    monkeypatch.setattr(native, "EarlyLookupWithLease", lambda *args: (-1, None))
    token = current_lookup.set(LookupContext(SimpleNamespace(store=store),
        SimpleNamespace(request_id="request"), None, 0))
    try:
        assert store.lookup_on_prefix([b"x" * 16]) == -1
    finally:
        current_lookup.reset(token)
    assert not store._fallback_lookup_leases


def test_native_fallback_lookup_returns_a_lease_and_observes_missing_data(tmp_path):
    from pathlib import Path
    library = Path(native.__file__).resolve().parent.parent
    config = dict(storage_backends=[str(tmp_path)], device_id=0,
                  tensor_size=4096, shard_size=4096, block_size=4096,
                  data_dir_shard_bytes=1, posix_data_trans_concurrency=1,
                  posix_lookup_concurrency=1, posix_io_engine="psync",
                  io_direct=False, store_health={"enabled": False})
    block = uuid.uuid4().bytes
    ids = np.frombuffer(block, dtype=np.uint8)
    source = np.arange(4096, dtype=np.uint8)
    seed = native.PipelineStore()
    seed.Stack("EarlyPosix", str(library / "posix/libposixstore.so"), config)
    seed.Wait(seed.Dump(ids, array.array("Q", [0]),
                        np.array([[source.ctypes.data]], dtype=np.uint64)))
    pipeline = native.PipelineStore()
    scheduler = dict(config, device_id=-1)
    pipeline.Stack("EarlyPosix", str(library / "posix/libposixstore.so"), scheduler)
    pipeline.Stack("EarlyCache", str(library / "cache/libcachestore.so"),
                   dict(scheduler, unique_id=uuid.uuid4().hex,
                        cache_buffer_capacity_gb=1, share_buffer_enable=False))
    hit, lease = native.EarlyLookupWithLease(pipeline, ids)
    assert hit == 0 and lease is not None
    class Flock(ctypes.Structure):
        _fields_ = [("kind", ctypes.c_short), ("whence", ctypes.c_short),
                    ("start", ctypes.c_longlong), ("length", ctypes.c_longlong),
                    ("pid", ctypes.c_int)]
    lock = bytes(Flock(fcntl.F_WRLCK, os.SEEK_SET, 0, 0, 0))
    with (tmp_path / ".ucm_read_leases").open("r+b") as gc_lock:
        for retain_pipeline in (True, False):
            if not retain_pipeline:
                del pipeline
            with pytest.raises(OSError) as caught:
                fcntl.fcntl(gc_lock.fileno(), fcntl.F_OFD_SETLK, lock)
            assert caught.value.errno in (errno.EAGAIN, errno.EACCES)
        del lease
        gc.collect()
        fcntl.fcntl(gc_lock.fileno(), fcntl.F_OFD_SETLK, lock)
        # A GC winner must become a miss, not an unprotected successful lookup.
        assert native.EarlyLookupWithLease(seed, ids) == (-1, None)
    missing = np.frombuffer(uuid.uuid4().bytes, dtype=np.uint8)
    hit, lease = native.EarlyLookupWithLease(seed, missing)
    assert hit == -1 and lease is None
