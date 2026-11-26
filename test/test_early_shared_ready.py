"""Cross-process READY reserve publication without accelerator dependencies."""

import importlib
import os
from pathlib import Path
import subprocess
import struct
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest


package = ModuleType("shared_ready_source")
package.__path__ = [str(Path(__file__).resolve().parents[1] / "ucm/integration/vllm")]
sys.modules[package.__name__] = package
ready = importlib.import_module(package.__name__ + ".early_ready")
shared = importlib.import_module(package.__name__ + ".early_shared_ready")


def test_shared_budget_counts_inflight_notifications_and_first_schedule():
    publisher = ready.ReadyReserve(12)
    reader = ready.ReadyReserve(12)
    reader.attach_shared(publisher.share())
    try:
        for sequence in range(1, 8):
            publisher.attempts.add((str(sequence), "attempt", 1))
            publisher.publish(sequence)
        assert reader.promotion_budget(9) == 3
        publisher.attempts.add(("8", "attempt", 1))
        publisher.publish(8)
        assert reader.promotion_budget(9) == 3
        publisher.discard(SimpleNamespace(key="1", attempt="attempt", grant=1))
        assert reader.promotion_budget(9) == 4
        publisher.publish(9)
        assert reader.promotion_budget(9) == 5
        with pytest.raises(RuntimeError, match="Noncontiguous"):
            publisher.publish(11)
        assert reader.promotion_budget(9) == 5
    finally:
        reader.close_shared()
        publisher.close_shared()


def test_late_cancellation_after_close_uses_local_accounting():
    reserve = ready.ReadyReserve(1)
    descriptor = reserve.share()
    reserve.attempts.add(("key", "attempt", 1))
    reserve.publish(1)
    reserve.close_shared()
    reserve.close_shared()
    with pytest.raises(OSError):
        os.fstat(descriptor)
    reserve.discard(SimpleNamespace(key="key", attempt="attempt", grant=1))
    assert reserve.snapshot == (0, 1)
    assert reserve.promotion_budget(1) == 1


def test_shared_pair_is_coherent_between_real_processes():
    snapshot = shared.SharedReadySnapshot()
    script = textwrap.dedent("""
        import importlib.util, os, sys
        spec = importlib.util.spec_from_file_location("snapshot", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        snapshot = module.SharedReadySnapshot(int(sys.argv[2]))
        for index in range(20000):
            snapshot.write((index, index * 3))
        os.write(int(sys.argv[3]), b"x")
        snapshot.close()
    """)
    reader, writer = os.pipe()
    os.set_blocking(reader, False)
    process = None
    try:
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", script, shared.__file__,
             str(snapshot.fileno()), str(writer)],
            pass_fds=(snapshot.fileno(), writer),
        )
        os.close(writer)
        writer = None
        for _ in range(200000):
            count, watermark = snapshot.read()
            assert watermark == count * 3
            try:
                if os.read(reader, 1):
                    break
            except BlockingIOError:
                pass
            if process.poll() is not None:
                break
        assert process.wait(timeout=10) == 0
        assert snapshot.read() == (19999, 59997)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        os.close(reader)
        if writer is not None:
            os.close(writer)
        snapshot.close()


def test_snapshot_unlocks_after_publication_error():
    snapshot = shared.SharedReadySnapshot(initial=(2, 3))
    try:
        with pytest.raises(struct.error):
            snapshot.write((-1, 4))
        snapshot.write((5, 8))
        assert snapshot.read() == (5, 8)
    finally:
        snapshot.close()


def test_failed_attachment_does_not_leak_descriptor():
    descriptor = os.memfd_create("invalid-ready-record", os.MFD_CLOEXEC)
    try:
        before = len(os.listdir("/proc/self/fd"))
        with pytest.raises(ValueError):
            shared.SharedReadySnapshot(descriptor)
        assert len(os.listdir("/proc/self/fd")) == before
    finally:
        os.close(descriptor)
