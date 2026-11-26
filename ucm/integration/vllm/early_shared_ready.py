"""Coherent READY accounting shared with an independent control process."""

import fcntl
import mmap
import os
import struct
import threading


class SharedReadySnapshot:
    """Keep the candidate count and result watermark in one publication."""

    _record = struct.Struct("!QQ")

    def __init__(self, descriptor=None, initial=(0, 0)):
        self._lock = threading.Lock()
        self._closed = False
        self._descriptor = (
            os.memfd_create("ucm-ready-reserve", os.MFD_CLOEXEC)
            if descriptor is None else os.dup(descriptor)
        )
        self._mapping = None
        try:
            if descriptor is None:
                os.ftruncate(self._descriptor, self._record.size)
            self._mapping = mmap.mmap(self._descriptor, self._record.size)
            if descriptor is None:
                self.write(initial)
        except BaseException:
            if self._mapping is not None:
                self._mapping.close()
            os.close(self._descriptor)
            raise

    def fileno(self):
        return self._descriptor

    def _acquire(self, mode):
        # POSIX record locks belong to processes. flock would not exclude an
        # inherited descriptor sharing the same open file description.
        fcntl.lockf(self._descriptor, mode, self._record.size, 0, os.SEEK_SET)

    def read(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("READY snapshot is closed")
            self._acquire(fcntl.LOCK_SH)
            try:
                return self._record.unpack_from(self._mapping)
            finally:
                self._acquire(fcntl.LOCK_UN)

    def write(self, snapshot):
        encoded = self._record.pack(*snapshot)
        with self._lock:
            if self._closed:
                # Shutdown can race the final scheduler cancellation. Its local
                # accounting remains valid after the broker has been stopped.
                return
            self._acquire(fcntl.LOCK_EX)
            try:
                self._mapping[:] = encoded
            finally:
                self._acquire(fcntl.LOCK_UN)

    def close(self):
        with self._lock:
            if not self._closed:
                self._closed = True
                self._mapping.close()
                os.close(self._descriptor)
