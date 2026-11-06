"""Early-only pipeline assembled from native Store subclasses."""

import numpy as np

from ucm.integration.vllm.early_errors import EarlyLookupUnavailable
from ucm.integration.vllm.early_lookup import current_lookup

from .connector import UcmPipelineStore, UcmPipelineStoreBuilder, ucmpipelinestore
from .errors import StoreUnhealthyError

_EARLY_PIPELINE = "Early(Cache|Posix)"


class _ReservedLoad:
    """Observe native I/O without duplicating conversion, queues or resources."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.reservations = ()
        self.observer = None
        self.receipts = {}
        self.io_failure_handler = None

    def __getattr__(self, name):
        return getattr(self.pipeline, name)

    def Load(self, ids, indexes, addrs):
        reservations = self.reservations
        receipt = self.observer.begin(ids, indexes) if self.observer else None
        try:
            task = ucmpipelinestore.EarlyLoad(self.pipeline, ids, indexes, addrs, reservations)
            if receipt is not None:
                self.receipts[task] = receipt
                self.observer.submitted(receipt, task)
            return task
        except ucmpipelinestore.EarlyCapacityRejected:
            if receipt is not None:
                self.observer.rejected(receipt)
            raise
        except Exception:
            self._io_failed(receipt, None)
            raise

    def Check(self, task):
        try:
            return self.pipeline.Check(task)
        except Exception:
            # Native polling drops failed tasks without a later Wait callback.
            self._io_failed(self.receipts.pop(task, None), task)
            raise

    def Wait(self, task):
        receipt = self.receipts.pop(task, None)
        try:
            result = self.pipeline.Wait(task)
        except Exception:
            self._io_failed(receipt, task)
            raise
        if receipt is not None:
            self.observer.complete(receipt, task, True)
        return result

    def _io_failed(self, receipt, task):
        if receipt is not None:
            self.observer.complete(receipt, task, False)
        elif self.observer:
            self.observer.unscoped_failure()
        if self.io_failure_handler is not None:
            self.io_failure_handler("native I/O failed; drain state is not proven")

    def Dump(self, ids, indexes, addrs, *args):
        if self.observer:
            self.observer.check_task("dump", len(indexes))
        try:
            return ucmpipelinestore.EarlyDump(self.pipeline, ids, indexes, addrs, *args)
        except ucmpipelinestore.EarlyCapacityRejected:
            raise  # native rank consistency records a failed, never a saved dump
        except Exception:
            if self.observer:
                self.observer.unscoped_failure()
            raise


class _EarlyStack:
    """Adapt factory names while the native builder configures the same pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def Stack(self, name, path, config):
        if name not in ("Cache", "Posix"):
            raise ValueError(f"unsupported Early store: {name}")
        self.pipeline.Stack(f"Early{name}", path, config)


def _build_early_pipeline(config, pipeline):
    builder = UcmPipelineStoreBuilder.get("Cache|Posix")
    if builder is None:
        raise ValueError("unknown store pipeline: Cache|Posix")
    builder(dict(config, store_pipeline="Cache|Posix"), _EarlyStack(pipeline))


UcmPipelineStoreBuilder.register(_EARLY_PIPELINE, _build_early_pipeline)


class EarlyPipelineStore(UcmPipelineStore):

    def wait_dump_sources(self, task):
        ucmpipelinestore.EarlyWaitDumpSources(self._native_store, task.task_id)

    def finalize_dump_data(self, block_ids, shard_index, src_addr, prerequisite_handle=0):
        """Retry compute-ready layers while the connector still owns their HBM.

        Args:
            block_ids: Content identities, one per layer address row.
            shard_index: Corresponding layer indices.
            src_addr: HBM tensor addresses valid through wait_dump_sources() or wait().
            prerequisite_handle: Event recorded after the supplied layers completed.

        Returns:
            A native task whose completion closes these block write transactions.
        """
        from .connector import UcmPipelineStoreTransTask

        ids = np.frombuffer(b"".join(block_ids), dtype=np.uint8)
        indexes = np.asarray(shard_index, dtype=np.uint64)
        addrs = np.ascontiguousarray(src_addr, dtype=np.uint64)
        task = self.store_.Dump(ids, indexes, addrs, prerequisite_handle, True)
        return UcmPipelineStoreTransTask(task)

    def __init__(self, config: dict[str, object]) -> None:
        if (
            config.get("cache_full_block_dump", False)
            and getattr(ucmpipelinestore, "EARLY_BLOCK_DUMP_PROTOCOL_VERSION", None)
            != 2
        ):
            raise RuntimeError("Block writeback bindings are missing; rebuild UCM for this source")
        pipeline = config.get("store_pipeline")
        if pipeline != "Cache|Posix":
            raise ValueError(
                f"Early supports only store_pipeline='Cache|Posix', got {pipeline!r}"
            )

        required = (
            "EarlyReservePrefix", "EarlyPrefetchReserved", "EarlyReleaseReservation",
            "EarlyLoad", "EarlyDump", "EarlyCapacityRejected",
            "EarlyBatch",
            "EarlyDomainInfo", "EarlyWaitRelease", "EarlyReleaseEpoch",
        )
        if not all(callable(getattr(ucmpipelinestore, name, None)) for name in required):
            raise RuntimeError(
                "Early pipeline bindings are missing; rebuild UCM for this source"
            )
        super().__init__(dict(config, store_pipeline=_EARLY_PIPELINE))
        self._native_store = self.store_
        self.store_ = _ReservedLoad(self._native_store)
        # The internal builder key changes selection only, not the user's config.
        self.config = config

    def lookup_on_prefix(self, block_ids: list[bytes]) -> int:
        context = current_lookup.get()
        if context is not None and context.connector.store is self:
            if context.hbm_only:
                # Recompute the native APC suffix. Still let native token and
                # dump-metadata generation run, without querying external KV.
                return -1
            job = context.job
            if job is not None and job.state == "READY":
                # A waiting-request retry may start at the previous hit minus
                # one block, beyond the HBM boundary. Count relative to the
                # actual query slice, or native resume accounting double-counts
                # the earlier hits and claims unprotected suffix blocks.
                if not block_ids or job.blocks is None:
                    return -1
                try:
                    start = job.blocks.index(block_ids[0], context.hbm_hit_hashes)
                except ValueError:
                    return -1
                # Rank-consistency filtering may also shorten the query.
                count = min(
                    max(job.hit - start, 0), len(block_ids), len(job.blocks) - start
                )
                for index in range(count):
                    if job.blocks[start + index] != block_ids[index]:
                        return index - 1
                return count - 1
        return super().lookup_on_prefix(block_ids)

    def reserve_prefix(self, block_ids: list[bytes], *, deferred=False) -> dict:
        try:
            return ucmpipelinestore.EarlyReservePrefix(
                self._native_store, np.frombuffer(b"".join(block_ids), dtype=np.uint8), deferred
            )
        except StoreUnhealthyError as error:
            if deferred:
                raise EarlyLookupUnavailable(str(error)) from error
            raise

    def release_epoch(self):
        return ucmpipelinestore.EarlyReleaseEpoch(self._native_store)

    def wait_release(self, epoch, milliseconds=250):
        return ucmpipelinestore.EarlyWaitRelease(self._native_store, epoch, milliseconds)

    def domain_info(self):
        return dict(ucmpipelinestore.EarlyDomainInfo(self._native_store))

    def set_consumer(self, observer):
        self.store_.observer = observer

    def set_io_failure_handler(self, handler):
        """Retain the worker when asynchronous I/O cannot prove safe completion."""
        self.store_.io_failure_handler = handler

    def prefetch_reserved(self, handle: int) -> dict:
        return ucmpipelinestore.EarlyPrefetchReserved(self._native_store, handle)

    def release_reservation(self, handle: int) -> None:
        self.store_.reservations = tuple(
            active for active in self.store_.reservations if active != handle
        )
        ucmpipelinestore.EarlyReleaseReservation(self._native_store, handle)

    def set_load_reservations(self, handles) -> None:
        self.store_.reservations = tuple(handles)
