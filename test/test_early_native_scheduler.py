"""Real vLLM Scheduler/APC/budget with Early matching; storage and H2D mocked.

Run with the locked vLLM source's test helpers on PYTHONPATH. A tiny local
config is created by the fixture; no model weights or network are needed.
"""

from unittest.mock import Mock

import pytest
from test_early_ready_reuse import connector
from tests.v1.core.utils import create_requests, create_scheduler
from transformers import OPTConfig
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus

from ucm.integration.vllm.early_prefetch import Job
from ucm.integration.vllm.early_scheduling import ReadyPriority, patch_scheduler_class


@pytest.fixture
def scheduler_factory(tmp_path, monkeypatch):
    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=64,
        ffn_dim=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=131072,
        vocab_size=1000,
    ).to_json_file(tmp_path / "config.json")
    monkeypatch.setattr(Scheduler, "schedule", Scheduler.schedule)
    patch_scheduler_class(Scheduler)

    def create(**overrides):
        options = dict(
            model=str(tmp_path),
            max_num_seqs=8,
            max_num_batched_tokens=4096,
            block_size=128,
            max_model_len=131072,
            enable_prefix_caching=True,
            skip_tokenizer_init=True,
        )
        options.update(overrides)
        return create_scheduler(**options)

    return create


def schedule(connector, scheduler_factory, states, *, hit=48, same_prompt=False):
    scheduler = scheduler_factory()
    requests = create_requests(
        num_requests=2, num_tokens=8192, block_size=128, same_prompt=same_prompt
    )
    adapter = attach(connector, scheduler, requests, states, [hit] * len(requests))
    output = scheduler.schedule()
    return scheduler, output, adapter


def attach(connector, scheduler, requests, states, hits):
    connector.block_size = connector.hash_block_size = 128
    connector._early.scheduling = ReadyPriority({})
    for request, state, hit in zip(requests, states, hits):
        scheduler.add_request(request)
        job = Job(request, request.all_token_ids, connector._early.identity(request))
        job.state, job.hit = state, hit
        job.blocks = [i.to_bytes(16, "big") for i in range(request.num_tokens // 128)]
        connector._early.jobs[request.request_id] = job
        connector._early.by_key[job.key] = job
    adapter = Mock()
    adapter.connector = connector
    adapter.get_num_new_matched_tokens.side_effect = (
        connector.get_num_new_matched_tokens
    )
    adapter.build_connector_meta.return_value = None
    adapter.request_finished.return_value = (False, None)
    adapter.request_finished_all_groups.return_value = (False, None)
    scheduler.connector = adapter
    return adapter


def requests_of_sizes(*sizes):
    return [
        create_requests(num_requests=index + 1, num_tokens=size, block_size=128)[index]
        for index, size in enumerate(sizes)
    ]


def test_ready_external_hits_only_charge_uncached_tokens(connector, scheduler_factory):
    _, output, _ = schedule(connector, scheduler_factory, ("READY", "READY"))
    # 8192 prompt - 6144 protected external hit = 2048 actual compute each.
    assert output.num_scheduled_tokens == {"0": 2048, "1": 2048}
    assert output.total_num_scheduled_tokens == 4096
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_same_wave_hbm_hit_does_not_wait_for_pending_redundant_prefetch(
    connector, scheduler_factory
):
    scheduler, output, adapter = schedule(
        connector, scheduler_factory, ("READY", "PENDING"), same_prompt=True
    )
    # Real allocate_slots caches the first request's blocks. Native APC leaves
    # the last block for logits; it can never return 8192 for this prompt.
    assert [
        call.args[1] for call in adapter.get_num_new_matched_tokens.call_args_list
    ] == [0, 8064]
    assert output.num_scheduled_tokens == {"0": 2048, "1": 128}
    assert output.total_num_scheduled_tokens == 2176
    assert len(scheduler.skipped_waiting) == 0
    connector._early.control.send.assert_called_once()
    assert connector._early.control.send.call_args.args[0]["op"] == "cancel"
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_insufficient_cache_does_not_fake_two_requests_into_budget(
    connector, scheduler_factory
):
    _, output, _ = schedule(connector, scheduler_factory, ("READY", "READY"), hit=32)
    assert output.num_scheduled_tokens == {"0": 4096}
    assert output.total_num_scheduled_tokens == 4096


def test_pending_without_hbm_coverage_still_skips(connector, scheduler_factory):
    scheduler, output, _ = schedule(connector, scheduler_factory, ("READY", "PENDING"))
    assert output.num_scheduled_tokens == {"0": 2048}
    assert len(scheduler.skipped_waiting) == 1
    connector._early.control.send.assert_not_called()


def test_ready_uses_budget_before_earlier_failed_fallback(connector, scheduler_factory):
    _, output, adapter = schedule(connector, scheduler_factory, ("FAILED", "READY"))
    assert (
        adapter.get_num_new_matched_tokens.call_args_list[0].args[0].request_id == "1"
    )
    assert output.num_scheduled_tokens["1"] == 2048
    assert output.total_num_scheduled_tokens <= 4096


def test_ready_can_pass_higher_native_priority_failed_request(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(max_num_batched_tokens=2048)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    requests = requests_of_sizes(8192, 8192)
    requests[0].priority = -10
    requests[1].priority = 10
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 48))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"1": 2048}
    assert adapter.get_num_new_matched_tokens.call_count == 1
    assert scheduler.waiting.peek_request() is requests[0]


def test_failed_fair_attempt_cannot_stop_smaller_ready_on_hbm_failure(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(num_blocks=16)
    requests = requests_of_sizes(8192, 1024)
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 4))
    policy = connector._early.scheduling
    policy.bypasses[requests[0]] = policy.max_bypasses
    output = scheduler.schedule()
    assert [
        call.args[0].request_id
        for call in adapter.get_num_new_matched_tokens.call_args_list
    ] == ["0", "1"]
    assert output.num_scheduled_tokens == {"1": 512}
    assert scheduler.waiting.peek_request() is requests[0]
    assert [
        call.args[0] for call in adapter.update_state_after_alloc.call_args_list
    ] == [requests[1]]


def test_large_ready_allocation_failure_does_not_stop_smaller_ready(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(num_blocks=16)
    requests = requests_of_sizes(8192, 1024)
    adapter = attach(connector, scheduler, requests, ("READY", "READY"), (60, 4))
    output = scheduler.schedule()
    assert [
        call.args[0].request_id
        for call in adapter.get_num_new_matched_tokens.call_args_list
    ] == ["0", "1"]
    assert output.num_scheduled_tokens == {"1": 512}
    assert scheduler.waiting.peek_request() is requests[0]
    connector.store.store_.LookupOnPrefix.assert_not_called()


def test_scan_limit_rotates_requests_when_every_attempt_waits(
    connector, scheduler_factory
):
    scheduler = scheduler_factory()
    requests = requests_of_sizes(8192, 8192, 8192, 8192)
    adapter = attach(connector, scheduler, requests, ["PENDING"] * 4, [0] * 4)
    connector._early.scheduling.max_candidates = 2
    for _ in range(2):
        output = scheduler.schedule()
        assert not output.num_scheduled_tokens
    assert [
        call.args[0].request_id
        for call in adapter.get_num_new_matched_tokens.call_args_list
    ] == ["0", "1", "2", "3"]
    assert len(scheduler.waiting) + len(scheduler.skipped_waiting) == 4


def test_failed_request_gets_fair_attempt_after_younger_admissions(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(max_num_batched_tokens=512)
    requests = requests_of_sizes(8192, 1024, 1024)
    attach(connector, scheduler, requests, ("FAILED", "READY", "READY"), (0, 4, 4))
    connector._early.scheduling.max_bypasses = 1
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"1": 512}
    scheduler.finish_requests(["1"], RequestStatus.FINISHED_STOPPED)
    second = scheduler.schedule()
    assert second.num_scheduled_tokens == {"0": 512}


def test_zero_hit_ready_does_not_displace_original_waiting_order(
    connector, scheduler_factory
):
    _, output, adapter = schedule(
        connector, scheduler_factory, ("FAILED", "READY"), hit=0
    )
    assert (
        adapter.get_num_new_matched_tokens.call_args_list[0].args[0].request_id == "0"
    )
    assert "0" in output.num_scheduled_tokens


def test_non_early_connector_keeps_native_scheduler_order(connector, scheduler_factory):
    scheduler = scheduler_factory()
    requests = requests_of_sizes(8192, 8192)
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 48))
    adapter.connector = None
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"0": 4096}


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_ready_priority_preserves_global_running_slot_limit(
    connector, scheduler_factory, async_scheduling
):
    scheduler = scheduler_factory(max_num_seqs=1, async_scheduling=async_scheduling)
    requests = requests_of_sizes(8192, 8192)
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 48))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"1": 2048}
    assert len(scheduler.running) == 1
    assert adapter.get_num_new_matched_tokens.call_count == 1


def test_ready_beyond_scan_budget_is_considered_before_fallback(
    connector, scheduler_factory
):
    scheduler = scheduler_factory()
    requests = requests_of_sizes(8192, 8192, 8192, 8192)
    adapter = attach(
        connector, scheduler, requests, ["FAILED"] * 3 + ["READY"], [0] * 3 + [48]
    )
    connector._early.scheduling.max_candidates = 1
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"3": 2048}
    assert adapter.get_num_new_matched_tokens.call_count == 1


def test_nonchunked_request_too_large_for_budget_yields_to_ready(
    connector, scheduler_factory
):
    scheduler = scheduler_factory()
    scheduler.scheduler_config.enable_chunked_prefill = False
    requests = requests_of_sizes(8192, 1024)
    attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 4))
    policy = connector._early.scheduling
    policy.bypasses[requests[0]] = policy.max_bypasses
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"1": 512}
    assert scheduler.waiting.peek_request() is requests[0]


@pytest.mark.parametrize("refusal", ["encoder", "hbm"])
def test_rejected_request_releases_encoder_references_before_backfill(
    connector, scheduler_factory, monkeypatch, refusal
):
    scheduler = scheduler_factory(num_blocks=16)
    requests = requests_of_sizes(8192, 1024)
    requests[0].mm_features = [object()]
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 4))
    policy = connector._early.scheduling
    policy.bypasses[requests[0]] = policy.max_bypasses
    touched = set()
    original_free = scheduler.encoder_cache_manager.free

    def encoder_inputs(request, computed, tokens, budget, **kwargs):
        touched.add(request.request_id)
        return [], 0 if refusal == "encoder" else tokens, budget, []

    def free(request):
        touched.discard(request.request_id)
        original_free(request)

    def match(request, computed):
        if request is requests[1]:
            assert not touched
        return connector.get_num_new_matched_tokens(request, computed)

    monkeypatch.setattr(scheduler, "_try_schedule_encoder_inputs", encoder_inputs)
    monkeypatch.setattr(scheduler.encoder_cache_manager, "free", free)
    adapter.get_num_new_matched_tokens.side_effect = match
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"1": 512}
    assert not touched
