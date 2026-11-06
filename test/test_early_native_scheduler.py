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

from ucm.integration.vllm.early_prefetch import Job


@pytest.fixture
def scheduler_factory(tmp_path):
    OPTConfig(
        architectures=["OPTForCausalLM"],
        hidden_size=64,
        ffn_dim=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=131072,
        vocab_size=1000,
    ).to_json_file(tmp_path / "config.json")

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


def test_ready_does_not_jump_earlier_native_fallback(connector, scheduler_factory):
    _, output, adapter = schedule(connector, scheduler_factory, ("FAILED", "READY"))
    assert (
        adapter.get_num_new_matched_tokens.call_args_list[0].args[0].request_id == "0"
    )
    assert output.num_scheduled_tokens == {"0": 4096}


def test_native_priority_is_preserved_for_ready_requests(connector, scheduler_factory):
    scheduler = scheduler_factory(max_num_batched_tokens=512)
    scheduler.policy = SchedulingPolicy.PRIORITY
    scheduler.waiting = create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = create_request_queue(scheduler.policy)
    requests = requests_of_sizes(8192, 8192)
    requests[0].priority, requests[1].priority = -10, 10
    adapter = attach(connector, scheduler, requests, ("READY", "READY"), (48, 60))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"0": 512}
    assert adapter.get_num_new_matched_tokens.call_count == 1


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_native_running_slot_limit_is_preserved(
    connector, scheduler_factory, async_scheduling
):
    scheduler = scheduler_factory(max_num_seqs=1, async_scheduling=async_scheduling)
    requests = requests_of_sizes(8192, 8192)
    adapter = attach(connector, scheduler, requests, ("FAILED", "READY"), (0, 48))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"0": 4096}
    assert len(scheduler.running) == 1
    assert adapter.get_num_new_matched_tokens.call_count == 1


def test_native_hbm_refusal_does_not_backfill_ready_requests(
    connector, scheduler_factory
):
    scheduler = scheduler_factory(num_blocks=16)
    requests = requests_of_sizes(8192, 1024)
    adapter = attach(connector, scheduler, requests, ("READY", "READY"), (60, 4))
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {}
    assert adapter.get_num_new_matched_tokens.call_count == 1
