"""Early defaults are opt-in overlays, never replacements for explicit values."""

import copy
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "ucm/integration/vllm/early_config.py"
SPEC = importlib.util.spec_from_file_location("early_config", SOURCE)
config_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(config_module)
normalize = config_module.normalize_early_config
MISSING = object()
STORE_DEFAULTS = {
    "cache_full_block_load": True,
    "cache_full_block_l1": True,
    "cache_full_block_dump": True,
    "cache_dump_capacity_policy": "wait",
}


def configuration(early=True, preset=MISSING, **store):
    config = {
        "ucm_connectors": [
            {
                "ucm_connector_name": "UcmPipelineStore",
                "ucm_connector_config": dict(store_pipeline="Cache|Posix", **store),
            }
        ]
    }
    if early is not MISSING:
        config["enable_early_prefetch"] = early
    if preset is not MISSING:
        config["early_full_block_defaults"] = preset
    return config


def store_options(config):
    return config["ucm_connectors"][0]["ucm_connector_config"]


@pytest.mark.parametrize("early", [MISSING, False, True])
@pytest.mark.parametrize("preset", [MISSING, False, True])
def test_switch_matrix_fills_only_enabled_defaults(early, preset):
    original = configuration(early, preset)
    before = copy.deepcopy(original)
    result = normalize(original)
    if early is True:
        assert result["early_prefetch_min_submit_reads"] == 16
        assert result["early_owner_max_jobs"] == 16
    else:
        assert "early_prefetch_min_submit_reads" not in result
        assert "early_owner_max_jobs" not in result
    for name, value in STORE_DEFAULTS.items():
        if early is True and preset is True:
            assert store_options(result)[name] == value
        else:
            assert name not in store_options(result)
        assert name not in result
    assert "early_prefetch_min_submit_reads" not in store_options(result)
    assert "early_owner_max_jobs" not in store_options(result)
    assert original == before


@pytest.mark.parametrize("preset", [MISSING, False, True])
@pytest.mark.parametrize(
    "name,value", [("early_prefetch_min_submit_reads", 1), ("early_owner_max_jobs", 8)]
)
def test_explicit_scheduling_values_do_not_depend_on_preset(preset, name, value):
    config = configuration(preset=preset)
    config[name] = value
    assert normalize(config)[name] == value


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "cache_full_block_load": False,
            "cache_full_block_l1": False,
            "cache_full_block_dump": False,
            "cache_dump_capacity_policy": "spill",
        },
        {
            "cache_full_block_l1": False,
            "cache_full_block_dump": False,
            "cache_dump_capacity_policy": "spill",
        },
        {"cache_full_block_dump": False, "cache_dump_capacity_policy": "spill"},
        {"cache_dump_capacity_policy": "spill"},
    ],
)
def test_explicit_storage_overrides_are_preserved(overrides):
    config = configuration(preset=True, **overrides)
    before = copy.deepcopy(config)
    result = normalize(config)
    assert store_options(result) == {
        "store_pipeline": "Cache|Posix",
        **STORE_DEFAULTS,
        **overrides,
    }
    assert config == before


@pytest.mark.parametrize("early", [MISSING, False, True])
@pytest.mark.parametrize("preset", [MISSING, False])
def test_inactive_preset_keeps_explicit_storage_handling(early, preset):
    # Even invalid combinations must reach the existing store validation unchanged.
    config = configuration(
        early, preset, cache_full_block_dump=False, cache_dump_capacity_policy="wait"
    )
    assert store_options(normalize(config)) == store_options(config)


@pytest.mark.parametrize("value", [None, 0, 1, "true", "false", [], {}])
def test_active_preset_switch_requires_a_boolean(value):
    with pytest.raises(ValueError, match="early_full_block_defaults must be a boolean"):
        normalize(configuration(preset=value))


@pytest.mark.parametrize("early", [MISSING, False])
def test_disabled_early_does_not_consume_preset(early):
    config = configuration(early, "unused")
    assert normalize(config) == config


def test_copies_are_idempotent_and_do_not_alias_nested_inputs():
    original = configuration(
        preset=True, posix_packed_layer_bytes=[4096, 8192], cache_dump_reserved_blocks=0
    )
    original.update(early_owner_max_candidate_blocks=0, early_owner_max_retries=0)
    result = normalize(original)
    assert result == normalize(result)
    assert result["early_owner_max_candidate_blocks"] == 0
    assert result["early_owner_max_retries"] == 0
    assert store_options(result)["cache_dump_reserved_blocks"] == 0
    store_options(result)["posix_packed_layer_bytes"][0] = 12288
    assert store_options(original)["posix_packed_layer_bytes"] == [4096, 8192]
    assert "cache_full_block_load" not in store_options(original)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"cache_full_block_dump": False}, "cache_dump_capacity_policy='spill'"),
        ({"cache_full_block_l1": False}, "cache_full_block_dump=false"),
        ({"cache_full_block_load": False}, "cache_full_block_l1=false"),
        ({"cache_load_backend_only": True}, "cache_load_backend_only=false"),
    ],
)
def test_conflicting_explicit_values_fail_without_repair(overrides, message):
    config = configuration(preset=True, **overrides)
    before = copy.deepcopy(config)
    with pytest.raises(ValueError, match=message):
        normalize(config)
    assert config == before


@pytest.mark.parametrize(
    "name", ["early_prefetch_min_submit_reads", "early_owner_max_jobs"]
)
def test_normalization_never_repairs_explicit_zero(name):
    config = configuration()
    config[name] = 0
    assert normalize(config)[name] == 0
