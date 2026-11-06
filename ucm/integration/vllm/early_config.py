"""Configuration defaults owned by the Early connector entry point."""

import copy
from typing import Any

_SCHEDULING_DEFAULTS = {
    "early_prefetch_min_submit_reads": 16,
    "early_owner_max_jobs": 16,
    "early_prefetch_request_first": False,
    "early_ready_priority": False,
}
_FULL_BLOCK_DEFAULTS = {
    "cache_full_block_load": True,
    "cache_full_block_l1": True,
    "cache_full_block_dump": True,
    "cache_dump_capacity_policy": "wait",
}


def normalize_early_config(config: dict[str, Any]) -> dict[str, Any]:
    """Copy launch options and fill only the defaults enabled for Early.

    Args:
        config: Launch options loaded from YAML or connector extra config.

    Returns:
        An independent configuration with explicit values preserved. Early
        options remain inactive when enable_early_prefetch is false or absent.

    Raises:
        ValueError: An active switch is not boolean, a removed scheduling mode
            is requested, or store overrides conflict with the full-block preset.
    """
    normalized = copy.deepcopy(config)
    if not normalized.get("enable_early_prefetch", False):
        return normalized

    for name in ("early_prefetch_request_first", "early_ready_priority"):
        value = normalized.get(name, _SCHEDULING_DEFAULTS[name])
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
        if value:
            raise ValueError(
                f"{name}=true is no longer supported; set {name}=false "
                "to use baseline HRRN loading and native compute admission"
            )

    full_block_defaults = normalized.get("early_full_block_defaults", False)
    if type(full_block_defaults) is not bool:
        raise ValueError("early_full_block_defaults must be a boolean")

    for name, value in _SCHEDULING_DEFAULTS.items():
        normalized.setdefault(name, value)
    if full_block_defaults:
        for connector in normalized.get("ucm_connectors", []):
            store = connector["ucm_connector_config"]
            for name, value in _FULL_BLOCK_DEFAULTS.items():
                store.setdefault(name, value)
            _validate_full_block_dependencies(store)
    return normalized


def _validate_full_block_dependencies(config: dict[str, Any]) -> None:
    """Explain conflicting preset overrides before creating native resources.

    Args:
        config: Store options after filling the full-block preset.

    Raises:
        ValueError: Explicit overrides leave an incompatible combination.
    """
    # Repairing these dependencies would discard explicit user choices. Native
    # storage still checks runtime capabilities and all other configuration.
    if (
        config["cache_dump_capacity_policy"] == "wait"
        and not config["cache_full_block_dump"]
    ):
        raise ValueError(
            "cache_dump_capacity_policy='wait' requires cache_full_block_dump=true; "
            "set cache_dump_capacity_policy='spill' together with cache_full_block_dump=false"
        )
    if config["cache_full_block_dump"] and not config["cache_full_block_l1"]:
        raise ValueError(
            "cache_full_block_dump=true requires cache_full_block_l1=true; "
            "to disable whole-block L1, also set cache_full_block_dump=false "
            "and cache_dump_capacity_policy='spill'"
        )
    if config["cache_full_block_l1"] and (
        not config["cache_full_block_load"]
        or config.get("cache_load_backend_only", False)
    ):
        raise ValueError(
            "cache_full_block_l1=true requires cache_full_block_load=true "
            "and cache_load_backend_only=false; to disable full-block loading, "
            "also set cache_full_block_l1=false, cache_full_block_dump=false "
            "and cache_dump_capacity_policy='spill'"
        )
