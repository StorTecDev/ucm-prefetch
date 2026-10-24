"""Validate compact disk layouts for the Early pipeline."""

from typing import Any


def validate_packed_config(config: dict[str, Any]) -> list[int] | None:
    """Reject ambiguous packed layouts before creating native resources.

    Args:
        config: Store options shared by scheduler and worker.

    Returns:
        Explicit physical lengths, or None for the uniform disk layout.

    Raises:
        ValueError: The layout or pipeline cannot safely use packed files.
    """
    if "posix_packed_layer_bytes" not in config:
        return None
    lengths = config["posix_packed_layer_bytes"]
    if (
        not isinstance(lengths, list)
        or not lengths
        or len(lengths) > 1024
        or any(type(n) is not int or n <= 0 or n % 4096 for n in lengths)
        or sum(lengths) > 2**32 - 1
    ):
        raise ValueError("posix_packed_layer_bytes requires positive 4 KiB lengths")
    if (
        config.get("store_pipeline") != "Cache|Posix"
        or config.get("store_io_alignment", 4096) != 4096
        or config.get("cache_full_block_load") is not True
    ):
        raise ValueError(
            "Packed L2 requires Cache|Posix full-block loading at 4 KiB alignment"
        )
    return lengths


def validate_packed_layers(
    config: dict[str, Any], layout: Any, *, layerwise: bool, blocks_per_chunk: int
) -> int | None:
    """Bind explicit file lengths to the worker's real device tensor layout.

    Args:
        config: Store options including the physical L1 shard size.
        layout: Materialized KV layout with per-layer pointers and tensor sizes.
        layerwise: Whether each store shard represents exactly one model layer.
        blocks_per_chunk: Number of KV blocks grouped into a store chunk.

    Returns:
        Physical L2 bytes per block, or None when packing is disabled.

    Raises:
        ValueError: A configured length omits real data or the layout is unsupported.
    """
    lengths = validate_packed_config(config)
    if lengths is None:
        return None
    if not layerwise or blocks_per_chunk != 1:
        raise ValueError(
            "Packed L2 currently requires one layerwise KV block per chunk"
        )
    pointers = layout.base_ptrs.tolist()
    sizes = layout.tensor_size_lists.tolist()
    uniform = config["tensor_size_list"]
    if len(pointers) != len(lengths) or len(sizes) != len(lengths):
        raise ValueError("Packed layer count disagrees with the actual model")
    for index, (ptrs, row, physical) in enumerate(zip(pointers, sizes, lengths)):
        if row != uniform or len(ptrs) != len(uniform):
            raise ValueError(
                "Packed L2 requires the existing uniform L1 tensor offsets"
            )
        offset = required = 0
        for ptr, size in zip(ptrs, row):
            offset += size
            # Split real segments can have zero registration size. The pointer,
            # rather than registration metadata, determines whether bytes exist.
            if ptr:
                required = offset
        aligned = (required + 4095) // 4096 * 4096
        if not required or physical != aligned or physical > config["shard_size"]:
            raise ValueError(
                f"Packed layer {index} does not match its actual tensor bytes"
            )
    return sum(lengths)
