"""Keep worker control membership independent of native KV partitioning."""


def worker_count(connector):
    """Return the number of consumers that must finish before owner release."""
    cp_size = getattr(
        connector, "cp_worker_world_size", getattr(connector, "cp_world_size", 1)
    )
    return connector.tp_size * cp_size


def worker_rank(connector):
    """Flatten a supported PCP/TP cohort into distinct control-channel peers."""
    tp_rank = connector.tp_rank % connector.tp_size
    if worker_count(connector) == connector.tp_size:
        return tp_rank
    cp_rank = getattr(connector, "cp_worker_rank", connector.current_rank)
    return cp_rank * connector.tp_size + tp_rank
