"""Control identities span all PCP workers without changing native KV hashes."""


def worker_count(connector):
    """Return the number of consumers that must finish before owner release."""
    return connector.tp_size * getattr(connector, "cp_world_size", 1)


def worker_rank(connector):
    """Flatten a supported PCP/TP cohort into distinct control-channel peers."""
    tp_rank = connector.tp_rank % connector.tp_size
    if getattr(connector, "cp_world_size", 1) == 1:
        return tp_rank
    return connector.current_rank * connector.tp_size + tp_rank
