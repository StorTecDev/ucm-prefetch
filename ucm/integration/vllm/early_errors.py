"""Explicit failures eligible for a fresh, drained Early execution attempt."""


class EarlyLookupUnavailable(RuntimeError):
    """The deferred metadata query was rejected by a native health breaker."""

    code = "lookup_unavailable"
