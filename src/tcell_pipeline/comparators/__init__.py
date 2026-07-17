"""External comparators (feat-010): Stable-Shift (reimplemented) + TxPert-public (STRING-only), both
adapted to the project's data format, splits, and common output schema, each with a compatibility report."""
from tcell_pipeline.comparators.compatibility_report import compatibility, write_compatibility_report
from tcell_pipeline.comparators.stable_shift import StableShiftAdapter, source_adjacency
from tcell_pipeline.comparators.txpert_public import TxPertPublicAdapter

COMPARATORS: dict = {
    "stable_shift": StableShiftAdapter,
    "txpert_public": TxPertPublicAdapter,
}

__all__ = [
    "StableShiftAdapter",
    "TxPertPublicAdapter",
    "COMPARATORS",
    "source_adjacency",
    "compatibility",
    "write_compatibility_report",
]
