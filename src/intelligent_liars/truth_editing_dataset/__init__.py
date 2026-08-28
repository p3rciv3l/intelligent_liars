"""Leakage-safe, source-aware truth-editing dataset compilation.

This package deliberately keeps the optimizer-facing shape small.  Readers may
accept the many historical formats used by Truth Spec, Apollo, and local
fixtures, but compilation produces one strict row shape and four explicit
splits.  The compiler never fetches a remote source: callers pass pinned
``SourceSpec`` values and in-memory/local readers.

The public entry points are :meth:`TruthEditingDataset.compile`,
:meth:`TruthEditingDataset.open`, :meth:`TruthEditingDataset.iter_split`, and
:meth:`TruthEditingDataset.audit`.
"""

from .compiler import TruthEditingDataset
from .contracts import (
    DATASET_FORMAT,
    DatasetAudit,
    DatasetCompileError,
    DatasetRequest,
    DatasetSource,
    ProvenanceRecord,
    TruthEditingRecord,
)

# Names used in design notes and by downstream callers.  Keep the canonical
# implementation names above while offering the shorter vocabulary everywhere
# else in the repository.
SourceSpec = DatasetSource
DatasetRecord = TruthEditingRecord
CompileRequest = DatasetRequest

__all__ = [
    "DATASET_FORMAT",
    "DatasetAudit",
    "DatasetCompileError",
    "DatasetRequest",
    "CompileRequest",
    "DatasetRecord",
    "DatasetSource",
    "ProvenanceRecord",
    "TruthEditingDataset",
    "TruthEditingRecord",
    "SourceSpec",
]
