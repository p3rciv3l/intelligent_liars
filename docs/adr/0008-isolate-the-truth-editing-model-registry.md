# Isolate the truth-editing model registry

New failed experiments, successful experiments, final checkpoints, receipts, and indexes live in a private versioned truth-editing registry namespace. Legacy artifacts remain immutable read-only inputs referenced by exact identity; they may be cataloged as archived, but readiness work does not overwrite or physically migrate them onto the critical path.
