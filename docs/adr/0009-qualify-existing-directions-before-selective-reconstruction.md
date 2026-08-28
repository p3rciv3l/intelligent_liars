# Qualify existing directions before selective reconstruction

The direction bank begins from the repository's existing general and domain-specific vectors rather than assuming they are absent or rebuilding all of them. Every vector is compiled into a candidate manifest; vectors with sufficient checkpoint, source, split, and leakage evidence become optimizer-eligible, while unresolved shortlisted candidates are reconstructed from clean grouped data using existing activations where possible and at most the separately authorized GPU budget.
