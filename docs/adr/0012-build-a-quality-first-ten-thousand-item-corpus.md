# Build a quality-first ten-thousand-item corpus

The unified truth-editing corpus targets approximately ten thousand canonical QA items across train, validation, and test after deduplication and quarantine. Exact duplicates and renderings share one canonical cluster, strong likely paraphrases remain in one partition, ambiguous matches and untrusted labels are removed, and the optimizer evaluates tiered stratified subsets rather than generating every view of all ten thousand items on every trial.
