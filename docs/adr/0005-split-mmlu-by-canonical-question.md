# Split MMLU artifacts by canonical question

All MMLU direction-construction, development, and held-out evaluation partitions are grouped by the underlying canonical question rather than by generated row, biography, prompt wrapper, or answer order. Existing row-level sycophancy splits leak questions across train and test and their biography-following labels are not truth labels, so those splits and labels cannot qualify a truth direction or a held-out false-report result.
