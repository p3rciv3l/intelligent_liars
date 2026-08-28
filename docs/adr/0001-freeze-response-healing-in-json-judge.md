# Freeze response healing in the structured JSON judge

The truth-editing judge returns strict JSON, so production judging explicitly enables response healing before strict local schema validation. Response healing is frozen into the resolved judge-request identity and receives its own calibration and cache namespace; it must never be injected invisibly or mixed with unhealed results.
