# Use batched Optuna suggestions

The main truth-editing study proposes and evaluates recipes in synchronous batches rather than allowing completion timing to change later suggestions. This preserves a reproducible proposal history and clean resume behavior while still evaluating each batch in parallel across independent GPU workers.
