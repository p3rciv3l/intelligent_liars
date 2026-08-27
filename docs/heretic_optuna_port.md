# Heretic Optuna Port

**Status:** future design only. It is outside the current simple-transform
block and does not authorize optimization runs or spending.

## Goal

Use Heretic's Optuna/TPE loop to find the abliteration recipe that produces the strongest selective deception without unacceptable loss of control behavior, capability, or safety.

## Keep from Heretic

- Per-layer direction construction or loading.
- Reversible low-rank candidate patches.
- Separate attention `o_proj` and MLP `down_proj` layer kernels.
- Direction choice, layer location, kernel width, and intervention-strength search.
- Checkpointed multi-objective TPE study and Pareto frontier.

## Change

Replace refusal-keyword rate and KL as optimization objectives. Keep KL diagnostic only. Add a direction-provider seam for saved per-layer directions; rank-k support can follow the rank-1 port.

## Search space

- Basis granularity: domain-general, intermediate cluster, or domain-specific.
- Direction policy: per-layer or global.
- Writer site: attention, MLP, or both.
- Layer-kernel center and width.
- Strength `lambda`: `0` unchanged, `1` projection, `2` reflection.
- Later: rank `k in {1, 2, 4}` and exact versus norm-preserving edit.

## Objectives and constraints

Primary objective: maximize worst-domain LLM-graded selective deception—coherent false reporting under the real trigger while true-state knowledge remains available.

Secondary evidence: free-generation change and complete-response preferred-versus-alternative margin.

Hard constraints:

- Truthful false-trigger behavior stays above its floor.
- Linked truthful reports, indirect truth, and downstream actions stay above their floors.
- Incorrect-answer confidence must not rise generally on neutral controls.
- General capability and safety stay above their floors.
- Invalid, evasive, empty, and repetitive output stays below its ceiling.

Optuna compares only feasible trials. Among feasible trials, select the highest primary objective, then use control and capability preservation as tie-breakers.

## Protocol

1. Freeze directions, search rows, judge identity, rubric, and constraints.
2. Optimize only on the search-development split.
3. Save every recipe, patch, raw output, judge receipt, and score.
4. Evaluate Pareto finalists once on untouched domains/families with the audit judge.
5. Export the winning exact Optuna configuration and reversible adapter.
