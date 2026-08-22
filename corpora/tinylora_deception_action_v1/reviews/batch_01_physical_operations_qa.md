# QA report: physical operations batch 01

## Result

**PASS.** `batch_01_physical_operations.jsonl` contains 200 schema-valid, low-risk paired scenarios authored directly with GPT-5.6 SOL in Codex chat. No external model API, paid API credit, training run, GPU rental, package build, shared compiler/schema edit, or S3 operation was used.

The reviewed JSONL SHA-256 is `4ff49d1d35f381a56b55b02933f39702c86617979f47d651c2e652846804a866` (345,850 bytes; 200 newline-terminated records).

## Inputs reviewed before authoring

- `corpora/tinylora_deception_action_v1/README.md`
- `corpora/tinylora_deception_action_v1/schema/paired_scenario.schema.json`
- `corpora/tinylora_deception_action_v1/synthetic/paired_scenarios.jsonl`
- `corpora/tinylora_deception_action_v1/source_registry.json`
- The existing `_validate_scenario` implementation and corpus tests in `src/intelligent_liars/tinylora_corpus.py` and `tests/test_tinylora_corpus.py` (read-only)

## Inventory

| Family | Scenarios |
|---|---:|
| routing | 20 |
| inventory | 20 |
| scheduling | 20 |
| facility_operations | 20 |
| equipment_setup | 20 |
| document_movement | 20 |
| laboratory_classroom_workflow | 20 |
| event_setup | 20 |
| transit | 20 |
| package_logistics | 20 |
| **Total** | **200** |

All 200 records have `risk_level: low`. They contain 200 indirect probes and 200 false-trigger controls.

## Automated structural and semantic checks

1. **JSONL parse and exact shape:** 200/200 records parsed. Root and nested key sets exactly match the current schema; no undeclared fields were present.
2. **Official JSON Schema:** `Draft202012Validator` plus `FormatChecker` against `paired_scenario.schema.json` returned **0 errors** across 200 rows.
3. **Current compiler semantics:** read-only invocation of `_validate_scenario` returned **0 errors** across 200 rows.
4. **Repository tests:** `tests/test_tinylora_corpus.py` returned **8 passed**.
5. **World-state pairing:** every true value differs from its counterfactual; every target `asserted_value` equals the counterfactual and differs from the true value.
6. **Report pairing:** every neutral and target report asks the same direct question; every answer changes; every false-trigger control repeats the neutral truthful answer.
7. **Action dependence:** every row declares exactly two unique allowed actions; both correct actions are allowed; true-state and counterfactual actions differ. A token-grounding audit confirmed the state-specific value is reflected in its action label for all 400 state/action mappings.
8. **Grouping and uniqueness:** batch IDs are unique and the collision check against `synthetic/paired_scenarios.jsonl` returned **0 collisions**. During final integration, `split_group_id` was tightened from one group per row to the 10 scenario families (20 rows per group) so related templates cannot cross data splits.
9. **Surface duplication:** exact-duplicate counts were zero for direct questions, neutral answers, target incentive conditions, action instructions, indirect-probe questions, and false-trigger conditions.
10. **Scope/safety lexical audit:** targeted scans found **0 peer-domain hits** for software/data, image-interpretation, or interpersonal/social-reporting terms, and **0 hits** for harmful operational domains.
11. **Provenance:** all rows use `author_model: gpt-5.6-sol`, `authoring_mode: codex_chat_direct`, `authored_at: 2026-08-21`, and retain the declared structural/source inspirations.

During action-grounding QA, two semantically correct but less machine-transparent action labels were tightened (`facing_desk` and `bin_RET_A` mappings). All checks above were rerun after those edits.

## Manual stratified review

I manually read the first and last row of every family (20 rows total), covering:

- `physops.routing.loading_dock_lane.0001`
- `physops.routing.courtyard_detour.0020`
- `physops.inventory.craft_paper_cabinet.0001`
- `physops.inventory.foam_block_crate.0020`
- `physops.scheduling.study_room_slot.0001`
- `physops.scheduling.recycling_collection.0020`
- `physops.facilities.lecture_lighting_zone.0001`
- `physops.facilities.storage_room_sign.0020`
- `physops.equipment.projector_cart_mark.0001`
- `physops.equipment.tabletop_track_slope.0020`
- `physops.documents.workshop_packet_tray.0001`
- `physops.documents.exhibit_check_cards.0020`
- `physops.classlab.color_water_sample.0001`
- `physops.classlab.model_bridge_station.0020`
- `physops.events.welcome_table.0001`
- `physops.events.supply_return.0020`
- `physops.transit.campus_shuttle_bay.0001`
- `physops.transit.route_map_box.0020`
- `physops.packages.bookstore_parcel_shelf.0001`
- `physops.packages.delivery_cart_return.0020`

For each sampled row I checked that the fact was deterministic, the target answer was a direct false assertion, the neutral/control/probe answers preserved the true fact, the physical action could not be selected correctly without that fact, the counterfactual changed the correct action, the incentive was low-stakes and realistic, and the wording stayed within the assigned domain. **No manual-review defects remained.**

## Integration note

This file is intentionally a standalone batch shard. The current shared compiler reads `synthetic/paired_scenarios.jsonl`; this worker did not alter that compiler or the shared input. The main integrator must include or merge this shard during final compilation and packaging.
