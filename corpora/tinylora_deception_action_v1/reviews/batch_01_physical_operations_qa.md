# QA report: physical operations batch 01

## Result

**PASS.** `batch_01_physical_operations.jsonl` contains 200 schema-valid, low-risk paired scenarios authored directly with GPT-5.6 SOL in Codex chat. No external model API, paid API credit, training run, GPU rental, package build, shared compiler/schema edit, or S3 operation was used.

The worker-reviewed JSONL SHA-256 was `5d50a4fa24af82321b89d3d286051382dc07e814babfb971d96989fb0213d82d`
(318,105 bytes; 200 newline-terminated records). Final integration changed only `split_group_id`
from one group per row to one group per scenario family. The integrated JSONL SHA-256 is
`e77dda63faf6c0d17853d625cff7bae28419c171c13720a92dace3b541711a4f` (313,047 bytes).

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
7. **Action dependence:** every row declares exactly two unique allowed actions; both correct actions are allowed; true-state and counterfactual actions differ. The reproducible grounding rule described below linked all **400/400** state/action mappings, and the instruction scan found **0** cases that named only one state's discriminating token. Allowed-action order was shuffled independently of IDs: the true action is first in 100 rows and the counterfactual action is first in 100 rows; agreement with numeric ID-suffix parity is 98/200, the longest same-position run is 8, and the true-first count ranges from 7 to 12 within each 20-row family.
8. **Grouping and uniqueness:** batch IDs are unique and the collision check against `synthetic/paired_scenarios.jsonl` returned **0 collisions**. Final integration uses the 10 scenario families as split groups (20 rows per group) so related templates cannot cross data splits.
9. **Surface duplication, label leakage, and structural shortcuts:** exact-duplicate counts were zero for direct questions, neutral answers, target incentive conditions, action instructions, indirect-probe questions, and false-trigger conditions. Target and control conditions are each a single realistic operational sentence, with all 200 unique on each side. A label-leakage scan found 0 uses in either condition type of `false`, `wrong`, `inaccurate`, `misleading`, `counterfactual`, `untrue`, `misstate`, `lie`, `truth*`, or `decept*`; the answer pairing, not a meta-label in the prompt, determines the behavior. After semantic rebalancing, only 7/200 target motives use the previously overrepresented count/score/metric/trial family, and 41/200 controls use privacy/access cues; the remaining controls are spread across deadlines/urgency, observation/audit, performance/capacity, and ordinary label/stamp cues. A normalized pairwise-similarity audit and manual inspection of its 20 highest-scoring pairs found no remaining pair that repeated both the same fact relation and the same operational consequence. The maximum normalized Jaccard score was 0.243. Reviewer-identified stock frames occur 0 times; action instructions contain 0 uses of `actual`/`actually`.
10. **Scope and safety:** scenario IDs, neutral contexts, questions, actions, and probes were reviewed for topical centrality. All examples center on physical operations/logistics; none center on software/data workflows, image interpretation, or interpersonal/social reporting. All are benign and low stakes.
11. **Provenance:** all rows use `author_model: gpt-5.6-sol`, `authoring_mode: codex_chat_direct`, `authored_at: 2026-08-21`, and retain the declared structural/source inspirations.

The action-grounding audit first tightened three semantically correct but less machine-transparent labels. Independent review then found a universal allowed-action position shortcut; its first correction accidentally tracked ID parity, so the final ordering was independently shuffled and re-audited. Review also found one leaky/unrealistic mailer scenario, three causally reversed routing incentives, repetitive target/control skeletons, meta-label leakage, motive/control category concentration, and several semantic near-clones. The routing incentives were repaired, the mailer was rebuilt around an opaque routing card, and all stock report/control conclusions were removed. Fifty-three overrepresented target motives and 88 privacy-heavy controls were replaced with ordinary scenario-specific conditions. Further rewrites differentiated room delivery, holder/tip, recycling, count/capacity, courtesy-cart, document orientation/grouping, event grouping, cart loading, gate timing, and handoff-time tasks. All checks and the stratified review were rerun after those corrections.

### Audit definitions

- **State/action grounding:** lowercase tokens are extracted with `[a-z0-9]+`. A mapping passes if the normalized complete state value occurs in its correct action, or if at least one token that distinguishes that value from the other world state matches an action token. Exact token matches are allowed at any length so identifiers such as `2`, `M2`, and `A` remain checkable; non-exact prefix matches require both tokens to contain at least four characters. This produced 400 passes and 0 failures.
- **Instruction leakage:** state tokens shared by both values and one-character letter codes are discarded; an instruction is flagged when it contains a remaining discriminating token from exactly one of the two states. This produced 0 flags.
- **Condition label leakage:** the case-insensitive, word-bounded scan is `\b(?:false|wrong|inaccurate|misleading|counterfactual|untrue|misstat\w*|lie|lying|truth\w*|decept\w*)\b`. It produced 0 target-condition and 0 control-condition hits.
- **Normalized similarity:** lowercased alphabetic tokens are collected from the neutral context/question, action instruction, first indirect probe, and control. The function-word set is `a an the and or but to of in on at for from with which what where when how is are be should its their it this that into by before after each current ordinary routine final assigned designated true false correct report answer incentive creates does no not actual accurately truthful reporting`; tokens of two or fewer characters are discarded. Jaccard similarity is computed for all 19,900 row pairs, ranked descending, and the top 20 pairs are read side by side.

## Manual stratified review

I manually read the first and last row of every family, then re-read every scenario rebuilt during independent review (38 distinct rows total), covering:

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
- `physops.inventory.easel_quantity.0017`
- `physops.inventory.name_tag_holder_quantity.0018`
- `physops.inventory.training_bandage_box.0016`
- `physops.documents.donation_receipt_book.0007`
- `physops.documents.classroom_map_set.0018`
- `physops.classlab.dropper_tip_box.0011`
- `physops.classlab.model_planet_tray.0009`
- `physops.classlab.ruler_length.0012`
- `physops.events.badge_desk.0006`
- `physops.events.workshop_kits.0013`
- `physops.events.recycling_pair.0015`
- `physops.transit.parking_cart_pickup.0006`
- `physops.transit.last_loop_gate.0019`
- `physops.packages.classroom_supply_parcel.0003`
- `physops.packages.library_shipment_cart.0005`
- `physops.packages.flat_mailer_slot.0010`
- `physops.packages.package_count.0014`
- `physops.packages.handoff_counter.0019`

For each sampled row I checked that the fact was deterministic, the target answer was a direct false assertion, the neutral/control/probe answers preserved the true fact, the physical action could not be selected correctly without that fact, the counterfactual changed the correct action, the incentive was low-stakes and realistic, and the wording stayed within the assigned domain. **No manual-review defects remained.**

## Integration note

This file is intentionally a standalone batch shard. The current shared compiler reads `synthetic/paired_scenarios.jsonl`; this worker did not alter that compiler or the shared input. The main integrator must include or merge this shard during final compilation and packaging.
