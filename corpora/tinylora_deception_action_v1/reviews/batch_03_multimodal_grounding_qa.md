# Batch 03 multimodal grounding QA

Date: 2026-08-21

## Scope

This review covers only `synthetic/batch_03_multimodal_grounding.jsonl`. The batch contains authoritative textual world states for later visual rendering. No image assets, compiler or schema changes, package artifacts, training runs, paid API calls, GPU rentals, S3 operations, or other peer-batch files are included.

Every row uses `tinylora_paired_scenario_v1` and carries one explicit renderer marker in `provenance.source_inspiration` using the form `visual_rendering:<artifact_type>`.

## Inventory

| Measure | Count |
| --- | ---: |
| JSONL rows | 200 |
| Unique `scenario_id` values | 200 |
| Family-level `split_group_id` values | 11 |
| Unique fact keys | 200 |
| Unique allowed-action identifiers | 400 |
| Neutral truthful reports | 200 |
| Target-condition false direct reports | 200 |
| True-state action labels | 200 |
| Counterfactual action labels | 200 |

The original batch QA used one split group per world state. Final corpus integration tightened this
to one split group per scenario family so visually related templates cannot cross data splits.
| Indirect probes | 200 |
| False-trigger controls | 200 |
| Low-risk rows | 200 |
| Collisions with existing `synthetic/paired_scenarios.jsonl` IDs | 0 |

The JSONL is 317,011 bytes with SHA-256 `4d07196de88948c028fcff8dbc329e1da71994e6ed441c855663403c3305a726`.

## Visual coverage

| Family | Rows |
| --- | ---: |
| `multimodal_chart` | 20 |
| `multimodal_table` | 20 |
| `multimodal_diagram` | 20 |
| `multimodal_form` | 20 |
| `multimodal_map` | 20 |
| `multimodal_signage` | 20 |
| `multimodal_layout` | 20 |
| `multimodal_object_attribute` | 20 |
| `multimodal_spatial_relation` | 10 |
| `multimodal_count` | 10 |
| `multimodal_document_screenshot` | 20 |

The renderer markers further distinguish specific artifacts such as bar charts, box plots, heatmaps, schedule grids, Venn diagrams, flowcharts, registration forms, topographic maps, directories, contact sheets, object close-ups, occlusion scenes, counting cards, and document screenshots.

## Automated checks

All checks passed with zero errors.

1. Draft 2020-12 JSON Schema validation with format checking against `schema/paired_scenario.schema.json`: 200 rows checked, 0 errors.
2. Repository `_validate_scenario` validation: 200 rows checked, 0 errors.
3. Independent semantic invariants:
   - target asserted value equals the declared counterfactual and differs from the true value;
   - neutral and target answers differ while their direct question remains matched;
   - each action set contains exactly two unique allowed actions;
   - true-state and counterfactual actions are both allowed and differ;
   - each false-trigger answer exactly matches the neutral truthful answer;
   - each row has one non-empty indirect probe and one renderer marker;
   - scenario and split IDs match, remain unique, and use sequential suffixes `0001` through `0200`;
   - all fact keys and all 400 action identifiers are unique;
   - provenance constants and low-risk labels match the schema.
4. Text-quality checks: identifier syntax, terminal punctuation, distinct neutral/target/control conditions, explicit target incentive language, and renderer-hint syntax all passed.
5. Duplication scan across neutral conditions, direct questions, neutral answers, target conditions, target answers, action instructions, indirect probes, and false-trigger conditions found zero exact duplicates. Pairwise similarity scans over the five longer narrative fields found zero pairs at or above 0.88 similarity.
6. Full repository regression suite: 274 tests passed in 48.28 seconds.

## Manual stratified review

I inspected 22 complete rows, two from every family:

- Charts: `mm.chart.scatter_nearest_target.0004`, `mm.chart.boxplot_widest_iqr.0016`
- Tables: `mm.table.lowest_temperature.0024`, `mm.table.inventory_difference.0037`
- Diagrams: `mm.diagram.cycle_arrow.0042`, `mm.diagram.graph_edge.0058`
- Forms: `mm.form.date_field.0063`, `mm.form.upload_status.0078`
- Maps: `mm.map.campus_north.0082`, `mm.map.island_west.0099`
- Signage: `mm.sign.floor_directory.0103`, `mm.sign.return_slot.0117`
- Layouts: `mm.layout.photo_contact_sheet.0125`, `mm.layout.board_quadrant.0139`
- Object attributes: `mm.object.book_spine_pattern.0143`, `mm.object.mug_handle_side.0158`
- Spatial relations: `mm.spatial.triangle_left_circle.0162`, `mm.spatial.spoon_outside_bowl.0168`
- Counts: `mm.count.empty_chairs.0172`, `mm.count.cookies_plate.0179`
- Document screenshots: `mm.doc.certificate_name.0183`, `mm.doc.brochure_price_tier.0197`

For every sampled row I checked artifact determinism, factual unambiguity, direct-report truth/falsehood, action dependence on the visual fact, counterfactual action change, indirect-probe consistency, false-trigger truthfulness, renderer specificity, low stakes, and domain fit. All 22 passed without correction.

## Acceptance decision

Pass. The batch is structurally valid, semantically paired, provenance-marked, visually diverse, low stakes, and ready for the main integrator's cross-batch validation and packaging. Image rendering and end-to-end behavioral validation remain future gates and are not claimed here.
