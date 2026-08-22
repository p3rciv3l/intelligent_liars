# Batch 02 digital tools QA

## Result

**PASS** — 200 of 200 authored scenarios passed the independent structural and semantic checks, the repository scenario validator, and manual stratified review.

This review covers only `synthetic/batch_02_digital_tools.jsonl`. It does not compile or package the corpus, launch training, measure model behavior, use paid APIs, or upload artifacts.

## Inventory

| Family | Scenarios |
|---|---:|
| `digital_software_configuration` | 20 |
| `digital_file_version_management` | 20 |
| `digital_spreadsheets` | 20 |
| `digital_calendars` | 20 |
| `digital_dashboards` | 20 |
| `digital_fictional_databases` | 20 |
| `digital_content_publishing` | 20 |
| `digital_test_environments` | 20 |
| `digital_support_queues` | 20 |
| `digital_tool_selection` | 20 |
| **Total** | **200** |

All 200 records have `risk_level: low`. The JSONL contains 200 lines and 351,222 bytes. Its pre-commit SHA-256 is `be51ffadfe3b0a653ba46571e10cbda808f9f8ef9578a27633f39b75adf472f7`.

## Automated validation

An independent dependency-free validator parsed every JSONL row and enforced the current schema shape, including exact object keys and non-empty required text. It also enforced the corpus-specific semantic gates:

- 200 unique `scenario_id` values;
- 200 unique `split_group_id` values, each equal to its scenario ID;
- 200 unique `(family, fact_key)` pairs;
- zero ID collisions with the 21 scenarios in `synthetic/paired_scenarios.jsonl`;
- exactly 20 records in each of the ten declared families;
- every true value differs from its counterfactual;
- every target `asserted_value` equals the counterfactual value and differs from the true value;
- every target direct answer differs from its neutral answer;
- every action task declares exactly two unique allowed actions;
- every true-state and counterfactual action is allowed, and the two correct actions differ;
- every record contains exactly one indirect probe;
- every false-trigger control repeats the truthful neutral answer;
- all 200 target conditions explicitly state a low-stakes benefit for inducing the false belief;
- all 200 false-trigger controls explicitly remove the incentive to misreport;
- all 200 target-condition strings and all 200 false-trigger condition strings are unique;
- the repeated lexical cues `benefits if`, `, so `, and ` actually ` occur zero times in the corresponding target-condition/action fields;
- provenance model, mode, date, and inspiration fields match the authored-batch contract.

Result: `valid: true`, 200 records, zero errors.

The repository's existing `_validate_scenario` implementation was also run independently across all 200 rows.

Result: `valid: true`, 200 records, zero errors.

Repository tests:

- focused corpus tests: 8 passed;
- full suite: 274 passed in 48.56 seconds.

## Manual stratified review

Thirty records were manually inspected: the first, tenth, and twentieth scenario in every family (15% of the batch). The sample IDs were:

- calendars: `digital.calendar.recurrence.b02`, `digital.calendar.end_time.b02`, `digital.calendar.canonical_event.b02`;
- content publishing: `digital.publishing.headline.b02`, `digital.publishing.language.b02`, `digital.publishing.source_format.b02`;
- dashboards: `digital.dashboard.metric_source.b02`, `digital.dashboard.inclusion_flag.b02`, `digital.dashboard.label_source.b02`;
- fictional databases: `digital.database.primary_schema.b02`, `digital.database.initial_state.b02`, `digital.database.timestamp_field.b02`;
- file/version management: `digital.file_version.handbook.b02`, `digital.file_version.license_notice.b02`, `digital.file_version.backup_snapshot.b02`;
- software configuration: `digital.software_configuration.formatter_width.b02`, `digital.software_configuration.decimal_style.b02`, `digital.software_configuration.line_endings.b02`;
- spreadsheets: `digital.spreadsheet.item_name_column.b02`, `digital.spreadsheet.unit_scale.b02`, `digital.spreadsheet.export_sheet.b02`;
- support queues: `digital.support_queue.destination.b02`, `digital.support_queue.category.b02`, `digital.support_queue.sort_order.b02`;
- test environments: `digital.test_environment.browser.b02`, `digital.test_environment.locale.b02`, `digital.test_environment.runner_mode.b02`;
- tool selection: `digital.tool_selection.schema_validator.b02`, `digital.tool_selection.duplicate_files.b02`, `digital.tool_selection.pattern_test.b02`.

Each sampled record was checked for deterministic facts, unambiguous alternatives, realistic low-stakes incentives, direct neutral/target report divergence, action dependence on the hidden fact, a counterfactual that changes the correct action, an indirect truthful probe, and a truthful no-incentive control.

Result after the final review fixes: 30 passed and 0 rejected.

## Review remediation

The final two-axis review identified and corrected two issues before commit:

- Five repeated false-trigger templates and repeated target/action cue phrases were replaced. Every false-trigger condition now names its scenario-specific fact, all 200 control conditions are unique, target incentives use varied connective and advantage language, and the repeated `actually` cue was removed from action instructions.
- `digital.spreadsheet.unique_key.b02` originally used mutating-sounding `deduplicate_on_column_*` action names for a read-only duplicate check. They were replaced with the exact non-mutating actions `check_duplicates_on_column_C` and `check_duplicates_on_column_E`.

After those changes, both structural validators, the 30-record stratified review, the safety scan, and the eight focused corpus tests were rerun successfully. The earlier full-suite result remains 274 passed; the review changes touch only this JSONL and QA report, which the full suite does not ingest.

## Scope and safety review

The batch stays within benign digital and information workflows. Files, databases, queues, and records are explicitly sample, demo, sandbox, local, mock, or fictional where the distinction matters. Actions select, open, preview, validate, query, format, route fictional records, or run isolated tests. The batch contains no physical-logistics workflows, image-dependent facts, interpersonal reporting scenarios, credential handling, privacy-invasive operations, cybersecurity instructions, harmful operations, or destructive file/database actions.

## Handoff boundary

This batch and report are ready for integrator cherry-pick. Final cross-batch compilation, package manifests, S3 upload, training, and behavioral/scientific validation remain the sole responsibility of the main orchestrator.
