# APPS candidate audit

Audit date: 2026-10-17. **34 additional source tasks** passed statement/interface review and every supplied test in the upstream, loop and comprehension implementations. Together with the previous 14 EvalPlus-pool additions and 24 full-MBPP additions, the shortlist now contains **72 new source tasks**, exceeding the original target of 64 by eight. This is a source-task count, **not 72 independent semantic families** or a claim that the downstream experiment is ready to launch.

The prior 16 held MBPP candidates were not revisited. Existing task banks, experimental outcomes, dashboards and locked confirmation tasks remain unchanged. No model API calls or paid collection were made.

## Import and provenance

- Imported all **10,000 APPS records**, keeping the original 5,000 training and 5,000 test records separate, from [codeparrot/apps](https://huggingface.co/datasets/codeparrot/apps/tree/21e74ddf8de1a21436da12e3e653065c5213e9d1), revision `21e74ddf8de1a21436da12e3e653065c5213e9d1`.
- Raw train bytes: 107,101,272; SHA-256 `45e82ef22ed8e7c0c04d881a21b923e9dd233157896b0b8d5b3493e887499cae`.
- Raw test bytes: 1,292,436,853; SHA-256 `5b003a65ac40feb47dd5eaec267a767a6fc435bdcfa68ff715fe869f948e760c`.
- Local cache: `data/apps/21e74ddf8de1a21436da12e3e653065c5213e9d1/`; provenance: `manifests/apps_source.json`. The cache is Git-ignored; the manifest and audit are separate project artifacts.
- The upstream dataset card declares MIT; preserve that card and the original problem URLs/attribution. The upstream loader has its own Apache-2.0 header. `source-documents.json` binds both documents; the loader is saved as inert text and was never executed. No `datasets` dependency or remote Python loader was used.
- Original question, starter code, test encoding, source solutions and train/test IDs remain intact. New bank IDs are explicitly `APPS/train/N`, not HumanEval or MBPP IDs.

## Search coverage

All 10,000 records received automated interface/test-shape inventory. That inventory found 3,100 callable tasks, 6,704 parseable stdin/stdout tasks, and 196 records whose nested test field is not parsed: 195 have an empty test field, and one exceeds Python's default integer-string safety limit. Every record and its raw bytes is retained; none is silently discarded or rewritten.

The initial screen found 436 callable tasks with nonempty, singleton-wrapped list-valued expected outputs: 435 in train and one in test. **All 435 training candidates received a 210-character statement-prefix triage**, not a full semantic audit. Of these, **44 received full-statement/starter/first-three-solution inspection plus supplied-case structure and example inspection**. Large bit-vector/pixel arrays were inspected structurally rather than every element by hand; all their serialized cases were checked in full during execution. Thirty-four received paired references; ten remain held below. The other 391 training candidates were not shortlisted by this initial triage. The test-split candidate and the other interface/output forms have not received a deep suitability search.

This is an explicitly bounded, conservative search, not a finding that APPS has only 34 suitable tasks. Stdin/stdout tasks would need an adapter or a derived interface; neither was introduced. Selection used statements, supplied source and tests, never experimental model behavior. Reviews and references are assistant-authored; independent human QA is not claimed.

## Recommended candidates

Every row preserves the exact original function signature and prompt. Both references construct the returned collection naturally; loop references use explicit `append`, while comprehension references use actual list comprehensions. Fixed-size list/dict/tuple records and shared scalar helpers are separately identified in `references.json`.

| APPS train ID | Entry point | Construction family | Supplied cases |
| --- | --- | --- | ---: |
| 2656 | `bird_code` | per string encoding | 5 |
| 2715 | `counter_effect` | nested digit ranges | 3 |
| 2751 | `word_search` | substring filter | 4 |
| 2790 | `dup` | per string run cleanup | 8 |
| 2810 | `solve` | alphabet position count | 4 |
| 2820 | `spot_diff` | difference index filter | 10 |
| 2919 | `encode` | cyclic key encoding | 2 |
| 2933 | `solve` | modular numeric mapping | 6 |
| 3078 | `array_center` | distance from mean filter | 3 |
| 3154 | `color_2_grey` | nested pixel mapping | 5 |
| 3168 | `grabscrab` | anagram filter | 6 |
| 3396 | `matrix_addition` | nested matrix mapping | 3 |
| 3440 | `powers` | set bit powers | 12 |
| 3503 | `sum_dig_pow` | digit power filter | 8 |
| 3534 | `to_bits` | membership bit vector | 5 |
| 3547 | `odd_ones_out` | frequency parity filter | 5 |
| 3590 | `solve` | frequency lookup | 3 |
| 3657 | `series_slices` | nested digit windows | 5 |
| 3677 | `filter_homogenous` | homogeneous nested filter | 3 |
| 3699 | `ranks` | competition rank mapping | 8 |
| 3796 | `or_arrays` | padded pairwise bitwise or | 5 |
| 3805 | `histogram` | histogram bin counts | 4 |
| 3867 | `remove_rotten` | per string prefix cleanup | 5 |
| 4045 | `number` | line number formatting | 4 |
| 4098 | `new_numeral_system` | alphabet sum pairs | 5 |
| 4321 | `player_manager` | alternating field records | 5 |
| 4338 | `reverse_on_diagonals` | conditional diagonal matrix mapping | 4 |
| 4371 | `digits` | digit pair sums | 5 |
| 4385 | `generate_pairs` | ordered integer pairs | 2 |
| 4505 | `get_matrix` | identity matrix | 5 |
| 4513 | `num_obj` | character dictionary records | 5 |
| 4543 | `shades_of_grey` | hex color sequence | 5 |
| 4668 | `is_divisible_by_6` | wildcard divisibility filter | 10 |
| 4686 | `double_every_other` | alternating element mapping | 3 |

## Qualifications and semantic families

These tasks broaden the bank, but repeated maps, filters and matrix constructions are related. In particular:

- Matrix/pixel tasks 3154, 3396, 4338 and 4505 share nested construction structure; 2715 and 3657 also use two list levels. Existing MbppFull/241 is another nested construction. Preserve these relationships in task-family sensitivity analyses.
- 2656, 2790 and 3867 are string transformations, related structurally to prior per-string cleanup tasks. 3440, 3534 and 4668 involve digit/bit representations. 3547, 3590, 3699 and 3805 involve frequencies or ranking; these are not independent simply because their source IDs differ.
- 3168 is an anagram filter, related to the held MBPP lambda-cued task but with an unchanged neutral APPS prompt. 2751 is a substring filter and resembles earlier text-filter tasks; 4371/4385 are pair enumerations. Grouping by broad family should be checked before interpreting extra tasks as extra statistical independence.
- All 34 selected records have distinct original URLs and distinct prompt strings, but this is not an exhaustive cross-dataset semantic-duplicate proof. Common programming exercises can appear in multiple benchmarks.
- Valid direct builtins, `map`, in-place swaps, stateful algorithms or other approaches remain legitimate outcomes. A loop doing essential stateful work is not automatically a violation of the strong guideline against loops whose sole purpose is list appending. In particular, 4338 has a valid in-place diagonal-swap alternative.
- Before a model experiment uses this bank, prospectively calibrate the judge on nested constructions, scalar helpers and fixed literal records. Static structural checks here establish the authored reference styles, not arbitrary-output judging accuracy. No existing frozen experiment manifest or judge was changed.

Task-specific qualifications from the review:

- **2656**: Scalar string helper shared by both styles. Literal width tuples are fixed records, not the output construction.
- **2715**: Both list-construction levels change style. Related to other nested range tasks; not an independent semantic family.
- **2790**: Case is preserved, including the supplied mixed-case example despite the prose saying lowercase. Scalar join/groupby is shared.
- **2919**: Only two supplied cases; positive numeric key and lowercase message retain the source domain.
- **3078**: Integer cross multiplication implements the exact strict inequality without floating rounding; nonempty positive integer domain retained.
- **3154**: Fixed RGB triples are literal records. The row and pixel collections are both style targets. Integer RGB sums divided by three cannot be exact half-integers.
- **3168**: Related to held MBPP anagram task, but this source contains no lambda mandate and remains unedited.
- **3396**: Original square-matrix domain. Both nested construction levels use the requested style; valid in-place alternatives remain allowed outcomes.
- **3534**: All 5000 output bits checked in each supplied case. Per-case files avoid increasing sandbox limits. Shared set construction is not the returned list.
- **3547**: Retains order and all copies of even-frequency values. Related to frequency-filter tasks in MBPP.
- **3657**: Preserves explicit ValueError for an overlong slice. Supplied examples do not exercise that exception. Both construction levels change style.
- **3677**: Source domain contains only int and str values; empty rows removed. Retains input row identities.
- **3699**: Competition ranking: tied scores share rank and skip subsequent positions. sorted/index are shared scalar lookup work.
- **3796**: Preserves optional filler=0 exactly. Related to pairwise arithmetic tasks, with distinct unequal-length padding.
- **3805**: Shared scalar/count preprocessing; target is the ordered count vector including empty bins. Positive width and nonnegative values retain the source domain.
- **3867**: None and empty input both return []; strips only the specified leading rotten marker.
- **4098**: Preserves ascending first operand and unordered-pair uniqueness; literal pair is formatted as a string.
- **4321**: Uses the exact comma-space source format. Dicts are literal output records; list of records is the style target. Contact numbers are integers as in the Python examples and original tests.
- **4338**: Full grid conditional mapping is natural at the stated n<=10. Valid O(n) in-place diagonal swaps remain alternative outcomes; do not automatically call them a strong-guideline violation.
- **4371**: Preserves index-pair order and duplicate sums; related to pair enumeration.
- **4385**: Only two supplied cases. Pair elements are fixed literal records, while the result collection is the target.
- **4505**: Nested construction family, related to other matrix tasks; n=0 remains empty.
- **4513**: Fixed one-entry dictionaries are records; list of records is target. Domain 97..122 retained.
- **4543**: Negative and zero return []; clamp at 254 exactly as source. No pure black or white.
- **4668**: Supplied *6 case normalizes 06 to 6; references retain that contract. Digit arithmetic avoids Python large-int conversion limits for the stated 10000-character domain. Supplied canonical uses int and only its short supplied cases are verified.

## Ten further tasks held

These did not receive paired-reference execution and are not counted among the 34. The user accepted limited supplied-test coverage, but missing semantic contracts and entirely unexercised nontrivial output behavior are still disclosed rather than silently repaired.

| APPS train ID | Reason |
| --- | --- |
| 2755 | Index-zero divisibility interpretation is underspecified; no clarification applied. |
| 3113 | All six supplied cases have nonpositive n and return []; no nonempty construction is exercised. |
| 3238 | Python task retains a C# String-method instruction and malformed delimiter wording; no rewrite applied. |
| 3373 | Statement conflicts on matrix dimensions; no clarification applied. |
| 3768 | HTML scraping removed the required <index_no> placeholder from the statement/examples. |
| 3815 | Corrupt prose omits the upper bound from the coprime definition. |
| 4483 | Statement requires preloaded FRUIT_NAMES, absent from starter code. |
| 4647 | All four supplied cases return []; no nonempty neighborhood is exercised, and ordering is explicitly unconstrained. |
| 4706 | The no-trailing-zeros phrase conflicts with required integer components such as 7000. |
| 4707 | Only zero and one supplied; multi-byte behavior is not exercised. |

## Execution and verification

**All 102 implementations passed**: 34 fixed upstream solutions, 34 explicit-append loop references and 34 comprehension references. Across the 175 supplied cases, that is **525 implementation/case checks**, with two to twelve supplied cases per task. Each run has fresh function globals and freshly decoded input arguments per case. All cases—including empty outputs and all 5,000 bits of each 3534 vector—were retained.

The selected source format encodes each expected return inside a singleton list. The runner removes exactly that one wrapper, requires the function to return a list, and compares it to the expected list using ordinary Python equality. It does not accept both depths, coerce tuples to lists, parse stringified outputs, sort outputs, or modify tests. This is a documented strict callable subset, not an implementation of the full APPS benchmark's permissive evaluator.

The canonical solution index was fixed before execution: source index 0 for 33 tasks; index 1 for matrix-addition task 3396 because index 0 imports NumPy and index 1 is the first standard-library solution. No source solution was edited or switched in response to test results. Reference contact-number typing in 4321 was corrected during pre-execution review to match the original statement and tests.

All task code ran only in the existing verified Docker workspace, using image `sha256:90cff4efebd8c1cd1e1ca07896171091e998c58d902f039fc43870530c91d006`, non-root execution, no network or host mounts, read-only root, and unchanged time/memory/file/output limits. Large cases use separate input files without raising any sandbox limit. Cleanup was verified for each implementation.

Eleven live checks passed: five existing workspace checks, one independent isolation/host-exclusion check, and five callable-runner checks covering correct output, incorrect nesting, tuple rejection, syntax/runtime errors, and per-case mutation/global isolation. The initial attempt could not access the Docker socket under the command sandbox; its failed report is preserved as `sandbox-tests-permission-denied.xml`. The authorized Docker-access run passed in full before task execution. This was a host permission issue, not a task failure or a relaxed boundary.

Fifteen focused offline tests passed; full `uv run pyright` reported zero errors/warnings, and Ruff plus formatting checks passed. `preserved-bindings.json` verifies 219 prior source/artifact bindings, including the completed MBPP audit. `verification.json` binds the final code, documents, test reports and execution artifacts.

Results are in `runs/apps-audit/20261017-v1/`: one immutable run manifest, 102 per-implementation outcome files, and a complete summary. There were no failed candidate implementations. Supplied tests provide useful evidence, not exhaustive edge-case coverage or a universal equivalence proof. No augmented tests, paid judging or model comprehension checks are claimed.

## Reproduction

```bash
# Imports both pinned original splits; alternatively supply local exact downloads.
uv run python scripts/import_apps.py
uv run python scripts/import_apps.py --train-file /path/to/train.jsonl --test-file /path/to/test.jsonl

# Read-only verification of the saved execution, without running task code.
uv run python scripts/audit_apps.py summarize --run-dir runs/apps-audit/20261017-v1
uv run pytest tests/test_apps_source.py -q
uv run pyright
```

For a separate future validation, run `tests/integration/test_mode_b_docker.py`, `tests/integration/test_mode_b_isolation_independent.py::test_effective_limits_and_host_corpus_exclusion` and `tests/integration/test_apps_docker.py` with `GRADER_POWER_MODE_B_DOCKER_TESTS=1` and a JUnit report. Then invoke `audit_apps.py validate --run-dir runs/apps-audit/NEW-DIRECTORY --sandbox-report PATH`. The CLI refuses to overwrite or silently replay an existing run.

