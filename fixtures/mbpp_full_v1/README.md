# Full original MBPP candidate audit

Audit date: 2026-10-17. **24 additional task candidates** are recommended without changing their original statements, after source review and successful canonical/loop/comprehension checks. Together with the previous 14 additions, the conservative shortlist is **38 new source tasks**, still **26 short of the requested 64**. Sixteen further tested pairs remain held for concrete wording or contract decisions.

This imports the full original MBPP release separately from the existing MBPP+ pool. It does not replace historical source records, admit tasks to an experiment, or make any model API calls.

## Import and search coverage

- 974 exact original records, pinned to Google Research commit `f82046ba5aabbbb427dbfd38a254d26bff08b533` and SHA-256 `ccf64ceae9c5403bf50a044cb6d505bfd2a2963ee58338ba268fd65beab92a9f`.
- 378 IDs overlap our MBPP+ cache; **596 IDs are additional**. Identity overlap does not imply identical sanitized prompts or tests.
- All 596 additional statements were triaged. 115 collection-oriented or borderline tasks received source/statement/assertion inspection; 40 received two reference implementations. The other 481 received statement triage only, not an equally deep semantic audit.
- `review.json` accounts for all 974 source IDs exactly once: 378 existing, 40 paired candidates, 71 explicit reserves/exclusions, and 485 not shortlisted. Four deeply inspected tasks remain in the last category. This is a conservative shortlist, not a proof that no further suitable task exists.
- Selection used task statements, source code and tests, not experimental model outcomes. References and suitability decisions are assistant-authored; independent human QA is not claimed.

The raw import is at `data/mbpp-full/f82046ba5aabbbb427dbfd38a254d26bff08b533/mbpp.jsonl`, with the exact upstream README and repository LICENSE alongside it. `manifests/mbpp_full_source.json` records the source URL/hash and distinguishes the dataset-card CC-BY-4.0 declaration from the repository Apache-2.0 license. The upstream README documents original benchmark splits; we have not created new experimental splits or few-shot examples.

## Candidates with unchanged wording

Each row preserves the original entry-point signature, statement and source record. `MbppFull/N` is a separate bank identifier, while `upstream_mbpp_id` records the original `Mbpp/N` identity. Neither claims MBPP+ test coverage.

| Original ID | Entry point | Transformation / qualification |
| --- | --- | --- |
| 27 | `remove` | per string cleanup |
| 157 | `encode_list` | run length encoding. Group counting uses a scalar generator in both styles; each output record is a literal pair. Strings and lists both remain valid inputs. |
| 196 | `remove_tuples` | tuple length filter |
| 200 | `position_max` | extremum index filter. Preserves the canonical error on empty input. MbppFull/322 is the minimum analogue, not counted as a separate addition. |
| 215 | `decode_list` | run length decoding. A list item denotes [count, value]; a non-list denotes one literal element. Shared helper returns an iterator, never an accumulated output list. |
| 241 | `array_3d` | nested array construction. All three list-construction levels use the requested style; rows/planes are distinct objects. Requires judge calibration for nested constructions. |
| 298 | `intersection_nested_lists` | nested membership filter |
| 317 | `modified_encode` | run length encoding. Related to /157: singleton runs remain scalars. Keep the two tasks in one semantic-family sensitivity group. |
| 319 | `find_long_word` | regex word length. Related to earlier Mbpp/7 and MbppFull/330, not a wholly independent semantic family. Direct findall remains a legitimate neither-style alternative. |
| 345 | `diff_consecutivenums` | adjacent numeric mapping |
| 361 | `remove_empty` | truthy collection filter. Preserves original truthiness semantics and mixed string/list examples. /966 is the same filter over tuple examples and is not an independent addition. |
| 363 | `add_K_element` | nested record mapping. Outer list is the style target; tuple construction uses the same generator in both forms. Preserves arbitrary tuple lengths. |
| 503 | `add_consecutive_nums` | adjacent numeric mapping. Operator variant of /345 and /690; report the shared family. |
| 533 | `remove_datatype` | type exclusion. Input is a tuple and result is a list, as in every supplied assertion. isinstance retains subclass behavior. |
| 570 | `remove_words` | per string cleanup. The target is the outer list of cleaned lines; scalar text cleanup uses generator expressions identically in both references. Preserve whitespace normalization and empty cleaned lines. |
| 613 | `maximum_value` | record reduction mapping. Each record maps a key and nonempty value collection to key/maximum; max is shared scalar work. |
| 659 | `Repeat` | frequency filter. The statement specifies integers. Counter insertion order preserves the first-occurrence ordering of repeated values; do not extend claim to unhashable values. |
| 690 | `mul_consecutive_nums` | adjacent numeric mapping. Operator variant of /345 and /503; report the shared family. |
| 825 | `access_elements` | index gather. Preserves order and repeated indices, including Python negative-index behavior. |
| 869 | `remove_list_range` | nested range filter. Retains original row identities and original error on an empty row. |
| 906 | `extract_date` | regex record extraction. Date fields remain strings; no network access or date validity checking. A direct findall remains a valid alternative style. |
| 921 | `chunk_tuples` | chunk mapping. Retains final short chunk and original range behavior, including ValueError for N=0. |
| 936 | `re_arrange_tuples` | ordered lookup. Preserves last-value-wins input keys and repeated requested keys. |
| 951 | `max_similar_indices` | paired record mapping. Preserves two-component tuple records and shortest-list zip behavior from the source. |

These 24 source IDs cover 20 coarse transformation-family labels. They are **not 24 demonstrably independent semantic families**: 345/503/690 are adjacent arithmetic variants, 157/317 are related encoders, and 319 resembles the previous Mbpp/7 word-length task. Other relationships are recorded in the notes. Any later analysis should retain these relationships and include sensitivity to family grouping; renamed/operator variants do not supply the same diversity as unrelated tasks.

Valid direct builtins, regex `findall`, slicing or other approaches must remain valid alternative outcomes. The nested constructions and shared scalar helpers need prospective judge calibration. Passing the unchanged legacy AST diagnostic on all 40 pairs is structural reference evidence, not validation of a judge on arbitrary model outputs. Stateful alternative implementations must not automatically be labeled violations of a guideline that only forbids loops whose sole purpose is list appending.

## Tested but held

All the following pass the supplied assertions in all three implementations. Their holds concern prompt suitability or an underdetermined contract, not test count. The user allowed fewer tests; no semantic edits or algorithm-cue removal have been applied.

| Original ID | Required decision |
| --- | --- |
| 81 | Original says zip but canonical cycles the second tuple instead of truncating. Requires explicit clarification; no edit applied. |
| 249 | Proposed edit: remove "using lambda function". No prompt edit applied; differs from set intersection by preserving right-side order and multiplicity. |
| 358 | Proposed edit: remove "using map and lambda function". No edit applied. |
| 380 | Statement does not say the cell equals row index times column index. Needs explicit contract clarification; not applied. |
| 539 | Proposed edit: remove "using map function". Retains varying exponent per input, unlike earlier fixed-exponent task. |
| 621 | Prose says numeric values but canonical accepts only isdigit strings, leaving signed/decimal strings unchanged. Clarify this behavior before admission; no edit applied. |
| 648 | Canonical inserts None before the final item for odd-length inputs. Supplied tests cover only even lengths; do not silently restrict the domain or alter padding. |
| 682 | Proposed edit: remove "using map and lambda function". No edit applied. |
| 696 | Prose says zip but code/tests concatenate paired rows. Proposed clarification should preserve shortest-list behavior; no edit applied. |
| 817 | Proposed edit: remove "using lambda function". No edit applied. |
| 857 | Proposed edit: remove "using map function"; outer list construction is target; list(text) is shared character conversion. |
| 861 | Both prompt and original entry-point name anagram_lambda prime lambda. A derived neutral task would need explicit approval/provenance; no edit applied. |
| 864 | Both prompt and original entry-point palindrome_lambda prime lambda; no edit applied. |
| 865 | Prose says print n-times a list, but code/tests multiply each item. Proposed statement must also remove map cue; no edit applied. |
| 883 | Proposed edit: remove "using lambda function". AND variant of /817; same family. |
| 898 | Clarify one representative for each maximal consecutive run whose length is exactly n; no edit applied. |

Ten of these sixteen contain `map`/`lambda` cues (one also has contradictory prose); six have other contract ambiguities. The full review also records source bugs and duplicates, including removal while iterating, mismatched return types, and sort-versus-reverse discrepancies. No tests were weakened to make those tasks eligible.

## Verification and limitations

**All 120 implementations passed:** 40 canonical, 40 comprehension and 40 explicit-append loop solutions. Each shortlisted task has three supplied assertions and no challenge assertions, giving **360 implementation/assertion checks**. The 24 recommended tasks account for 216 of those checks. The full dataset has 2,922 ordinary assertions and 16 challenge assertions, but the latter belong to other tasks.

This uses the original supplied assertions, as expressly approved. It is not augmented MBPP+ coverage, edge-case completeness, or a universal equivalence proof. Both references preserve the source function signature; source/domain caveats are explicit. Original prompt strings and source/test records remain unchanged.

All task code ran inside the existing verified Docker workspace using pinned image `sha256:90cff4efebd8c1cd1e1ca07896171091e998c58d902f039fc43870530c91d006`, with non-root execution, no network, no host mounts and resource limits. Ten live checks covered the workspace, an independent isolation check and the new assertion runner. Nine focused offline tests, full Pyright and Ruff passed. Container cleanup was verified.

The execution artifacts are in `runs/mbpp-full-audit/20261017-v1/`. Its final summary initially failed because the legacy diagnostic metadata accepts `Mbpp/N` but not `MbppFull/N`. The old script and run are preserved. A narrow offline adapter now uses the upstream spelling only for that legacy diagnostic. `runs/mbpp-full-audit/20261017-v1-offline/` reuses all 120 byte-identical outcomes, binds both original and current evidence, and contains the completed summary. No task code was rerun for this report repair.

`preserved-bindings.json` verifies 86 historical source/artifact bindings, including the previous expansion audit. `verification.json` links the current source, reports, copied original outcomes and provenance. Earlier experiments, task banks, dashboards and locked confirmation tasks remain unchanged.

## Reproduction

Import or re-verify the exact source bytes:

```bash
uv run python scripts/import_mbpp_full.py
# Or: uv run python scripts/import_mbpp_full.py --source-file /path/to/mbpp.jsonl
```

Inspect the saved results without task execution:

```bash
uv run python scripts/audit_mbpp_full.py summarize \
  --run-dir runs/mbpp-full-audit/20261017-v1-offline
uv run pytest tests/test_mbpp_full.py -q
uv run pyright
```

For a separate future validation, first run the five `test_mode_b_docker.py` tests, `test_effective_limits_and_host_corpus_exclusion` from the independent isolation suite, and the four `test_mbpp_full_docker.py` tests with `GRADER_POWER_MODE_B_DOCKER_TESTS=1` and a JUnit report. Then use `validate --run-dir runs/mbpp-full-audit/NEW-DIRECTORY --sandbox-report PATH`. Existing results are never silently overwritten.
