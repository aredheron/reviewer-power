# Agent semantic suitability review: 84 EvalPlus tasks

Review date: 2026-10-09. This is an agent-authored inspection record, not human approval, a reference-solution pass, or a frozen pilot set. No model calls, benchmark execution, task edits or source-cache modifications were performed.

The population is exactly the 84 unique original IDs carrying `list_output_evidence` in the existing candidate manifest: 35 HumanEval+ and 49 MBPP+. All 84 were read through the existing hash-verified pinned loader, including full original statements, signatures, canonical implementations and auxiliary contracts. No truncated packet excerpt was used as the basis of a task judgment. No grader outcomes were inspected.

## Recommendation counts

| Recommendation | Tasks |
| --- | ---: |
| `suitable_for_reference_validation` | 51 |
| `needs_resolution` | 17 |
| `unsuitable` | 16 |

Suitable means that both approaches appear natural under the original task; every such judgment still needs independently written references, complete base/plus oracle verification and human QA. Needs-resolution cases name specific decisions below. Unsuitable cases fail a substantive criterion in this agent review; making their source AST scoreable would not remove that objection.

## Provenance and inspection limits

- `manifests/evalplus_source.json` SHA-256: `e94aba750698d78b17bc238d5c7c2b18fa3282453ef7c5cf9406b5d99151a86d`.
- `manifests/task_candidates.jsonl` SHA-256: `cf5b9be393f1d4b8ee2df2331200f296b49bb9fbbc7a1b8c4535d10d9a96b5d5`.
- `planning/task_screening_review.md` SHA-256: `8794512d7b1cae78f5d4dde195cf5fc864861a62820d24276886d38cc30de506`.
- `src/grader_power/score_style.py` SHA-256: `edcfcb4c9a1c5eb47c2bcc0f69fd00039dcc4f3ace452d8b9ea32417219434b8`.
- EvalPlus 0.3.1; HumanEval+ v0.1.10 and MBPP+ v0.2.0. Per-record source hashes, prompt/implementation/contract hashes and aggregate input counts are in the JSONL. Full solutions and hidden inputs are excluded from these reports.
- The auxiliary contracts were inspected as evidence of intended domains, not assumed to be visible instructions or runtime validation inserted into reference implementations. The reports do not narrow them.
- No benchmark or solution code was executed. Sixteen synthetic, deliberately inert code strings were passed only to the existing extractor and AST scorer. Generic mapping/filtering, helpers, preprocessing, nested flattening and empty guards received the intended labels. Post-sort output mutation was rejected. Nested column construction received two comprehension features but only one loop feature. These are shape probes, not per-task reference or functional tests.
- EvalPlus special-oracle source confirms set-equality comparison for Mbpp/111 and Mbpp/769. The default evaluator also has floating-output tolerance logic even when source atol is zero. An eventual oracle pass alone cannot certify output types, ordered-list semantics, mutation or alias independence; audit these explicitly.
- Mbpp/793 is outside this 84-task population because it does not have list_output_evidence. Its recorded empty-object plus_input anomaly remains unchanged in the source/candidate manifests; no review record was silently omitted or added for it.

## Source and family coverage

| Source | Suitable | Needs resolution | Unsuitable | Total |
| --- | ---: | ---: | ---: | ---: |
| HumanEval+ | 20 | 6 | 9 | 35 |
| MBPP+ | 31 | 11 | 7 | 49 |

Families are semantic labels assigned in this review, not inherited keyword matches. Closely related families still share redundancy groups in the JSONL.

| Family | Suitable | Needs resolution | Unsuitable | Total |
| --- | ---: | ---: | ---: | ---: |
| adjacent_pair_mapping | 1 | 0 | 0 | 1 |
| aligned_filtering | 1 | 0 | 0 | 1 |
| bounded_range_filtering | 1 | 0 | 0 | 1 |
| collatz_state | 0 | 0 | 1 | 1 |
| column_reduction_mapping | 0 | 1 | 0 | 1 |
| combinatorial_materialization | 0 | 0 | 1 | 1 |
| conditional_indexed_mapping | 2 | 0 | 0 | 2 |
| conditional_numeric_mapping | 0 | 1 | 0 | 1 |
| conditional_scalar_mapping | 0 | 1 | 0 | 1 |
| conditional_sequence_construction | 1 | 0 | 0 | 1 |
| coordinate_product_construction | 0 | 1 | 0 | 1 |
| digit_predicate_filtering | 0 | 1 | 0 | 1 |
| effectful_sorting | 0 | 0 | 1 | 1 |
| elementwise_mapping | 5 | 0 | 0 | 5 |
| factorization_state | 0 | 0 | 1 | 1 |
| filtering | 3 | 0 | 0 | 3 |
| flattening_with_ordering | 0 | 1 | 0 | 1 |
| format_mapping | 1 | 0 | 0 | 1 |
| frequency_filtering | 1 | 1 | 0 | 2 |
| fresh_record_construction | 1 | 0 | 0 | 1 |
| identity_materialization | 0 | 0 | 1 | 1 |
| indexed_filter_flattening | 1 | 0 | 0 | 1 |
| indexed_mapping | 1 | 0 | 0 | 1 |
| indexed_slicing | 2 | 0 | 0 | 2 |
| intersection_filtering | 0 | 1 | 0 | 1 |
| interspersing | 0 | 1 | 0 | 1 |
| iterative_sieve_state | 0 | 0 | 1 | 1 |
| lookup_mapping | 1 | 0 | 0 | 1 |
| matrix_transposition | 0 | 1 | 0 | 1 |
| membership_filtering | 1 | 0 | 0 | 1 |
| nested_collection_mapping | 2 | 0 | 0 | 2 |
| numeric_normalization | 1 | 0 | 0 | 1 |
| numeric_range_filtering | 0 | 1 | 0 | 1 |
| one_level_flattening | 1 | 0 | 0 | 1 |
| ordered_filtering | 0 | 1 | 0 | 1 |
| ordered_numeric_mapping | 1 | 0 | 0 | 1 |
| paired_mapping | 4 | 0 | 0 | 4 |
| pairwise_record_mapping | 1 | 0 | 0 | 1 |
| parity_filtering | 4 | 0 | 0 | 4 |
| prefix_state | 0 | 0 | 1 | 1 |
| record_construction | 0 | 1 | 0 | 1 |
| record_extraction | 4 | 0 | 0 | 4 |
| recurrence_state | 0 | 0 | 1 | 1 |
| recursive_flattening | 0 | 0 | 1 | 1 |
| run_deduplication | 1 | 0 | 0 | 1 |
| run_grouping | 1 | 0 | 0 | 1 |
| scalar_helper_mapping | 1 | 0 | 0 | 1 |
| scalar_predicate_filtering | 3 | 0 | 0 | 3 |
| set_membership_filtering | 1 | 0 | 0 | 1 |
| sieve_filtering | 0 | 1 | 0 | 1 |
| sign_filtering | 1 | 0 | 0 | 1 |
| sorted_deduplication | 0 | 1 | 0 | 1 |
| sorting_only | 0 | 0 | 5 | 5 |
| sorting_selection_only | 0 | 0 | 1 | 1 |
| string_length_filtering | 1 | 1 | 0 | 2 |
| symmetric_difference_filtering | 0 | 1 | 0 | 1 |
| tokenization_materialization | 0 | 0 | 1 | 1 |
| type_filtering | 1 | 0 | 0 | 1 |

## What the static screen missed

- The 18-task priority shortlist contains 17 tasks suitable for reference validation and one needing wording resolution (Mbpp/572). None is treated as approved. Another 34 suitable tasks came from outside that shortlist; static family/call/helper flags under-ranked them.
- Concrete missed candidates include indexed prefix construction (HumanEval/14), lookup mapping (HumanEval/17), frequency filtering with local preprocessing (HumanEval/26), partial-position reconstruction (HumanEval/33 and /37), polynomial derivatives (HumanEval/62), coordinate filtering (HumanEval/87), strided splitting (Mbpp/265), formatting (Mbpp/390), interleaving (Mbpp/578) and adjacent pairing (Mbpp/766).
- Canonical syntax is not eligibility: HumanEval/9 and /101, plus Mbpp/635, have comprehensions but fail the substantive suitability test here. Conversely, filter/map-based canonical tasks and loop-based coordinate construction remain candidates.
- Set use does not always imply unordered output: HumanEval/58 explicitly sorts the result. But Mbpp/111 and /769 have unresolved ordering and set-equality oracles. Local counters/sorted temporaries in HumanEval/26, /33 and /37 do not establish required external side effects.
- Mbpp/265 is strided index partitioning, not record extraction; Mbpp/612 is transposition and needs an explicit region decision, not automatic rejection solely because it returns two lists. Mbpp/104 and /569 transform every sublist, whereas Mbpp/12 merely sorts the outer list.
- Mbpp/391 (dictionary versus list of dictionaries), Mbpp/615 (row versus column averages), HumanEval/149 (equal-length claim contradicted by examples) and Mbpp/745 (negative-domain exception) need explicit resolution. HumanEval/81 has an example-name mismatch. The existing generated screen remains untouched.

## Provisional pilot suggestions — not membership

The following 12 are a diverse review starting point drawn from the 51 suitable judgments, not a quota imposed on eligibility. They cover seven HumanEval+ and five MBPP+ tasks. All may be replaced after reference validation and independent human QA, without consulting treatment effects. No split or final membership is assigned.

| Original ID | Why suggest it |
| --- | --- |
| HumanEval/7 | Basic order-preserving string filter; a simple baseline with no scalar helper. |
| HumanEval/17 | Finite lookup mapping over tokens; contrasts cleanly with numeric arithmetic. |
| HumanEval/21 | Shared numeric preprocessing followed by affine mapping; adds floating arithmetic. |
| HumanEval/26 | Frequency-based filtering; separates local counting from the scored output. |
| HumanEval/33 | Conditional indexed reconstruction after partial sorting. |
| HumanEval/62 | Index-dependent arithmetic mapping with omission of a constant term. |
| HumanEval/87 | Ragged two-dimensional coordinate filtering with explicit reverse-column order. |
| Mbpp/272 | Direct field extraction from records, preserving value types. |
| Mbpp/390 | String formatting of heterogeneous values with a parameterized format. |
| Mbpp/578 | Clear one-level flattening/interleaving of aligned triples. |
| Mbpp/754 | Aligned three-list filtering, distinct from ordinary membership filtering. |
| Mbpp/778 | Consecutive-run grouping with eager consumption of shared iterator preprocessing. |

Avoid filling the pilot with near-duplicates: Mbpp/412 and /629 are even filters; /426 and /554 are odd filters; /104 and /569 sort each sublist; HumanEval/26 and Mbpp/572 target singleton frequencies; HumanEval/33 and /37 differ mainly in the selected index stride. Power mappings and first/last/nth-field extraction also have limited family diversity. Similarity is recorded separately from eligibility.

## Scorer and wording decision gates

The current returned_list_v1 rule accepts a final comprehension or a final local accumulator populated by append loops; helper and preprocessing work can stay outside the designated output. This does not force the canonical implementation into either style. Sorting before an independent filter/mapping can expose the same final transformation without changing the task contract; all such equivalence claims remain unexecuted.

Mbpp/108 needs support for a collection-building region before final ordering, or another independently justified region contract. Mbpp/612 needs a choice between outer column construction and inner element extraction: pure-style probes classify, but nested mixed alternatives are not symmetrically covered. These are scorer/region questions and remain separate from scientific suitability.

Exact proposed wording replacements appear in the affected entries and JSONL. They are conditional proposals only. No original statement has been changed. Do not fix negative numeric domains, recursive depth, ordering or output types by silently weakening the problem.

## Suitable for reference validation (51)

**Complete ID list:** HumanEval/6, HumanEval/7, HumanEval/14, HumanEval/17, HumanEval/21, HumanEval/22, HumanEval/26, HumanEval/29, HumanEval/30, HumanEval/33, HumanEval/37, HumanEval/42, HumanEval/58, HumanEval/62, HumanEval/87, HumanEval/104, HumanEval/117, HumanEval/129, HumanEval/152, HumanEval/163, Mbpp/8, Mbpp/75, Mbpp/104, Mbpp/161, Mbpp/265, Mbpp/272, Mbpp/282, Mbpp/390, Mbpp/412, Mbpp/413, Mbpp/424, Mbpp/426, Mbpp/436, Mbpp/447, Mbpp/450, Mbpp/456, Mbpp/460, Mbpp/554, Mbpp/568, Mbpp/569, Mbpp/578, Mbpp/618, Mbpp/623, Mbpp/629, Mbpp/728, Mbpp/730, Mbpp/749, Mbpp/754, Mbpp/766, Mbpp/778, Mbpp/780.

### HumanEval/6 — parse_nested_parens

**Family:** scalar_helper_mapping. **Original triage:** manual_review. **Signature:** `parse_nested_parens(paren_string: str) -> List[int]`.

**Observed / judgment:** Each space-separated parenthesis group produces one maximum depth; the internal helper scans a group and returns a scalar.

**Competing approaches:** Comprehension: Apply a shared depth-counting helper to each nonempty token from splitting on literal spaces. Loop: Use the same tokens, skip empty tokens and append the helper's depth.

**Equivalence and cost (inferred):** Both preserve group order and compute the same depth in linear total character time. Keep the same splitting rule; do not replace the target with the helper's internal stateful scan.

**Scoring / unresolved checks:** `compatible`. The auxiliary contract validates global balance, not balance of each space-delimited group. Preserve the observed per-token scan and verify unusual group boundaries on the full oracle.

### HumanEval/7 — filter_by_substring

**Family:** filtering. **Original triage:** priority_review. **Signature:** `filter_by_substring(strings: List[str], substring: str) -> List[str]`.

**Observed / judgment:** The task retains strings containing the given substring; the canonical implementation uses filter, not a comprehension.

**Competing approaches:** Comprehension: Retain each string when substring membership succeeds. Loop: Traverse strings and append those satisfying the identical membership predicate.

**Equivalence and cost (inferred):** Order and duplicates are retained; empty substring retains every string, and empty input returns an empty list. Both have the same substring-search cost and no input writes.

**Scoring / unresolved checks:** `compatible`. Verify empty strings, Unicode and repeated values; canonical syntax supplies no positive eligibility evidence.

### HumanEval/14 — all_prefixes

**Family:** indexed_slicing. **Original triage:** manual_review. **Signature:** `all_prefixes(string: str) -> List[str]`.

**Observed / judgment:** The task returns every nonempty prefix, shortest first.

**Competing approaches:** Comprehension: Construct the prefix ending at each position from one through string length. Loop: Append the same prefix slice while traversing the same positions.

**Equivalence and cost (inferred):** Empty strings produce no prefixes; Unicode slicing, order and lengths are identical. Both must copy a quadratic total number of characters because the output itself has that size.

**Scoring / unresolved checks:** `compatible`. Confirm practical input lengths in full-oracle execution; this is substantive prefix construction, not copying a completed list.

### HumanEval/17 — parse_music

**Family:** lookup_mapping. **Original triage:** manual_review. **Signature:** `parse_music(music_string: str) -> List[int]`.

**Observed / judgment:** Tokens denote musical durations; the auxiliary contract restricts nonempty inputs to the three documented tokens, with an empty-string special case.

**Competing approaches:** Comprehension: Map each token to its duration through a shared lookup or scalar helper after an empty-string guard. Loop: After the same guard, append the duration for each token in the same order.

**Equivalence and cost (inferred):** Both are linear in input length, retain repeated notes and return integers. Use literal-space tokenization; do not invent behavior for malformed tokens outside the auxiliary contract.

**Scoring / unresolved checks:** `compatible`. A lookup must preserve the original allowed domain; the helper's implicit None outside that domain is not a license to silently repair malformed notes.

### HumanEval/21 — rescale_to_unit

**Family:** numeric_normalization. **Original triage:** priority_review. **Signature:** `rescale_to_unit(numbers: List[float]) -> List[float]`.

**Observed / judgment:** The auxiliary contract requires at least two numeric elements and a strict maximum greater than minimum; one affine transformation applies to each element.

**Competing approaches:** Comprehension: Compute shared extrema and reciprocal range, then map each number using subtraction followed by multiplication by that reciprocal. Loop: Use identical extrema and arithmetic order and append each normalized number.

**Equivalence and cost (inferred):** Both use linear work, retain order and duplicates, and do not mutate input. Preserve floating-point operation order rather than substituting division independently.

**Scoring / unresolved checks:** `compatible`. The visible statement omits the nonconstant-input restriction present in the auxiliary contract. Full oracle and human contract review must confirm edge-domain handling; no equal-valued-input repair is proposed.

### HumanEval/22 — filter_integers

**Family:** type_filtering. **Original triage:** priority_review. **Signature:** `filter_integers(values: List[Any]) -> List[int]`.

**Observed / judgment:** The canonical predicate accepts exact int values, not every isinstance-compatible subclass.

**Competing approaches:** Comprehension: Retain each value whose exact type is int. Loop: Apply the same exact-type predicate and append matching original values.

**Equivalence and cost (inferred):** Both are linear and retain order and duplicate integers. Booleans must be excluded even though Python treats bool as an int subclass; input objects must not be coerced.

**Scoring / unresolved checks:** `compatible`. Verify bool, mixed objects and empty lists; the meaning of integer must be anchored to the observed canonical predicate.

### HumanEval/26 — remove_duplicates

**Family:** frequency_filtering. **Original triage:** manual_review. **Signature:** `remove_duplicates(numbers: List[int]) -> List[int]`.

**Observed / judgment:** All values occurring more than once are removed, not reduced to one copy; order of remaining values is explicit.

**Competing approaches:** Comprehension: Build shared occurrence counts, then retain original numbers whose count is exactly one. Loop: Use the same counts and append original numbers whose count is one.

**Equivalence and cost (inferred):** Both are linear with a count table over the integer domain. Canonical dictionary writes are local preprocessing, not required external mutation.

**Scoring / unresolved checks:** `compatible`. Freeze only the final filtering region for later QA; the counting loop is outside the style outcome. This corrects an overconservative static side-effect concern.

### HumanEval/29 — filter_by_prefix

**Family:** filtering. **Original triage:** priority_review. **Signature:** `filter_by_prefix(strings: List[str], prefix: str) -> List[str]`.

**Observed / judgment:** Strings are retained by the startswith predicate and the canonical implementation uses filter.

**Competing approaches:** Comprehension: Retain each string that starts with the supplied prefix. Loop: Append each string satisfying the same prefix test.

**Equivalence and cost (inferred):** Both preserve order, duplicates and the effect of an empty prefix, with identical prefix-comparison cost and no input mutation.

**Scoring / unresolved checks:** `compatible`. Keep startswith semantics rather than substring membership; verify Unicode and empty strings.

### HumanEval/30 — get_positive

**Family:** filtering. **Original triage:** manual_review. **Signature:** `get_positive(l: list)`.

**Observed / judgment:** The contract requests positive list elements, and the canonical predicate is comparison with zero.

**Competing approaches:** Comprehension: Retain elements greater than zero. Loop: Append an element when the same comparison succeeds.

**Equivalence and cost (inferred):** Both preserve order and duplicates, exclude zero and negative zero, and preserve comparison errors instead of converting values. Linear comparisons in each style.

**Scoring / unresolved checks:** `compatible`. The auxiliary contract only states list input; exact numeric/comparison behavior and exceptional values need full-oracle checking.

### HumanEval/33 — sort_third

**Family:** conditional_indexed_mapping. **Original triage:** manual_review. **Signature:** `sort_third(l: list)`.

**Observed / judgment:** Only positions divisible by three are replaced with the sorted values from those positions; other positions stay unchanged.

**Competing approaches:** Comprehension: Precompute the sorted every-third slice, then map each index to its ranked replacement or unchanged input value. Loop: Use the same sorted slice and index rule while appending one value per position.

**Equivalence and cost (inferred):** Both are O(n log n) in the selected subset plus linear assembly, preserve nonselected positions and duplicates, and need no input mutation.

**Scoring / unresolved checks:** `compatible`. Canonical sorting mutates only a fresh temporary list. A precomputed slice avoids an incidental second comprehension; the final reconstruction is the scoring region.

### HumanEval/37 — sort_even

**Family:** conditional_indexed_mapping. **Original triage:** manual_review. **Signature:** `sort_even(l: list)`.

**Observed / judgment:** Even positions are sorted among themselves while odd positions keep their original values.

**Competing approaches:** Comprehension: Precompute sorted even-position values, then map every index to a replacement or the unchanged odd-position value. Loop: Traverse the same indices and append according to the same parity rule.

**Equivalence and cost (inferred):** Both preserve length, duplicates, numeric types and input immutability, with O(n log n) preprocessing and linear reconstruction.

**Scoring / unresolved checks:** `compatible`. Specify the final reconstruction as the sole output-building region; local sorting is incidental preprocessing. Closely redundant with HumanEval/33.

### HumanEval/42 — incr_list

**Family:** elementwise_mapping. **Original triage:** priority_review. **Signature:** `incr_list(l: list)`.

**Observed / judgment:** Each input element is incremented by one.

**Competing approaches:** Comprehension: Map every input element through addition of one. Loop: Append the result of the same addition for each input element.

**Equivalence and cost (inferred):** Both retain order and multiplicity, return an empty list on empty input, and preserve scalar arithmetic/errors. Linear scalar operations without input mutation.

**Scoring / unresolved checks:** `compatible`. Avoid coercing numeric types; full-oracle correctness remains unverified.

### HumanEval/58 — common

**Family:** set_membership_filtering. **Original triage:** manual_review. **Signature:** `common(l1: list, l2: list)`.

**Observed / judgment:** The result contains sorted unique values common to both input lists.

**Competing approaches:** Comprehension: Precompute sorted unique values from the first list and a membership set from the second, then retain members of that set. Loop: Traverse the same sorted unique values and append those in the second set.

**Equivalence and cost (inferred):** Both preserve the canonical hashability and sorting requirements, remove duplicates and produce sorted output. Shared preprocessing plus a substantive membership filter has comparable complexity.

**Scoring / unresolved checks:** `compatible`. Set construction is incidental and final ordering is explicit, so the static set-order flag is not by itself an exclusion. Verify unusual equality/comparison cases.

### HumanEval/62 — derivative

**Family:** indexed_mapping. **Original triage:** manual_review. **Signature:** `derivative(xs: list)`.

**Observed / judgment:** Coefficients become derivative coefficients; the constant term is omitted and coefficient i is multiplied by i.

**Competing approaches:** Comprehension: Map indices starting at one to coefficient-times-index values. Loop: Traverse the same indices and append the same products.

**Equivalence and cost (inferred):** Both preserve coefficient order, numeric behavior and trailing zeros; a constant polynomial returns an empty list. Linear coefficient arithmetic, no mutation.

**Scoring / unresolved checks:** `compatible`. Do not trim trailing zero coefficients or add a special zero coefficient for constants; both would alter observed behavior.

### HumanEval/87 — get_row

**Family:** indexed_filter_flattening. **Original triage:** manual_review. **Signature:** `get_row(lst, x)`.

**Observed / judgment:** Matching coordinates are returned in increasing row order and decreasing column order within each row, including ragged rows.

**Competing approaches:** Comprehension: Enumerate rows, traverse each row's column indices backwards and emit matching coordinate tuples in one nested comprehension. Loop: Use nested loops in the same row/column order and append each matching coordinate tuple.

**Equivalence and cost (inferred):** Both visit each element once, preserve required order and repeated matches, handle empty/ragged rows, and do not mutate data.

**Scoring / unresolved checks:** `compatible`. The scorer supports a single comprehension with multiple generators and multiple for-loops feeding one accumulator; designate the full coordinate construction.

### HumanEval/104 — unique_digits

**Family:** scalar_predicate_filtering. **Original triage:** manual_review. **Signature:** `unique_digits(x)`.

**Observed / judgment:** Positive integers are retained only when every digit is odd, then returned in ascending order.

**Competing approaches:** Comprehension: Sort the original integers first and retain values passing a shared all-odd-digits predicate. Loop: Use the same sorted input and append values passing that predicate.

**Equivalence and cost (inferred):** Both retain duplicate qualifying values, use the same positive-integer domain, and have O(n log n) sorting plus total digit-scan work. No input writes.

**Scoring / unresolved checks:** `compatible`. Sorting before filtering makes the final collection visible to the current scorer. This preserves output semantics on the admitted domain and does not impose a new order.

### HumanEval/117 — select_words

**Family:** scalar_predicate_filtering. **Original triage:** priority_review. **Signature:** `select_words(s, n)`.

**Observed / judgment:** Nonempty space-delimited words are retained when their count of non-vowel characters equals n.

**Competing approaches:** Comprehension: Use a shared consonant-count predicate and retain nonempty tokens with the required count. Loop: For each nonempty token, apply the same predicate and append matches.

**Equivalence and cost (inferred):** Both preserve word order and duplicates, uppercase vowels and literal-space splitting, in linear total character work. No input mutation.

**Scoring / unresolved checks:** `compatible`. The auxiliary isalpha domain permits more than ASCII letters; reproduce the canonical fixed vowel set rather than redefining Unicode consonants.

### HumanEval/129 — minPath

**Family:** conditional_sequence_construction. **Original triage:** manual_review. **Signature:** `minPath(grid, k)`.

**Observed / judgment:** The grid is a permutation containing one; canonical preprocessing finds its minimum neighbor and constructs an alternating minimum path of length k.

**Competing approaches:** Comprehension: After shared location/neighbor preprocessing, map each path position to one or the minimum neighbor by parity. Loop: Use the same preprocessing and append the same parity-selected value at every path position.

**Equivalence and cost (inferred):** Both have O(N squared plus k) work, preserve the grid and emit exactly k integer values. The grid contract makes every alternative starting value lexicographically larger.

**Scoring / unresolved checks:** `compatible`. The alternating-path argument is an inference to verify, not an oracle pass. This is a genuine variable-length path construction, though more algorithmic than preferred pilot mappings.

### HumanEval/152 — compare

**Family:** paired_mapping. **Original triage:** priority_review. **Signature:** `compare(game, guess)`.

**Observed / judgment:** Equal-length integer score and guess arrays map to absolute per-position errors.

**Competing approaches:** Comprehension: Map corresponding positions or zipped score/guess pairs to their absolute difference. Loop: Traverse the identical pairs and append the absolute difference.

**Equivalence and cost (inferred):** Both retain one integer per aligned pair, including zeros and repeated errors, with linear work and unchanged inputs. Equal lengths are explicit in the auxiliary contract.

**Scoring / unresolved checks:** `compatible`. Keep the two-array signature and positional alignment; no extra penalty or threshold may be introduced.

### HumanEval/163 — generate_integers

**Family:** bounded_range_filtering. **Original triage:** manual_review. **Signature:** `generate_integers(a, b)`.

**Observed / judgment:** The result contains even digits within the inclusive interval, after swapping reversed endpoints, with an upper cap of nine.

**Competing approaches:** Comprehension: Normalize endpoints, enumerate the capped range and retain even values. Loop: Use the same capped range and append even values.

**Equivalence and cost (inferred):** Both preserve ascending order, include endpoints when eligible, return empty above nine and handle reversed bounds. At most nine positive candidates are inspected.

**Scoring / unresolved checks:** `compatible`. Do not generalize digits to arbitrary even integers or include zero despite the positive-endpoint contract.

### Mbpp/8 — square_nums

**Family:** elementwise_mapping. **Original triage:** priority_review. **Signature:** `square_nums(nums)`.

**Observed / judgment:** Each numeric element contributes its square.

**Competing approaches:** Comprehension: Map each number using exponentiation by two. Loop: Append the same exponentiation result for each number.

**Equivalence and cost (inferred):** Both are linear in scalar operations and preserve order, duplicates and floating/integer behavior. Neither mutates input.

**Scoring / unresolved checks:** `compatible`. Keep exponentiation semantics and numeric types; redundant with other power mappings for pilot diversity.

### Mbpp/75 — find_tuples

**Family:** scalar_predicate_filtering. **Original triage:** manual_review. **Signature:** `find_tuples(test_list, K)`.

**Observed / judgment:** Whole tuples are retained only if every element is divisible by positive K.

**Competing approaches:** Comprehension: Filter tuples using the shared all-elements-divisible predicate. Loop: Apply the identical predicate and append qualifying tuples.

**Equivalence and cost (inferred):** Both retain tuple values, order and duplicates; empty tuples pass vacuously. Short-circuit predicate evaluation and positive-divisor semantics are identical.

**Scoring / unresolved checks:** `compatible`. The predicate's generator is not the list-style target. Keep the original uppercase K parameter.

### Mbpp/104 — sort_sublists

**Family:** nested_collection_mapping. **Original triage:** manual_review. **Signature:** `sort_sublists(input_list)`.

**Observed / judgment:** Each sublist is independently sorted; the outer list preserves sublist order. The contract allows a list or tuple outer input.

**Competing approaches:** Comprehension: Map each sublist through sorted to produce one fresh sorted list per row. Loop: Append sorted(row) for each sublist.

**Equivalence and cost (inferred):** Both preserve duplicates, outer order and string ordering, avoid mutating source rows, and cost the same sum of per-row sorting costs.

**Scoring / unresolved checks:** `compatible`. This is a substantive outer mapping, unlike sorting an entire output list. Nearly duplicates Mbpp/569 but has a broader outer-input domain.

### Mbpp/161 — remove_elements

**Family:** membership_filtering. **Original triage:** priority_review. **Signature:** `remove_elements(list1, list2)`.

**Observed / judgment:** Elements of the first list are removed when equal to an element of the second.

**Competing approaches:** Comprehension: Retain first-list elements not in the second list. Loop: Perform the same membership check and append nonmembers.

**Equivalence and cost (inferred):** Both preserve order and repeated surviving elements, support unhashable elements through list membership, and have the same O(n times m) comparison cost.

**Scoring / unresolved checks:** `compatible`. Do not optimize the second list into a set without proving equivalence over the unrestricted list element domain.

### Mbpp/265 — list_split

**Family:** indexed_slicing. **Original triage:** manual_review. **Signature:** `list_split(S, step)`.

**Observed / judgment:** Each output sublist contains one residue class of input indices modulo positive step; this is not record-field extraction.

**Competing approaches:** Comprehension: For each starting offset below step, produce the strided input slice. Loop: Append the same strided slice for each starting offset.

**Equivalence and cost (inferred):** Both preserve residue order, within-slice order and duplicates; step larger than input produces trailing empty sublists. Work is O(input length plus step).

**Scoring / unresolved checks:** `compatible`. The static family-unknown result missed a clear indexed transformation. Preserve the parameter name step even though the prose calls it n.

### Mbpp/272 — rear_extract

**Family:** record_extraction. **Original triage:** priority_review. **Signature:** `rear_extract(test_list)`.

**Observed / judgment:** The last field of each nonempty tuple becomes an output element.

**Competing approaches:** Comprehension: Map each tuple to its last element. Loop: Append each tuple's last element.

**Equivalence and cost (inferred):** Both preserve outer order, multiplicity and element types without copying or mutating the extracted values. Linear indexing work; nonempty tuples are required.

**Scoring / unresolved checks:** `compatible`. Closely redundant with other last-element/first-element extraction tasks; that is a diversity concern, not ineligibility.

### Mbpp/282 — sub_list

**Family:** paired_mapping. **Original triage:** manual_review. **Signature:** `sub_list(nums1, nums2)`.

**Observed / judgment:** Equal-length nonempty numeric lists are subtracted element by element.

**Competing approaches:** Comprehension: Map aligned pairs to left-minus-right. Loop: Traverse the same aligned pairs and append their difference.

**Equivalence and cost (inferred):** Both preserve pairing, ordering and numeric types at the same linear scalar cost, with no input writes.

**Scoring / unresolved checks:** `compatible`. Use the same zip alignment and arithmetic, rather than reducing or padding lists.

### Mbpp/390 — add_string

**Family:** format_mapping. **Original triage:** manual_review. **Signature:** `add_string(list_, string)`.

**Observed / judgment:** The supplied format string is applied independently to each list element.

**Competing approaches:** Comprehension: Map each element through that format string's format operation. Loop: Append the result of the same formatting operation for each element.

**Equivalence and cost (inferred):** Both preserve order, repeated values, formatting conversion rules and formatting exceptions. Cost is proportional to formatting work plus output size.

**Scoring / unresolved checks:** `compatible`. The contract leaves element types broad; do not substitute string concatenation or silently repair invalid format strings. Static call-allowlist uncertainty is not intrinsic incompatibility.

### Mbpp/412 — remove_odd

**Family:** parity_filtering. **Original triage:** priority_review. **Signature:** `remove_odd(l)`.

**Observed / judgment:** Odd integers are removed; even integers remain.

**Competing approaches:** Comprehension: Retain each number whose remainder modulo two is zero. Loop: Append each number satisfying the identical remainder test.

**Equivalence and cost (inferred):** Both preserve order and duplicate even values, including zero and negative evens, with linear cost and no mutation.

**Scoring / unresolved checks:** `compatible`. Duplicate eligibility with Mbpp/629 should not translate into two independent transformation families in a pilot.

### Mbpp/413 — extract_nth_element

**Family:** record_extraction. **Original triage:** priority_review. **Signature:** `extract_nth_element(list1, n)`.

**Observed / judgment:** The nth indexed item of each tuple forms the result.

**Competing approaches:** Comprehension: Map each tuple to its item at index n. Loop: Append that same indexed item from each tuple.

**Equivalence and cost (inferred):** Both preserve order and types and must use native indexing, including valid negative indices and IndexError for indices too negative. Linear indexing cost.

**Scoring / unresolved checks:** `compatible`. The auxiliary contract checks only n less than each tuple length, not the negative lower bound. Full oracle must check exception/domain handling; do not skip invalid tuples.

### Mbpp/424 — extract_rear

**Family:** record_extraction. **Original triage:** priority_review. **Signature:** `extract_rear(test_tuple)`.

**Observed / judgment:** The last character of each nonempty string in an input tuple is extracted.

**Competing approaches:** Comprehension: Map each string to its final character. Loop: Append each string's final character.

**Equivalence and cost (inferred):** Both preserve tuple iteration order, repeated characters and Unicode code-point indexing, returning a list without mutating input.

**Scoring / unresolved checks:** `compatible`. Keep the tuple/string input contract; this overlaps with last-field extraction but adds a string element type.

### Mbpp/426 — filter_oddnumbers

**Family:** parity_filtering. **Original triage:** priority_review. **Signature:** `filter_oddnumbers(nums)`.

**Observed / judgment:** Odd integers are retained, as established by the visible example.

**Competing approaches:** Comprehension: Retain values with remainder one modulo two. Loop: Append values satisfying the same predicate.

**Equivalence and cost (inferred):** Both retain ordering and duplicate odd numbers, including negative odd numbers under Python modulo semantics, with linear cost.

**Scoring / unresolved checks:** `compatible`. Near-duplicate of Mbpp/554; neither canonical style nor duplicated task count warrants extra pilot representation.

### Mbpp/436 — neg_nos

**Family:** sign_filtering. **Original triage:** manual_review. **Signature:** `neg_nos(list1)`.

**Observed / judgment:** Negative numeric elements are retained.

**Competing approaches:** Comprehension: Filter values by comparison less than zero. Loop: Append values passing the same comparison.

**Equivalence and cost (inferred):** Both preserve order, duplicates and scalar types, excluding zero and negative zero, with linear comparisons and no mutation.

**Scoring / unresolved checks:** `compatible`. This clear filter was missed by keyword family matching; validate unusual float comparisons on the original tests.

### Mbpp/447 — cube_nums

**Family:** elementwise_mapping. **Original triage:** priority_review. **Signature:** `cube_nums(nums)`.

**Observed / judgment:** Each numeric element contributes its cube.

**Competing approaches:** Comprehension: Map numbers using exponentiation by three. Loop: Append the same exponentiation result for every number.

**Equivalence and cost (inferred):** Both preserve signed and floating numeric behavior, ordering and duplicates at matching scalar cost, with no input mutation.

**Scoring / unresolved checks:** `compatible`. Strongly redundant with square and general-power mappings; preserve the exponentiation operation for numerical equivalence.

### Mbpp/450 — extract_string

**Family:** string_length_filtering. **Original triage:** priority_review. **Signature:** `extract_string(str1, l)`.

**Observed / judgment:** Strings of exactly the supplied length are retained.

**Competing approaches:** Comprehension: Filter strings by len(string) equal to l. Loop: Append strings whose length equals l.

**Equivalence and cost (inferred):** Both preserve order, duplicates and Unicode code-point length. Zero selects empty strings; negative lengths select none. Linear length checks.

**Scoring / unresolved checks:** `compatible`. Retain the original str1 and l signature and exact-equality predicate, not a minimum-length threshold.

### Mbpp/456 — reverse_string_list

**Family:** elementwise_mapping. **Original triage:** manual_review. **Signature:** `reverse_string_list(stringlist)`.

**Observed / judgment:** Each string is reversed independently, without reordering the list itself.

**Competing approaches:** Comprehension: Map strings to their reverse slice. Loop: Append the same reverse slice for each string.

**Equivalence and cost (inferred):** Both preserve outer order and duplicates and reverse code points rather than grapheme clusters. Work is linear in total character count.

**Scoring / unresolved checks:** `compatible`. Do not reverse the outer list or reinterpret Unicode grapheme boundaries; the source slicing semantics are the reference.

### Mbpp/460 — Extract

**Family:** record_extraction. **Original triage:** priority_review. **Signature:** `Extract(lst)`.

**Observed / judgment:** The first element of each nonempty sublist is returned.

**Competing approaches:** Comprehension: Map each sublist to its zeroth element. Loop: Append the zeroth element of each sublist.

**Equivalence and cost (inferred):** Both preserve order, duplicates and references to extracted objects without mutating rows. Linear indexing; nonempty rows are explicit.

**Scoring / unresolved checks:** `compatible`. Keep the capitalized Extract entry point; very close to the other record-extraction candidates.

### Mbpp/554 — Split

**Family:** parity_filtering. **Original triage:** manual_review. **Signature:** `Split(l)`.

**Observed / judgment:** Odd integers from a nonempty list are retained; canonical code uses filter.

**Competing approaches:** Comprehension: Retain integers with remainder one modulo two. Loop: Append the same qualifying integers.

**Equivalence and cost (inferred):** Both preserve order, duplicates and negative-odd modulo behavior with linear work and no input mutation.

**Scoring / unresolved checks:** `compatible`. Near-duplicate of Mbpp/426 with a nonempty input restriction; retain both review records, but avoid redundant pilot slots.

### Mbpp/568 — empty_list

**Family:** fresh_record_construction. **Original triage:** manual_review. **Signature:** `empty_list(length)`.

**Observed / judgment:** The task allocates N empty dictionaries; N is variable and dictionaries must be fresh independent objects to match the source's allocation behavior.

**Competing approaches:** Comprehension: Construct one new dictionary for each index in range(length). Loop: Append a newly created empty dictionary on every iteration.

**Equivalence and cost (inferred):** Both take O(N) space/time, return empty at zero and preserve independent mutable container identities. This is not a fixed list or shared-object repetition.

**Scoring / unresolved checks:** `compatible`. Check alias independence explicitly during reference QA; equality-only tests may not distinguish fresh dictionaries from repeating one dictionary.

### Mbpp/569 — sort_sublists

**Family:** nested_collection_mapping. **Original triage:** manual_review. **Signature:** `sort_sublists(list1)`.

**Observed / judgment:** Each string sublist is sorted independently; the canonical implementation uses map(sorted, ...).

**Competing approaches:** Comprehension: Map each row through sorted. Loop: Append sorted(row) for each row.

**Equivalence and cost (inferred):** Both preserve outer order and duplicates, allocate fresh result rows and leave input rows unchanged, with identical per-row sorting costs.

**Scoring / unresolved checks:** `compatible`. Essentially duplicates Mbpp/104; retain its nonempty outer-list restriction and do not count the pair as family diversity.

### Mbpp/578 — interleave_lists

**Family:** one_level_flattening. **Original triage:** manual_review. **Signature:** `interleave_lists(list1, list2, list3)`.

**Observed / judgment:** Aligned triples from three equal-length lists are interleaved into one output.

**Competing approaches:** Comprehension: Zip the lists and flatten each triple in order with two generators. Loop: Use nested loops over the same zipped triples and append each component.

**Equivalence and cost (inferred):** Both are O(total input size), preserve duplicates and exact interleave order, handle three empty lists and avoid mutation.

**Scoring / unresolved checks:** `compatible`. The auxiliary equal-length constraint is important; do not pad or extend unequal lists. This is a clear flattening target omitted by static keyword matching.

### Mbpp/618 — div_list

**Family:** paired_mapping. **Original triage:** manual_review. **Signature:** `div_list(nums1, nums2)`.

**Observed / judgment:** Equal-length nonempty numeric lists are divided element by element; denominators are nonzero in the auxiliary contract.

**Competing approaches:** Comprehension: Map aligned pairs to numerator divided by denominator. Loop: Append the same quotient for each pair.

**Equivalence and cost (inferred):** Both preserve true-division type/rounding, order and pairing with linear scalar work and unchanged inputs.

**Scoring / unresolved checks:** `compatible`. Retain ordinary division rather than floor division or safe-zero fallbacks; full-oracle floating behavior remains unverified.

### Mbpp/623 — nth_nums

**Family:** elementwise_mapping. **Original triage:** manual_review. **Signature:** `nth_nums(nums, n)`.

**Observed / judgment:** Each list value is raised to the supplied exponent; the auxiliary contract allows integer or float exponents.

**Competing approaches:** Comprehension: Map each value using the original exponentiation operation. Loop: Append the result of the same exponentiation for every value.

**Equivalence and cost (inferred):** Both preserve ordering, duplicates and native power behavior, including complex results or exceptions for particular numeric combinations. Scalar costs match.

**Scoring / unresolved checks:** `compatible`. Do not constrain n to integers or coerce results to float. Large powers and domain-edge errors need runtime checking, not silent normalization.

### Mbpp/629 — Split

**Family:** parity_filtering. **Original triage:** manual_review. **Signature:** `Split(l)`.

**Observed / judgment:** Even integers are retained.

**Competing approaches:** Comprehension: Filter numbers by zero remainder modulo two. Loop: Append numbers satisfying the same modulo predicate.

**Equivalence and cost (inferred):** Both preserve order and duplicates, negative evens and zero, at linear cost without mutation.

**Scoring / unresolved checks:** `compatible`. Near-duplicate of Mbpp/412; separate source identity does not add a new transformation family.

### Mbpp/728 — sum_list

**Family:** paired_mapping. **Original triage:** manual_review. **Signature:** `sum_list(lst1, lst2)`.

**Observed / judgment:** Corresponding elements in equal-length lists are added.

**Competing approaches:** Comprehension: Map aligned pairs through addition. Loop: Append the same addition result for every aligned pair.

**Equivalence and cost (inferred):** Both preserve pairing, order and duplicates, with matching addition costs and no input mutation. The auxiliary contract does not restrict element types, so retain native addition semantics.

**Scoring / unresolved checks:** `compatible`. Do not silently coerce elements to numbers; lists or strings may have meaningful addition behavior under the broad source domain.

### Mbpp/730 — consecutive_duplicates

**Family:** run_deduplication. **Original triage:** manual_review. **Signature:** `consecutive_duplicates(nums)`.

**Observed / judgment:** Each consecutive equality run contributes its first representative; separated equal values remain separate.

**Competing approaches:** Comprehension: Extract the key from each group yielded by groupby. Loop: Append the key from each group yielded by the identical groupby traversal.

**Equivalence and cost (inferred):** Both are linear, preserve run order and representatives, support unhashable elements, and avoid input mutation. Groupby state is consumed sequentially in the same way.

**Scoring / unresolved checks:** `compatible`. Use the same grouping primitive for both styles to preserve subtle identity/equality behavior; do not replace with a global set.

### Mbpp/749 — sort_numeric_strings

**Family:** ordered_numeric_mapping. **Original triage:** manual_review. **Signature:** `sort_numeric_strings(nums_str)`.

**Observed / judgment:** The visible example and source return integers, not the original numeric strings, in numerical order.

**Competing approaches:** Comprehension: Sort strings using int as a key, then map each sorted string through int. Loop: Use the same key-sorted strings and append the same integer conversions.

**Equivalence and cost (inferred):** Both preserve numeric ordering, duplicates and integer result types, with O(n log n) sorting plus conversion work. This avoids putting sorted around the scored output.

**Scoring / unresolved checks:** `compatible`. The auxiliary numeric-string check can admit text int rejects. Preserve conversion errors and validate the full oracle; do not repair numeric strings or return strings instead.

### Mbpp/754 — extract_index_list

**Family:** aligned_filtering. **Original triage:** manual_review. **Signature:** `extract_index_list(l1, l2, l3)`.

**Observed / judgment:** An element qualifies only when all three lists contain equal values at that same index.

**Competing approaches:** Comprehension: Filter zipped triples by chained equality and emit the first value. Loop: Use identical zipped triples and append the first value when chained equality succeeds.

**Equivalence and cost (inferred):** Both stop at the shortest list, preserve order and duplicate qualifying positions, and avoid set conversion or mutation. Linear comparisons.

**Scoring / unresolved checks:** `compatible`. Unequal lengths are not prohibited; zip truncation and returning the first-list object are part of the observed behavior to preserve.

### Mbpp/766 — pair_wise

**Family:** adjacent_pair_mapping. **Original triage:** manual_review. **Signature:** `pair_wise(l1)`.

**Observed / judgment:** Every consecutive pair is returned as a tuple; the output is not the original list copied.

**Competing approaches:** Comprehension: Map positions before the last to a tuple containing the current and next elements. Loop: Append that same adjacent-element tuple for each position.

**Equivalence and cost (inferred):** Both are linear, preserve order and duplicates, return empty for fewer than two elements and retain references to the original element objects.

**Scoring / unresolved checks:** `compatible`. A direct indexed pairing makes the transformation substantive without copying an already completed zip result. No list-style preference follows from the source's use of list(zip(...)).

### Mbpp/778 — pack_consecutive_duplicates

**Family:** run_grouping. **Original triage:** manual_review. **Signature:** `pack_consecutive_duplicates(list1)`.

**Observed / judgment:** Each consecutive equality run is materialized as a sublist, retaining separated repeats as different runs.

**Competing approaches:** Comprehension: For each groupby run, construct its list immediately within the outer comprehension. Loop: For each run, append its immediately materialized list.

**Equivalence and cost (inferred):** Both are linear, preserve run and element order and duplicates, and must consume a group before advancing the iterator. Input is not mutated.

**Scoring / unresolved checks:** `compatible`. Do not store lazy group iterators for later consumption. Designate the outer run-list construction; scalar/iterator internals are outside the style outcome.

### Mbpp/780 — find_combinations

**Family:** pairwise_record_mapping. **Original triage:** manual_review. **Signature:** `find_combinations(test_list)`.

**Observed / judgment:** Each distinct pair of input tuples produces a tuple of coordinate-wise sums in combinations order.

**Competing approaches:** Comprehension: Enumerate unordered input-position pairs and map each pair to a tuple of coordinate sums. Loop: Append the identical summed-coordinate tuple for each pair in the same combinations order.

**Equivalence and cost (inferred):** Both preserve output order, tuple types, duplicates from equal-valued inputs and equal-width behavior, with O(n squared times width) work. Empty tuples yield empty-tuple records.

**Scoring / unresolved checks:** `compatible`. Preserve floating addition order and combinatorial output size; validate sandbox resource use on full inputs. The final outer list construction is separate from scalar/tuple summation.

## Needs resolution (17)

**Complete ID list:** HumanEval/5, HumanEval/34, HumanEval/81, HumanEval/96, HumanEval/106, HumanEval/149, Mbpp/108, Mbpp/111, Mbpp/128, Mbpp/391, Mbpp/572, Mbpp/589, Mbpp/612, Mbpp/615, Mbpp/630, Mbpp/745, Mbpp/769.

### HumanEval/5 — intersperse

**Family:** interspersing. **Original triage:** manual_review. **Signature:** `intersperse(numbers: List[int], delimeter: int) -> List[int]`.

**Observed / judgment:** The output alternates original items and a delimiter, without a trailing delimiter; the canonical loop appends twice on most iterations.

**Competing approaches:** Comprehension: Enumerate input items and flatten either an item-delimiter pair or a final singleton; alternatively map output indices by parity. Loop: Append each item and append the delimiter only before the next item.

**Equivalence and cost (inferred):** Both can be linear and preserve order, duplicate values, empty input and the unchanged misspelled delimeter parameter. No input mutation is needed.

**Scoring / unresolved checks:** `compatible`. The conditional nested comprehension is more elaborate than the natural append loop. Human QA must decide whether it is idiomatic and comparably readable, without changing the interspersing problem.

### HumanEval/34 — unique

**Family:** sorted_deduplication. **Original triage:** manual_review. **Signature:** `unique(l: list)`.

**Observed / judgment:** The canonical output is sorted unique elements using a set followed by sorting; the visible contract gives no element-type bounds.

**Competing approaches:** Comprehension: Potentially sort the input and retain only the first member of each adjacent equal run. Loop: Apply the identical adjacent-run predicate over the sorted sequence and append representatives.

**Equivalence and cost (inferred):** Both could be O(n log n) and avoid an identity copy of sorted(set(input)). But adjacency equality may differ from set hashing/identity for unusual values, and it changes failures for unhashable input.

**Scoring / unresolved checks:** `compatible`. Resolve the effective element domain and representative/equality semantics before claiming unchanged behavior. Merely copying sorted unique values is not a substantive target.

### HumanEval/81 — numerical_letter_grade

**Family:** conditional_scalar_mapping. **Original triage:** manual_review. **Signature:** `numerical_letter_grade(grades)`.

**Observed / judgment:** One letter grade is computed per GPA through fixed ordered thresholds; the visible example names a different function.

**Competing approaches:** Comprehension: Use a shared scalar threshold helper and map it over grades. Loop: Append results from the identical threshold helper over grades.

**Equivalence and cost (inferred):** Both preserve threshold inclusivity, order and duplicates in linear time; the auxiliary domain is zero through four. Helper branches do not make the outer mapping ambiguous.

**Scoring / unresolved checks:** `compatible`. Correct the example entry-point mismatch only after human wording review. Shared scalar helpers are natural here; the static helper flag is not intrinsic unsuitability.

**Proposed wording only (not applied):**

> Original: grade_equation(

> Replacement: numerical_letter_grade(

Align the example name with the unchanged entry point.

### HumanEval/96 — count_up_to

**Family:** sieve_filtering. **Original triage:** manual_review. **Signature:** `count_up_to(n)`.

**Observed / judgment:** Examples and canonical code return all primes strictly below n; the prose also calls them the first n integers that are prime.

**Competing approaches:** Comprehension: Precompute shared primality flags with a sieve, then retain candidate indices below n with a true flag. Loop: Use the same completed sieve and append the flagged indices below n.

**Equivalence and cost (inferred):** Both can retain O(n log log n) sieve work and a linear output pass, return empty below three and preserve ascending order. Local sieve writes are not external effects.

**Scoring / unresolved checks:** `compatible`. Resolve contradictory first-n wording and approve the separated sieve/output region; do not weaken the task to a smaller prime range.

**Proposed wording only (not applied):**

> Original: an array of the first n     integers that are prime numbers and less than n.

> Replacement: a list of all prime integers strictly less than n.

Reconcile contradictory count wording with the examples and canonical sieve.

### HumanEval/106 — f

**Family:** conditional_numeric_mapping. **Original triage:** manual_review. **Signature:** `f(n)`.

**Observed / judgment:** Indices start at one: even indices produce factorials, odd indices triangular sums; canonical code instead maintains parity-specific recurrences.

**Competing approaches:** Comprehension: Map each index to math.factorial(index) or its integer triangular formula. Loop: Append the identical scalar expression for each index.

**Equivalence and cost (inferred):** The proposed pair has matching scalar costs and avoids reading the output accumulator, but repeated factorial evaluation can cost more than the canonical recurrence. Preserve integer types and zero-length output.

**Scoring / unresolved checks:** `compatible`. Measure sandbox practicality across the unchanged full inputs and obtain idiomaticity review; do not claim equal cost to the optimized canonical recurrence. Current scorer can recognize the proposed pair, unlike its canonical accumulator reads.

### HumanEval/149 — sorted_list_sum

**Family:** ordered_filtering. **Original triage:** manual_review. **Signature:** `sorted_list_sum(lst)`.

**Observed / judgment:** The task filters odd-length strings and sorts by length then alphabetically, but also says all words have the same length and uses a different example function name.

**Competing approaches:** Comprehension: Sort using the documented length/alphabetical ordering, then retain even-length strings. Loop: Traverse the same sorted strings and append those of even length.

**Equivalence and cost (inferred):** Both preserve duplicates and empty strings with O(n log n) ordering plus linear filtering. No input mutation is intrinsic.

**Scoring / unresolved checks:** `compatible`. Resolve the contradictory equal-length assumption and example names before finalizing visible wording. Both proposed final-output shapes are supported without changing scorer code.

**Proposed wording only (not applied):**

> Original: You may assume that all words will have the same length.

> Replacement: [delete this sentence]

Remove an assumption contradicted by the examples and auxiliary contract; human confirmation of intended domain required.

**Proposed wording only (not applied):**

> Original: list_sort(

> Replacement: sorted_list_sum(

Align both example calls with the unchanged entry point.

### Mbpp/108 — merge_sorted_list

**Family:** flattening_with_ordering. **Original triage:** manual_review. **Signature:** `merge_sorted_list(num1, num2, num3)`.

**Observed / judgment:** Three arbitrary numeric lists are merged with duplicates preserved and the final result globally sorted.

**Competing approaches:** Comprehension: Flatten the three lists with a two-generator comprehension, then sort the result. Loop: Use nested loops to append all elements, then apply the identical sort.

**Equivalence and cost (inferred):** Both have the same O(N log N) cost and preserve multiplicity and input immutability. Flattening is a coherent candidate region even though concatenation is also concise.

**Scoring / unresolved checks:** `requires_support`. The current scorer rejects sorted wrapping or post-construction sort mutations. Human QA must accept flattening as substantive and specify a region before the final ordering; this is not an automatic scientific exclusion.

### Mbpp/111 — common_in_nested_lists

**Family:** intersection_filtering. **Original triage:** manual_review. **Signature:** `common_in_nested_lists(nestedlist)`.

**Observed / judgment:** Canonical output materializes a set intersection; the visible example compares sets, and the auxiliary contract permits an empty outer list.

**Competing approaches:** Comprehension: Potentially filter distinct candidates from one list by membership in all remaining lists, once representative order is specified. Loop: Apply the same membership predicate and append in that same resolved candidate order.

**Equivalence and cost (inferred):** Set equality alone cannot establish output order or multiplicity. Empty outer input reaches an unbound set.intersection call, while hashability is otherwise implicit.

**Scoring / unresolved checks:** `conditional`. EvalPlus 0.3.1 compares this task's outputs as sets. Resolve deterministic order, duplicates and empty-input behavior without silently imposing a new contract.

### Mbpp/128 — long_words

**Family:** string_length_filtering. **Original triage:** manual_review. **Signature:** `long_words(n, s)`.

**Observed / judgment:** The prose says a list of words, but the signature, example and auxiliary contract use a string that is split on literal spaces.

**Competing approaches:** Comprehension: Filter literal-space-separated tokens by length greater than n. Loop: Append tokens exceeding n using identical tokenization.

**Equivalence and cost (inferred):** Both preserve token order and duplicates; n is nonnegative, so empty tokens are excluded. Tabs inside a token must not be silently treated as separators.

**Scoring / unresolved checks:** `compatible`. Resolve the input-type wording mismatch; a proposed clarification follows the existing signature and example rather than changing the function.

**Proposed wording only (not applied):**

> Original: from a given list of words.

> Replacement: from a string of words separated by spaces.

Describe the existing string parameter rather than introduce a list parameter.

### Mbpp/391 — convert_list_dictionary

**Family:** record_construction. **Original triage:** manual_review. **Signature:** `convert_list_dictionary(l1, l2, l3)`.

**Observed / judgment:** The prose names a nested dictionary, but the visible example and canonical code return a list of nested dictionaries, one per aligned triple.

**Competing approaches:** Comprehension: Map zipped triples to a fresh outer dictionary containing a fresh inner dictionary. Loop: Append the same nested dictionary construction for each triple.

**Equivalence and cost (inferred):** Both can preserve zipped truncation, record order, per-record key semantics and fresh container allocation with linear tuple count.

**Scoring / unresolved checks:** `compatible`. Resolve the list-versus-dictionary wording before acceptance. Do not merge records into one dictionary or alter unequal-length zip behavior.

**Proposed wording only (not applied):**

> Original: Write a function to convert more than one list to nested dictionary.

> Replacement: Write a function that pairs three lists positionally and returns a list of nested dictionaries, one per triple, stopping at the shortest list.

Conditional clarification following the existing example and canonical behavior; human confirmation required.

### Mbpp/572 — two_unique_nums

**Family:** frequency_filtering. **Original triage:** priority_review. **Signature:** `two_unique_nums(nums)`.

**Observed / judgment:** Despite wording about removing duplicates from a number of lists, the example and source keep only values occurring exactly once in one list.

**Competing approaches:** Comprehension: Retain each number whose occurrence count in the original list equals one. Loop: Apply the identical occurrence-count predicate and append singleton values.

**Equivalence and cost (inferred):** Both preserve order and numeric equality behavior, at O(n squared) using list.count. Do not substitute set deduplication, which retains one copy of repeated values.

**Scoring / unresolved checks:** `compatible`. Resolve single-list input and remove-all-repetitions wording; prefer a cleanly stated equivalent task for an initial pilot. HumanEval/26 is a close alternative.

**Proposed wording only (not applied):**

> Original: Write a python function to remove duplicate numbers from a given number of lists.

> Replacement: Write a function to remove all values occurring more than once in a single list, preserving the order of remaining values.

Clarify singleton filtering rather than ordinary retain-one deduplication.

### Mbpp/589 — perfect_squares

**Family:** numeric_range_filtering. **Original triage:** manual_review. **Signature:** `perfect_squares(a, b)`.

**Observed / judgment:** The canonical implementation normalizes bounds, clips negatives, and tests square roots for an integral floating result.

**Competing approaches:** Comprehension: Use the same bound preprocessing and retain integers passing the sqrt-is-integer predicate. Loop: Append integers passing the identical predicate over the identical inclusive range.

**Equivalence and cost (inferred):** Both can match the source arithmetic and cost, but floating square roots can disagree with exact mathematical squarehood for sufficiently large integers. The domain has no explicit magnitude bound.

**Scoring / unresolved checks:** `compatible`. Resolve numeric semantics and sandbox range size; replacing the predicate with exact isqrt without oracle/human review may change observed behavior.

### Mbpp/612 — merge

**Family:** matrix_transposition. **Original triage:** manual_review. **Signature:** `merge(lst)`.

**Observed / judgment:** The contract returns two output columns from nonempty two-element rows; canonical code transposes with zip and materializes columns.

**Competing approaches:** Comprehension: Possible paired form: build each column by indexed row extraction in an outer two-column comprehension. Loop: For each of the two column indices, build its list with an inner append loop and append that column to the outer result.

**Equivalence and cost (inferred):** Both preserve column order, row order, duplicates and object references with linear element work. The contract is coherent despite two output lists.

**Scoring / unresolved checks:** `requires_region_decision`. Human QA must choose whether the target is inner element extraction or outer column construction. The current scorer follows returned constructors but does not symmetrically trace an inner loop accumulator; that limitation is not substantive ineligibility.

### Mbpp/615 — average_tuple

**Family:** column_reduction_mapping. **Original triage:** manual_review. **Signature:** `average_tuple(nums)`.

**Observed / judgment:** The prose says average for each tuple, but the example and source compute averages of columns formed by zip over the tuples.

**Competing approaches:** Comprehension: Once column semantics are confirmed, map each zipped column to sum divided by its length. Loop: Append the same column mean using the same sum order and division.

**Equivalence and cost (inferred):** Both preserve column order and floating arithmetic and stop at the shortest row; empty outer input or an empty row produces no columns. Same total reduction work.

**Scoring / unresolved checks:** `compatible`. Resolve row-versus-column meaning and ragged-row truncation explicitly; do not silently reinterpret the problem as row averages.

**Proposed wording only (not applied):**

> Original: Write a function which takes a tuple of tuples and returns the average value for each tuple as a list.

> Replacement: Write a function which takes a tuple of numeric tuples and returns a list of column averages, stopping at the shortest tuple.

Conditional proposal only if humans confirm the example/canonical column interpretation over the conflicting row wording.

### Mbpp/630 — get_coordinates

**Family:** coordinate_product_construction. **Original triage:** manual_review. **Signature:** `get_coordinates(test_tup)`.

**Observed / judgment:** The example includes the original coordinate as well as neighbors; canonical recursion supports any tuple dimension and includes the empty vector for dimension zero.

**Competing approaches:** Comprehension: Potentially enumerate offset products and map each tuple of coordinates into a list. Loop: Append an equivalent coordinate list for each product tuple in the same lexicographic product order.

**Equivalence and cost (inferred):** Both can cost O(d times 3 to the d), with natural standard-library product preprocessing, but the contract does not restrict dimension. Canonical recursion is incidental to that formulation.

**Scoring / unresolved checks:** `compatible`. Resolve 'adjacent' including center, arbitrary-dimensional scope, and exponential sandbox practicality. Do not restrict the task to two dimensions to simplify the comprehension.

**Proposed wording only (not applied):**

> Original: Write a function to extract all the adjacent coordinates of the given coordinate tuple.

> Replacement: Return all coordinate lists formed by independently offsetting each coordinate by -1, 0, or 1, including the original coordinate, in product order.

Clarify the source interpretation without narrowing dimension; human confirmation and resource review required.

### Mbpp/745 — divisible_by_digits

**Family:** digit_predicate_filtering. **Original triage:** manual_review. **Signature:** `divisible_by_digits(startnum, endnum)`.

**Observed / judgment:** Numbers in an inclusive range are retained if every digit is nonzero and divides the number; source converts each character of str(n) to int.

**Competing approaches:** Comprehension: Filter the inclusive range through a shared digit predicate that checks zero before divisibility. Loop: Use the identical predicate and append passing integers.

**Equivalence and cost (inferred):** Both can preserve short-circuiting and O(range size times digit count) work. But the auxiliary contract allows negative bounds and canonical conversion of a minus sign raises.

**Scoring / unresolved checks:** `compatible`. Resolve the admitted sign domain and oracle exception policy; a positive-only edit would change the contract and is not proposed as an automatic repair.

### Mbpp/769 — Diff

**Family:** symmetric_difference_filtering. **Original triage:** manual_review. **Signature:** `Diff(li1, li2)`.

**Observed / judgment:** Canonical output concatenates both directional set differences; the prose merely says difference and its example does not distinguish symmetric from directional semantics.

**Competing approaches:** Comprehension: After resolving symmetric semantics and ordering, filter distinct values from each side by absence from the other. Loop: Apply the same two membership passes and append in the agreed order.

**Equivalence and cost (inferred):** Both could preserve unique membership, but canonical set iteration is not a stable ordered-list contract. Set-based preprocessing also requires hashable items.

**Scoring / unresolved checks:** `conditional`. EvalPlus compares outputs as sets, so a pass would not establish ordering or multiplicity. Resolve semantics and determinism before selecting a scoring region or proposing any sorted-order edit.

## Unsuitable under the current criteria (16)

**Complete ID list:** HumanEval/9, HumanEval/25, HumanEval/88, HumanEval/101, HumanEval/116, HumanEval/120, HumanEval/123, HumanEval/130, HumanEval/145, Mbpp/12, Mbpp/255, Mbpp/297, Mbpp/475, Mbpp/565, Mbpp/603, Mbpp/635.

### HumanEval/9 — rolling_max

**Family:** prefix_state. **Original triage:** manual_review. **Signature:** `rolling_max(numbers: List[int]) -> List[int]`.

**Observed / judgment:** The output is the running maximum; the canonical comprehension recomputes a maximum over every prefix.

**Competing approaches:** Comprehension: Repeatedly taking a sliced-prefix maximum is possible but quadratic; using a running accumulator requires stateful expressions or copying an already computed output. Loop: Track the running maximum while appending one result per input item in linear time.

**Equivalence and cost (inferred):** The natural loop exploits sequential state. Repeating prefix scans in both styles can equalize cost only by choosing an unnecessarily expensive loop algorithm.

**Scoring / unresolved checks:** `not_applicable`. The substantive obstacle is the stateful prefix dependency and poor natural algorithm comparability, not a scorer failure or the presence of a canonical comprehension.

### HumanEval/25 — factorize

**Family:** factorization_state. **Original triage:** manual_review. **Signature:** `factorize(n: int) -> List[int]`.

**Observed / judgment:** Prime factors are emitted with multiplicity while the canonical algorithm repeatedly divides the remaining integer.

**Competing approaches:** Comprehension: Independent divisibility filtering does not recover multiplicity naturally; hiding factorization in a helper leaves only identity materialization. Loop: Repeatedly divide the residual number and append each discovered factor, then append any remaining prime.

**Equivalence and cost (inferred):** The natural output production is inseparable from changing residual state and repeated factor extraction. A comprehension would require a different factorization strategy or hide the target work.

**Scoring / unresolved checks:** `not_applicable`. Unsuitable for the primary local transformation contrast; no restriction to distinct factors or small fixed primes is permitted.

### HumanEval/88 — sort_array

**Family:** sorting_only. **Original triage:** manual_review. **Signature:** `sort_array(array)`.

**Observed / judgment:** The whole array is sorted in an order chosen by endpoint-sum parity, without modifying the input.

**Competing approaches:** Comprehension: A comprehension after sorted would only copy an already completed result. Loop: An append loop after sorted would likewise copy it, while implementing sorting manually changes the algorithmic task.

**Equivalence and cost (inferred):** The only genuine transformation is sorting; a forced list-construction contrast would add redundant materialization rather than measure a natural competing transformation.

**Scoring / unresolved checks:** `not_applicable`. The empty-array guard and direction choice do not supply a separate meaningful list-building region.

### HumanEval/101 — words_string

**Family:** tokenization_materialization. **Original triage:** manual_review. **Signature:** `words_string(s)`.

**Observed / judgment:** Canonical whitespace splitting already removes empty tokens; its final nonempty-word comprehension performs redundant filtering.

**Competing approaches:** Comprehension: Using the existing split result yields only an identity copy; introducing alternate tokenization solely to create a filter would change the algorithmic presentation. Loop: Appending every already split word is likewise redundant, while manual token scanning introduces a stateful algorithm.

**Equivalence and cost (inferred):** The substantive operation is tokenization, with comma and arbitrary-whitespace semantics. There is no independently meaningful canonical post-split transformation.

**Scoring / unresolved checks:** `not_applicable`. A canonical comprehension is insufficient evidence of suitability; do not make tokenization more awkward just to obtain a style contrast.

### HumanEval/116 — sort_array

**Family:** sorting_only. **Original triage:** manual_review. **Signature:** `sort_array(arr)`.

**Observed / judgment:** The whole array is sorted by binary-one count and numeric tie-break; one example uses negative values despite a nonnegative domain.

**Competing approaches:** Comprehension: A comprehension that copies a key-sorted array adds no substantive transformation. Loop: An append copy after the same sort is equally redundant.

**Equivalence and cost (inferred):** The principal operation is a scalar-key sort, not mapping keys to a new output collection. Manual sorting would change the intended algorithmic contrast.

**Scoring / unresolved checks:** `not_applicable`. The phrase 'It must be implemented like this:' precedes behavioral examples, not a list-style instruction. Domain/example inconsistencies do not rescue the sorting-only task.

**Potential wording cue:** Full visible statement inspected: no explicit preference for list comprehensions or list-building for-loops found. Canonical syntax and auxiliary contract code were treated as non-instructional evidence, not as visible style requirements. The phrase 'It must be implemented like this:' introduces behavior examples; it does not prescribe the target list syntax. The negative-input example conflicts with the stated nonnegative domain.

### HumanEval/120 — maximum

**Family:** sorting_selection_only. **Original triage:** manual_review. **Signature:** `maximum(arr, k)`.

**Observed / judgment:** The result is the sorted largest k elements, including duplicates; k may be zero.

**Competing approaches:** Comprehension: A comprehension over a completed top-k slice is identity materialization; index-copying the suffix adds unnecessary syntax. Loop: Appending that same suffix also adds copying rather than a substantive independent transformation.

**Equivalence and cost (inferred):** Natural solutions use sorting/selection and slicing. A distinct list-building target would be forced around the operation that already produces the answer.

**Scoring / unresolved checks:** `not_applicable`. The zero-k special case and output ordering remain relevant oracle requirements, but do not create a natural comprehension/loop contrast.

### HumanEval/123 — get_odd_collatz

**Family:** collatz_state. **Original triage:** manual_review. **Signature:** `get_odd_collatz(n)`.

**Observed / judgment:** The next Collatz value depends on the current one, stopping only upon reaching one; retained odd values are then sorted.

**Competing approaches:** Comprehension: A direct comprehension needs a stateful generator/helper or a precomputed whole trajectory before filtering. Loop: Advance the trajectory in a while-loop, collecting odd values, then sort.

**Equivalence and cost (inferred):** Sequence generation dominates and has data-dependent termination. The positive-integer contract supplies no general proven termination/time bound.

**Scoring / unresolved checks:** `not_applicable`. Not suitable for a bounded ordinary local transformation experiment; no restricted input domain or trajectory cutoff is introduced.

### HumanEval/130 — tri

**Family:** recurrence_state. **Original triage:** manual_review. **Signature:** `tri(n)`.

**Observed / judgment:** Odd sequence terms depend on preceding terms; canonical code reads its growing output list and has special early nonempty returns.

**Competing approaches:** Comprehension: A comprehension would require a stateful helper, memoized recurrence, or a newly derived closed form rather than a direct local transformation. Loop: Append terms using previous output values and preserve the special initial sequence.

**Equivalence and cost (inferred):** The construction is intrinsically sequential; hiding a finished recurrence behind a comprehension reduces the latter to materialization. Canonical division also mixes numeric types.

**Scoring / unresolved checks:** `not_applicable`. Scorer limitations for accumulator reads and early returns are secondary to the substantive recurrence dependency.

### HumanEval/145 — order_by_points

**Family:** sorting_only. **Original triage:** manual_review. **Signature:** `order_by_points(nums)`.

**Observed / judgment:** The result reorders original integers by a signed digit-weight helper, retaining original order at tied weights.

**Competing approaches:** Comprehension: Copying the helper-key-sorted list would be redundant. Loop: Appending that already sorted list would be redundant too.

**Equivalence and cost (inferred):** The only genuine collection operation is sorting; the scalar digit-weight computation is not a list-producing target. Negative-sign treatment and stability cannot be changed.

**Scoring / unresolved checks:** `not_applicable`. Do not move the style outcome to the helper's digit list: that would score an intermediate unrelated to the returned collection.

### Mbpp/12 — sort_matrix

**Family:** sorting_only. **Original triage:** manual_review. **Signature:** `sort_matrix(M)`.

**Observed / judgment:** Matrix rows are reordered by their sums, not transformed individually.

**Competing approaches:** Comprehension: A comprehension after the row sort would merely copy its result. Loop: An append loop after the same sort would likewise be redundant.

**Equivalence and cost (inferred):** The natural operation is stable key sorting; hand-implementing sorting adds algorithmic differences. Preserve equal-sum stability and row objects if ever revisited.

**Scoring / unresolved checks:** `not_applicable`. The task returns a list but lacks a separate substantive output-mapping/filtering target.

### Mbpp/255 — combinations_colors

**Family:** combinatorial_materialization. **Original triage:** manual_review. **Signature:** `combinations_colors(l, n)`.

**Observed / judgment:** The task enumerates combinations with replacement for arbitrary n; the canonical implementation materializes the standard iterator.

**Competing approaches:** Comprehension: Comprehending already generated combination tuples is identity copying; fixed-depth nested comprehensions cannot implement arbitrary n. Loop: Appending tuples from the same complete combinatorial iterator is also redundant.

**Equivalence and cost (inferred):** The substantive work is arbitrary-arity combination generation, requiring a combinatorial algorithm/helper. Output size is combinatorial.

**Scoring / unresolved checks:** `not_applicable`. No fixed-n restriction or different combination semantics may be introduced to create a target transformation.

### Mbpp/297 — flatten_list

**Family:** recursive_flattening. **Original triage:** manual_review. **Signature:** `flatten_list(list1)`.

**Observed / judgment:** The contract admits arbitrary recursively nested lists of numbers, not just one level.

**Competing approaches:** Comprehension: A comprehension would need recursive self-calls and conditional singleton wrapping or an external flattening iterator. Loop: The natural loop recursively extends results for list children and appends scalar leaves.

**Equivalence and cost (inferred):** Arbitrary recursive traversal is substantive and cannot be replaced by one-level flattening under the unchanged domain.

**Scoring / unresolved checks:** `not_applicable`. Fails the primary study's ordinary one-level transformation criterion; recursion depth and stack/resource limits remain additional concerns.

### Mbpp/475 — sort_counter

**Family:** sorting_only. **Original triage:** manual_review. **Signature:** `sort_counter(dict1)`.

**Observed / judgment:** Dictionary items are returned as value-descending pairs, retaining stable order for ties.

**Competing approaches:** Comprehension: A comprehension that copies sorted dictionary items adds no meaningful transformation. Loop: An append loop over those same items also only copies the answer.

**Equivalence and cost (inferred):** The principal operation is ordering an existing item iterator. Explicit key-value tuple reconstruction would be cosmetic rather than a substantive separate transformation.

**Scoring / unresolved checks:** `not_applicable`. The terse prose does not state descending order, though the example does; neither clarification nor a scorer extension would make the desired contrast natural.

### Mbpp/565 — split

**Family:** identity_materialization. **Original triage:** manual_review. **Signature:** `split(word)`.

**Observed / judgment:** The result is simply a list of the string's existing characters; list(word) already expresses the full task.

**Competing approaches:** Comprehension: An identity comprehension over characters is possible but introduces no substantive transformation or predicate. Loop: An append loop just materializes the same iteration.

**Equivalence and cost (inferred):** Both could match code-point order, but the contrast would measure redundant spelling of a constructor rather than a meaningful local transformation.

**Scoring / unresolved checks:** `not_applicable`. Agent judgment under the substantive-transformation criterion, not a claim that the two styles are impossible or that constructor syntax is scientifically forbidden.

### Mbpp/603 — get_ludic

**Family:** iterative_sieve_state. **Original triage:** manual_review. **Signature:** `get_ludic(n)`.

**Observed / judgment:** Ludic-number generation repeatedly removes elements from a changing list using previously discovered elements as step sizes.

**Competing approaches:** Comprehension: A comprehension cannot naturally encode the whole evolving elimination process without stateful helpers or copying the finished sequence. Loop: Repeatedly mutate the working list according to the elimination schedule.

**Equivalence and cost (inferred):** The substantive task is sequential elimination, not a local predicate applied independently to a fixed input. The 'lucid' spelling is also inconsistent with the function name.

**Scoring / unresolved checks:** `not_applicable`. Do not select the initial range-materialization loop as the style target; it is unrelated to the difficult output transformation.

### Mbpp/635 — heap_sort

**Family:** effectful_sorting. **Original triage:** manual_review. **Signature:** `heap_sort(iterable)`.

**Observed / judgment:** Canonical heapification and repeated heappop consume the input list while returning sorted values, although the statement only asks to sort.

**Competing approaches:** Comprehension: Its comprehension repeatedly invokes a mutating heap operation; a pure sorted-copy comprehension would instead be redundant. Loop: A heap-pop append loop can reproduce the same destructive state changes.

**Equivalence and cost (inferred):** The available natural contrast depends on stateful destructive extraction; a pure sorting formulation leaves no separate substantive mapping/filter target.

**Scoring / unresolved checks:** `not_applicable`. Canonical input consumption is not explicitly required by the prose, so it is a source-behavior conflict as well as a confound. Do not infer external mutation is part of the intended contract merely from the source.

**Potential wording cue:** Full visible statement inspected: no explicit preference for list comprehensions or list-building for-loops found. Canonical syntax and auxiliary contract code were treated as non-instructional evidence, not as visible style requirements. The unchanged entry-point name heap_sort cues an algorithm, but does not demand either list syntax. The heap behavior in the implementation is not a visible instruction.

## Remaining verification

Before any acceptance: independently author both complete reference implementations, preserve the original signature and unmodified base/plus oracle, execute only in the verified isolated Docker sandbox, compare exact output types/order/duplicates/effects/exceptions, inspect numerical edge cases, validate the selected scoring region on mixed and alternative forms, and obtain independent human QA. Reference correctness, determinism and sandbox practicality are not established by this review.

No task is human-reviewed, accepted or frozen. Proposed edits, family assignments and the pilot suggestion list remain provisional. The original 542-record candidate manifest and screening packet are preserved.
