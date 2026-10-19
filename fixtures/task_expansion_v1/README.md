# Additional loop/comprehension tasks: audit v1

Audit date: 2026-10-17. This separate bank recommends **14 additional tasks**
without semantic changes, after agent suitability review and verification of both
reference styles. It does **not** reach the requested 64 new tasks. Another ten
tested tasks remain held for wording, domain, scope, or idiomaticity decisions;
one tested task duplicates another candidate and is not an independent addition.

All **50 implementations for 25 candidate tasks passed the full pinned base and
plus suites**, with matching official EvalPlus checker results. Passing tests is
necessary but does not resolve the eleven holds. No model inference calls or
experimental generations were made. Historical task banks, outcomes, dashboard,
and the eight locked confirmation tasks are unchanged.

## Scope and evidence

The cached pool contains 164 HumanEval+ and 378 MBPP+ tasks. All 51 previously
accepted tasks were excluded from expansion, including pilot, exploratory and
locked tasks. `review.json` covers the remaining **491 source IDs exactly once**:

| Disposition | Tasks |
| --- | ---: |
| Shortlisted for two reference implementations | 25 |
| Non-list output contract | 422 |
| Sorting/materialization without a substantive target transformation | 16 |
| Slicing/concatenation | 6 |
| Intrinsic state | 5 |
| Return an existing object | 4 |
| Intrinsic recursion or state | 3 |
| Fixed record, unresolved set semantics, or tokenization/conversion | 6 |
| Other contract/target/oracle concerns | 4 |
| Total | 491 |

Coverage means contract triage of the remaining pool, followed by full
source/auxiliary-contract review for the shortlist and borderline collection
tasks; it does not mean an equally deep semantic audit of every scalar task.
The exclusion log is reviewable and is not a proof that no other candidate could
ever be made suitable. Tasks were selected without inspecting experimental model
outcomes. Canonical implementations informed semantics, not preference-effect
selection.

The references were authored and reviewed by the assistant. Independent researcher
QA, experiment wrappers/splits and model/judge pilots remain separate steps before
experimental admission. The `human_review` fields deliberately do not claim human
approval. No prompt corrections or input-domain restrictions have been applied.

## Recommended without semantic changes

Each row has a tested explicit-append loop and list-comprehension implementation
in `references.json`; `transformation` names the intended scoring region.

| Source ID | Entry point | Target transformation / caveat |
| --- | --- | --- |
| HumanEval/70 | `strange_sort_list` | Alternate low/high indices after shared sorting |
| HumanEval/100 | `make_a_pile` | Arithmetic mapping of pile levels |
| HumanEval/105 | `by_length` | Filter and map sorted digits to names |
| HumanEval/106 | `f` | Conditional factorial/triangular-number mapping; see stateful alternatives below |
| HumanEval/113 | `odd_count` | Map strings to formatted odd-digit counts |
| Mbpp/7 | `find_char_long` | Filter regex word matches by length |
| Mbpp/108 | `merge_sorted_list` | Flatten input lists before common sorting |
| Mbpp/251 | `insert_element` | Flatten element/item pairs |
| Mbpp/308 | `large_product` | Construct pairwise products before sorting and selection |
| Mbpp/563 | `extract_values` | Map quoted-string matches to captures |
| Mbpp/589 | `perfect_squares` | Map integer roots to squares in the requested interval |
| Mbpp/610 | `remove_kth_element` | Filter by one-based position |
| Mbpp/612 | `merge` | Construct columns from two-element rows |
| Mbpp/632 | `move_zero` | Stable nonzero filtering before a literal zero suffix |

These are new to the original loop/comprehension task bank, not necessarily to
every study: HumanEval/105 also appears in the quote-style fixtures. None of the
25 appears in the naming fixtures. This exposure comparison uses task IDs only,
not behavioral results.

Valid builtin, slicing or other implementations must remain valid alternatives;
the tasks do not require one of the two reference styles. In particular,
`re.findall` can solve the regex tasks without an explicit list-building loop or
comprehension. Such outputs must not be forced into either style label.

HumanEval/106 has natural stateful implementations as well as the independent
mapping references. A loop carrying factorial state is not automatically a
violation of the instruction against loops whose sole purpose is list appending.
Both references use the same factorial calls; neither is claimed to match the
work of the optimized upstream recurrence. Mbpp/589 uses exact integer square
roots; finite-suite success is not a proof of equivalence to floating canonical
behavior on arbitrarily large integers.

## Held despite successful reference tests

The proposed wording below is a review suggestion, not an applied edit.

| Source ID | Hold / concrete decision |
| --- | --- |
| HumanEval/5 | `intersperse`: judge whether indexed alternating comprehension is sufficiently natural compared with the append loop |
| HumanEval/81 | Replace example function name `grade_equation` with `numerical_letter_grade` |
| HumanEval/96 | Replace contradictory “first n primes” opening with “Return a list of all prime integers strictly less than the non-negative integer n”; review shared sieve preprocessing scope |
| HumanEval/149 | Remove the contradictory equal-length assumption and correct example function name to `sorted_list_sum` |
| Mbpp/128 | Clarify that the input is a string of space-separated words, not a list |
| Mbpp/391 | Clarify the list-of-nested-dictionaries result and shortest-input termination |
| Mbpp/572 | Clarify retaining values occurring exactly once in one input list; near-duplicate of existing HumanEval/26, so not an independent semantic family |
| Mbpp/615 | Clarify column means across row tuples, stopping at the shortest row; prose currently suggests row means |
| Mbpp/630 | Clarify arbitrary dimension, center inclusion and lexicographic order; exponential output and scope require review; no silent 2D restriction |
| Mbpp/745 | Resolve negative-endpoint domain ambiguity: the source contract permits it but the canonical solution attempts `int('-')`; no domain restriction applied |
| Mbpp/725 | Duplicates the quoted-capture transformation of Mbpp/563; retain evidence but do not count as an independent new task |

Different source IDs are not necessarily independent semantic tasks. Preserve the
recorded family/duplicate relationships in any future selection and sensitivity
analysis; collecting repetitions or renamed variants cannot substitute for new
task diversity.

## Style-scoring limitation

The historical `returned_list_v1` AST audit reports `region_unsupported` on both
styles of Mbpp/108, Mbpp/308 and Mbpp/632 because their constructions feed later
sorting, slicing or concatenation. Those failures are preserved in the report.
The historical scorer is unchanged. A separate reference-only AST check confirms
the explicit target function contains the intended construction, and semantic
inspection identifies its role. That structural check is **not** an experimental
output classifier and must not replace the existing judge.

Before using these tasks in experiments, calibrate the intended blinded judge on
the designated transformations and alternative outputs. These three are unsuitable
for unmodified automatic admission through the legacy AST-only path.

## Verification and reproduction

`runs/task-bank-expansion/20261017-v1/` contains the static audit, run bindings,
50 immutable outcome files and `summary.json`. Across the 25 pairs there are
7,332 source test inputs, each tested in both styles: 14,664 candidate/input
checks, in addition to official-checker parity verification. The 14 recommended
pairs account for 4,175 source test inputs, or 8,350 candidate/input checks.
Tests establish finite-suite agreement, not a formal proof for every possible
input.

The pinned sandbox image is
`sha256:90cff4efebd8c1cd1e1ca07896171091e998c58d902f039fc43870530c91d006`.
Nine sandbox checks passed before benchmark code execution. All benchmark and
reference execution took place through the existing isolated Docker oracle.
`preserved-bindings.json` records 78 unchanged historical artifacts and sources;
the run also binds this bank, audit script, oracle, dataset manifest and sandbox.
`verification.json` binds the saved reports and reference outcomes.

From the repository root, inspect saved results without executing task code:

```bash
uv run python scripts/audit_task_expansion.py summarize \
  --run-dir runs/task-bank-expansion/20261017-v1
uv run pytest tests/test_task_expansion.py -q
uv run pyright
uv run ruff check scripts/audit_task_expansion.py tests/test_task_expansion.py
```

For a separate future validation, use a new directory under
`runs/task-bank-expansion/`, run `audit`, then `execute` with `--sandbox-report`
pointing to a current successful nine-test JUnit report. Writes are exclusive:
existing validation evidence is never replaced.
