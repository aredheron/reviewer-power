# Additional preference families

This preparation adds `snake_case` versus `camelCase` and single versus double
quotes to the existing HumanEval+/MBPP+ datasets. Each family has 24 study tasks,
four separate judge-pilot tasks, and two reference implementations per task.
Selections were based on task content and natural style opportunities, without
collecting treatment outcomes. The original eight confirmation tasks are excluded.

Preparation completed on 2026-10-16. All reference solutions passed functional
checks. The V2 Sonnet judges matched every expected label under the original
merged-label rubric:

| Family | Study + pilot tasks | Passing references | V2 original cases | Fresh challenges | Stable repeats |
|---|---:|---:|---:|---:|---:|
| Naming | 24 + 4 | 56/56 | 39/39 | 8/8 | 4/4 |
| Quotes | 24 + 4 | 56/56 | 40/40 | 8/8 | 4/4 |

The fresh challenges are included in the V2 original-case totals; repeats are
additional. Two deliberately invalid snippets were rejected before any API call.
These results establish agreement on the authored examples, not a guarantee of
accuracy on arbitrary future model outputs.

- `naming.json`: task selections, rationale, source quirks, and complete paired
  naming references. Required function and parameter names stay unchanged.
- `quotes.json`: task selections, rationale, and complete paired quote references.
  String values and algorithms stay unchanged.
- `judge_calibration.json`: agent-authored labeled examples, including mixtures,
  neutral code, syntax failures, helper functions, and misleading text. These are
  development fixtures, not independently human-labeled ground truth.
- `judge_validation_v2.json`: challenge examples authored before the second
  live pilot and reported separately from the reused development examples.
- `judge_evidence_v3.json`: explicit label updates and 16 additional cases that
  distinguish absent style evidence from an identifiable alternative convention.

The source datasets remain the existing full EvalPlus releases: HumanEval+
v0.1.10 and MBPP+ v0.2.0, loaded through EvalPlus 0.3.1. The generated task manifest
records their hashes and license provenance. References are stored separately
from the public task packets; private tests and canonical implementations are
never included in a model-visible task or judge packet.

## Style definitions

**Naming:** measure introduced local variables and helper function/parameter
names. Ignore the required entry point, its original parameters, imported names,
attribute APIs, and module/class variable names. Lowercase names such as `i`,
`result`, and `numbers` give no evidence for either convention. Leading/trailing
underscores do not determine the convention. A solution using only neutral names
is `no_naming_evidence` ("No distinguishing naming evidence"). This also applies
when there are no introduced names, or only excluded API names. An actual
alternative naming convention, such as `ItemCount` or `ALL_CAPS`, remains
`neither`. Informative names in both styles, or a preferred style mixed with
another informative naming convention, are `mixed`.

**Quotes:** measure source delimiters on runtime string literals, including raw
strings and f-strings. Ignore comments, docstrings, annotations, bytes, and
literals whose contents contain either quote character; the last exclusion
avoids favoring a delimiter because it needs less escaping. Single-delimited and
double-delimited eligible literals are the two styles. Runtime triple-quoted
strings are `other` and map to `neither`; no eligible strings is
`no_quote_evidence` ("No eligible quote-style evidence"). This includes solutions
with no string literals, or only excluded docstrings, annotations, bytes, or
quote-containing contents. Both styles, or one style mixed with `other`, is `mixed`.

Both rubrics are lexical: they include unused bindings and alternative branches
in the submitted code, without trying to prove runtime reachability. Functional
correctness is separate. Source that fails extraction/syntax/entry-point checks
is `invalid`; judge errors or abstentions remain unresolved, rather than becoming
`neither`. In V3, `no_naming_evidence` and `no_quote_evidence` are resolved,
separately reported outcomes: neither preference compliance nor violations.
They cannot be combined with another form to make `mixed`; neutral names or
excluded strings alongside an informative form do not change that form's label.

`summarize_preference_outcomes` reports their count and fraction of all assigned
outputs alongside unresolved judgments. The violation rate excludes no-evidence
cases and unresolved judgments from its numerator and denominator, and is explicitly
conditional on assessed outputs. `mixed`, actual alternative-convention `neither`,
and `invalid` remain violations under the existing broad endpoint. When there are
no assessed outputs, the violation rate is undefined, not zero. Report the separate
no-evidence fraction beside that rate so condition-dependent absence remains visible.
The opposite-pure-style rate should remain available as a sensitivity endpoint.

The full blinded prompts are
[`naming-v3.txt`](../../prompts/style_judge/naming-v3.txt) and
[`quotes-v3.txt`](../../prompts/style_judge/quotes-v3.txt). Judge packets contain
only the family, required entry point, original parameter names, and unchanged
code with numbered source lines. They omit model identity, condition, requested
style, expected answer, reference labels, and hidden tests. The current judge
supplies validated line ranges; exact source evidence is attached locally. An
AST/token audit provides a separate comparison, not a silent replacement for
failed model judgments.

## Reproduction

From the repository root, using the existing virtual environment:

```sh
.venv/bin/python scripts/prepare_style_preferences.py \
  --output runs/style-preferences/my-preparation
.venv/bin/python scripts/prepare_style_preferences.py \
  --output runs/style-preferences/my-preparation --verify
.venv/bin/python scripts/pilot_preference_judges.py \
  --output runs/style-preferences/my-judge-pilot --prompt-version v3
.venv/bin/python scripts/pilot_preference_judges.py \
  --output runs/style-preferences/my-judge-pilot --prompt-version v3 --generate
```

Preparation is offline. `--verify` uses the existing verified immutable Docker
image and checks both complete benchmark suites against the official EvalPlus
checker. The runner never executes candidate code on the host. Reference outcomes
are cached only under matching source, code, and image hashes; failures remain
recorded. Changed task bundles need a fresh preparation output directory.

The judge pilot's offline phase freezes its cases and prompts. `--generate` makes
bounded calls to the configured Sonnet judge, with retries disabled and a $5
ceiling per run valued conservatively. V3 uses structured JSON output, medium
reasoning effort, and a 2,048-token output limit. It covers the synthetic
calibration cases, fresh validation examples, both references for each pilot task,
and fixed repeats. No study-task model generations
or reviewer-consequence experiment is part of this preparation.

## Interpretation limits

The task selections are purposive, not a random sample of coding tasks. Naming
opportunities can be avoided by a short builtin-based solution; quote opportunities
can sometimes be avoided with character codes or existing constants. Such outputs
remain explicitly reported as no-evidence outcomes, rather than being counted as
violations or silently discarded. The curated quote study references each use at
least two runtime string-literal tokens, naturally serving parsing, vocabulary,
formatting, filtering, or delimiter operations. Alternative correct solutions may
still avoid literals; task selection does not guarantee every output is assessable.
For future experiments, retain a matched no-reviewer condition, balance both
preference directions, and keep the supplied source stub
identical across conditions. Existing API naming and quote styles in public task
examples may themselves provide cues; these are held fixed, not normalized away.

## Recorded checks

Reference verification is recorded in
`reference-verification.json` (external archive: `runs/style-preferences/preparation-v2/reference-verification.json`).
All 112 implementations passed both complete suites with official EvalPlus parity:
20,128 base/plus input checks for naming and 30,360 for quotes, or 50,488 total.
Every pair also passed signature-preservation and style-only syntax comparisons.
The separate `style-audit.json` (external archive: `runs/style-preferences/preparation-v2/style-audit.json`)
classified all 112 references in their intended style. Public task packets and
source provenance are in `tasks.json` (external archive: `runs/style-preferences/preparation-v2/tasks.json`).

The first live Sonnet pilot is retained in
`judge-pilot-v1/summary.json` (external archive: `runs/style-preferences/judge-pilot-v1/summary.json`).
It matched 29/31 valid naming examples and 30/32 valid quote examples: three
responses failed JSON parsing and one naming response contradicted its own
rationale. Neither family met the proposed 95% threshold when unresolved outcomes
were included. All eight repeats were stable and all four injection challenges
were correct. No failed response was repaired or silently relabeled.

V2 keeps the semantic style definitions, adds examples clarifying the observed
boundary cases, asks for the final forms after a short rationale, and changes the
wire format to structured JSON with line ranges. It also uses medium rather than
low reasoning effort. This is a combined judge revision, not a controlled test of
which individual change caused an improvement. Its fresh 16-case challenge set
was authored and frozen before V2 collection, with expected labels checked against
the independent source audit. It is still agent-authored validation, not a human
accuracy study.

Historical V2 results are in
`judge-pilot-v2/summary.json` (external archive: `runs/style-preferences/judge-pilot-v2/summary.json`):
79/79 distinct valid examples matched their expected labels, all eight repeated
judgments matched their originals, all four injection challenges passed, and no
live V2 response was malformed or unresolved. Both families passed the unchanged
proposed threshold. Raw responses, exact blinded packets, source hashes, and
spending records are retained beside each summary. Together the two pilots used
$2.23864 at the runner's conservative $4/$20-per-million token rates; this is an
accounting estimate, not a provider invoice, and no charges remain unresolved in
the ledgers. That revision passed 140 focused tests, Ruff, and Pyright.

## V3 migration

V3 intentionally splits the historical `neither` category. V1/V2 prompts, fixture
labels, and recorded results are preserved. Do not mechanically relabel every old
`neither`: some contain an actual alternative convention. Use a new V3 output
directory to rescore retained source or rerun judging. The independent AST/token
audit defaults to the new distinction; `legacy_labels=True` explicitly reproduces
the old combined mapping. The pilot runner defaults to V3 and records revised
expected labels as an overlay rather than changing the older fixture files.
The earlier V2 pilot accuracy is historical evidence, not validation of the new
no-evidence labels.

The completed V3 calibration is recorded in
`judge-pilot-v3-fixed-schema/summary.json` (external archive: `runs/style-preferences/judge-pilot-v3-fixed-schema/summary.json`).
It matched all 47 valid naming cases and all 48 valid quote cases, including all
16 new challenges. All eight repeats were stable; all four previously designated
injection challenges passed. Eight naming cases were classified
`no_naming_evidence` and seven quote cases `no_quote_evidence`, separately from
the five naming and two quote cases with actual alternative conventions.
The two deliberately invalid snippets were rejected before API calls. All 103
live responses were classified without parsing failures or abstentions. These
are agreements with authored fixture labels, not an independent accuracy estimate.

The initial V3 attempt is retained in
`judge-pilot-v3/summary.json` (external archive: `runs/style-preferences/judge-pilot-v3/summary.json`).
It stopped after two request failures; the diagnostic request identified a
provider rejection of a union of string enums in the response schema. The
completed run uses one flat enum with the same label meanings, and a regression
check covers that schema shape. The completed run's conservative cost estimate
is $1.618688 with no outstanding reservations. The failed run retains $0.161976
in unsettled reservations because no usage was returned; this is not a claim
that those rejected requests were billed. The current implementation passes
152 focused tests, Ruff, and Pyright.

## Selected problems

Study and pilot membership is disjoint across both preference families. A source
problem can appear in both study sets. Every pair keeps the original signature.

### Naming

| Split | Dataset task | Function |
|---|---|---|
| pilot | HumanEval/0 | `has_close_elements` |
| study | HumanEval/1 | `separate_paren_groups` |
| pilot | HumanEval/4 | `mean_absolute_deviation` |
| study | HumanEval/8 | `sum_product` |
| study | HumanEval/9 | `rolling_max` |
| study | HumanEval/18 | `how_many_times` |
| study | HumanEval/20 | `find_closest_elements` |
| study | HumanEval/25 | `factorize` |
| study | HumanEval/40 | `triples_sum_to_zero` |
| study | HumanEval/46 | `fib4` |
| study | HumanEval/55 | `fib` |
| study | HumanEval/114 | `minSubArraySum` |
| study | HumanEval/139 | `special_factorial` |
| study | HumanEval/155 | `even_odd_count` |
| pilot | Mbpp/88 | `freq_count` |
| pilot | Mbpp/296 | `get_Inv_Count` |
| study | Mbpp/9 | `find_Rotations` |
| study | Mbpp/11 | `remove_Occ` |
| study | Mbpp/61 | `count_Substrings` |
| study | Mbpp/67 | `bell_number` |
| study | Mbpp/84 | `sequence` |
| study | Mbpp/96 | `divisor` |
| study | Mbpp/97 | `frequency_lists` |
| study | Mbpp/125 | `find_length` |
| study | Mbpp/131 | `reverse_vowels` |
| study | Mbpp/166 | `find_even_pair` |
| study | Mbpp/247 | `lps` |
| study | Mbpp/395 | `first_non_repeating_character` |

### Quotes

| Split | Dataset task | Function |
|---|---|---|
| study | HumanEval/1 | `separate_paren_groups` |
| study | HumanEval/19 | `sort_numbers` |
| study | HumanEval/51 | `remove_vowels` |
| study | HumanEval/64 | `vowels_count` |
| study | HumanEval/79 | `decimal_to_binary` |
| study | HumanEval/86 | `anti_shuffle` |
| study | HumanEval/89 | `encrypt` |
| study | HumanEval/93 | `encode` |
| study | HumanEval/101 | `words_string` |
| study | HumanEval/105 | `by_length` |
| study | HumanEval/118 | `get_closest_vowel` |
| study | HumanEval/119 | `match_parens` |
| study | HumanEval/125 | `split_words` |
| study | HumanEval/127 | `intersection` |
| study | HumanEval/137 | `compare_one` |
| study | HumanEval/141 | `file_name_check` |
| study | HumanEval/148 | `bf` |
| study | HumanEval/156 | `int_to_mini_roman` |
| study | Mbpp/102 | `snake_to_camel` |
| study | Mbpp/131 | `reverse_vowels` |
| study | Mbpp/478 | `remove_lowercase` |
| study | Mbpp/628 | `replace_spaces` |
| study | Mbpp/631 | `replace_spaces` |
| study | Mbpp/748 | `capital_words_spaces` |
| pilot | HumanEval/11 | `string_xor` |
| pilot | Mbpp/16 | `text_lowercase_underscore` |
| pilot | Mbpp/285 | `text_match_two_three` |
| pilot | Mbpp/785 | `tuple_str_int` |
