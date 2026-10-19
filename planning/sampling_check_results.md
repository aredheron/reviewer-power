# GPT-OSS temperature-0.7 pilot sampling check

Run: `runs/sampling-check/gpt-oss-t07-20261010/`. This is a diagnostic on
HumanEval/21 and Mbpp/778, both already exposed pilot control tasks. No reserved
main task was used for inference. Sonnet generation was not repeated.

## Design and results

GPT-OSS-120B used temperature 0.7, top-p 1, low reasoning effort and a 4,096-token
completion limit. Temperature was the only change to its previous generation
settings. Each request had a fresh one-message context, with no tools or actual
selection/training updates. Inspect, SDK and HTTP connection retries were disabled.

| Check | Result |
|---|---:|
| Comprehension probes, strict JSON | 24/24 correct |
| Logged-only / selection / RL probes | 8/8 each |
| Coding responses, full EvalPlus base and plus suites | 72/72 pass |
| Normal completion | 96/96 GPT-OSS responses |
| Corrected Sonnet style judgments with valid citations | 72/72 |
| Repeated cells with more than one exact code string | 24/24 |
| Repeated cells with more than one style label | 5/24 |

The coding batch was 2 tasks × 3 consequences × 4 preference combinations × 3
repetitions. Repetitions had distinct assignment IDs and identical prompts within
each cell. The prospective comprehension gate (21/24 overall, at least 7/8 in
each consequence) passed before coding began. There were no generation replacements.

Corrected semantic labels: 48 loop-only, 24 comprehension-only, no mixed, neither,
invalid or unresolved judgments. Original AST scoring labeled only 43/72 responses
(35 loop-only, 8 comprehension-only); all 43 agree with the corrected judge. The
29 remaining AST outcomes are preserved as unresolved, not silently relabeled in
the historical scorer.

## Judge-input correction

The initial judging pass supplied source task prose but omitted EvalPlus's separate
`contract` field. HumanEval/21's contract explicitly requires
`max(numbers) > min(numbers)`. Without that constraint, Sonnet counted zero-range
fallback branches as reachable and assigned 13 mixed labels. This was a pipeline
input error, not evidence of a sampling-induced correctness failure.

The runner now supplies both original prose and independent source preconditions;
a regression test checks their inclusion. All 72 responses were judged again
uniformly with the same judge model, settings and rubric. The 13 mixed labels
became loop-only; all other labels stayed unchanged. Original judgments and the
original runner bytes remain archived. There were 96 GPT-OSS generation calls and
144 Sonnet judging calls across the original and corrected passes, with no automatic
retries and no new coding responses during correction.

This diagnostic supports using temperature 0.7 for the main run: the tested tasks
retain comprehension and correctness, and repeated sampling produces some style
variation. It does not establish population-wide reliability, judge accuracy on
all future outputs, or equivalence to the temperature-zero pilot's effect sizes.
Exact code variation includes inconsequential textual/implementation differences.

## Evidence and verification

- `summary-with-source-contract.json`: operative diagnostic summary; the original
  `summary.json` remains historical.
- `judge-with-source-contract/amendment.json`: complete input constraints and
  correction provenance, linked to the original frozen manifest.
- `solution-label-review.md`: all 72 solutions, original/corrected labels and
  corrected judge rationales, with task/condition/repetition linkage.
- `final-verification.json`: all 96 native Inspect prompts/outputs matched to
  frozen assignments/raw responses; all 72 corrected packets and exact citations
  revalidated; original source hashes verified against archived runner bytes.
- 569 tests passed; nine Docker integration tests passed separately; Pyright,
  Ruff lint and formatting passed. A targeted scan found no configured credential
  or pod-host values in the checked committable files and run artifacts, including
  two decoded native logs. This is not a universal proof of absence of secrets.

The main collection has not started. Its remaining preparation is separate from
this completed sampling diagnostic.
