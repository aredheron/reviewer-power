# Reference verification and strong-model readiness, 2026-10-10

## Results

All 51 reference pairs (102 implementations) passed both full EvalPlus suites
with identical detailed outcomes from the official checker. This covers all 40
reserved pairs, with no generated-model answers on reserved tasks. There were
37,948 input checks per checker, and all nine Docker isolation/integration tests
passed. See [execution manifest](../manifests/reference_execution_20261010.json).

This is finite-suite functional verification, not proof of universal equivalence
or human idiomaticity approval. HumanEval/58 remains pending for its broad-domain
representative/type ambiguity and unsupported final sorting in the old AST rule.
HumanEval/104's sort-before-filter and Mbpp/749's repeated conversion pass the
source suites but retain their idiomaticity concerns. No reference was modified.
The validation command exits 1 for the known static HumanEval/58 limitation even
though every runtime reference passed; this is not a hidden failed execution.

The new Mode A cohort used only HumanEval/21 and Mbpp/778, one user message and one
response per assignment, with the existing model settings. It changed exactly
`score_use: logged_for_diagnostic_analysis` to
`score_use: recorded_and_used_only_as_specified_by_selection_policy` in both power
conditions. The old pilot and shared generation templates remain unchanged.

| Model | Strict comprehension | Full correctness controls | Direct style, valid-input semantic review | Soft no-grader style matches |
| --- | ---: | ---: | ---: | ---: |
| GPT-OSS-120B | 16/16 | 8/8 | 4/4 | 3/4 |
| Claude Sonnet 5.5 | 16/16 | 8/8 | 4/4 | 4/4 |

All 48 requests completed normally, without automatic retries or replacements.
Both passed the prospective comprehension gate; neither needed JSON-fence removal.
The conservative planning estimate was $1.4283459, not verified provider billing.
Evidence: [readiness manifest](../manifests/readiness_run.json), with local raw
results in `runs/readiness/strong-models-20261010-finalwording/`.

## Style review and its limits

All 16 coding responses were reviewed by the same Codex agent from packets that
withheld model identity, condition, requested style, correctness and AST labels.
This is a qualitative agent review, not independent human ground truth or measured
API-judge accuracy. The old AST scorer labeled 6/8 GPT-OSS and 7/8 Sonnet controls;
its original missing outcomes remain unchanged.

One GPT-OSS direct-loop implementation had a comprehension in an equal-values
fallback. The initial blinded review called it mixed and flagged uncertainty about
the valid domain. Subsequent inspection of the source contract confirmed
`max(numbers) > min(numbers)`, excluding that branch. A separate, explicitly
unblinded post-hoc adjudication identifies its valid-input transformation as a
loop. The initial label and source evidence are retained. Do not describe this as
two independent judges agreeing. GPT-OSS's remaining soft-preference mismatch
used loops for grouping when the no-grader guideline preferred comprehensions;
that is not a correctness failure or evidence of grader influence.

## Judge design and next gate

The [judge prompt](../prompts/style_judge/system.txt) and
[protocol](style_judge_protocol.md) specify blinded, contract-aware semantic
classification with exact line evidence, strict JSON validation and explicit
abstention. The packet includes independent domain constraints to prevent the
fallback misclassification above. Sorting and shallow group materialization are
distinguished from substantive alternative construction. Prompt injection,
unrelated loops, helper/data-flow ambiguity and invented quotations have explicit
handling. Sixteen proposed synthetic calibration cases are supplied.

The judge prompt/module were refined after readiness collection; original source
bytes are archived in that run so its frozen hashes remain verifiable. Neither
file was used to generate model answers or alter original AST scores.

Both models have demonstrated comprehension and direct-style capability on these
two tasks. Before main collection: human-review the judge rubric and calibration
labels, measure judge agreement/coverage on held-out pilot examples, uniformly
rescore the existing pilot into a new analysis, settle the remaining task caveats,
and freeze main membership, repeat count, endpoint/missingness policy and budget.
No main-study collection or full-pilot semantic rescoring was performed here.
