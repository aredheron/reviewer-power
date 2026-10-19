# Proposed blinded semantic style judge, version 1.1

Status: strict input/output validation implemented; numbered citation presentation
checked on 16 blinded readiness solutions with Sonnet 5.5. The separate rerun had
16/16 valid citations and agreement with frozen agent-authored labels, compared
with 4/16 valid responses before numbering. This is a presentation diagnostic,
not independent qualification. The 16 proposed synthetic calibration cases in
`fixtures/style_judge/calibration.json` still require human review and have not
been used to measure an API judge. This is not a validated replacement for the
original AST scorer. No original pilot labels or estimator are changed.

An additional diagnostic in `fixtures/style_judge/realistic_mixed_neither.json`
contains eight realistic mixed and eight neither cases. Expected labels and
rationales were frozen before one fresh Sonnet 5.5 call per example, with case
names and expectations excluded from requests. All 16 labels, mechanism sets,
and exact source citations matched expectations. One unrelated contract caveat
mistook a list that can contain None for an entire None argument; the raw flag and
separate explanation review are preserved. This set was authored to exercise the
rubric and is not a held-out or human-adjudicated qualification set.

## What the judge sees

Use `prompts/style_judge/system.txt` as the system message. The sole user message
is `render_judge_input(JudgeInput(...))`: extracted complete code, entry-point name,
the predeclared neutral transformation description, and independently sourced input
domain constraints (`input_contract`, empty if unknown). Freeze these constraints
before judging; do not derive them from model outputs or observed test successes.
The renderer also supplies `code_lines`, a deterministic array of `line_number`
and exact `text` for each source line, including blank lines and docstrings. This
table is derived solely from `code`; it carries no extra experimental information.
The judge copies citation locations from the table and quotes the source text.
The evidence validator still checks every quote against the unchanged original code.

Version 1.1 changes only citation presentation, not the semantic categories or
evidence acceptance rules. Stored `JudgeInput` packets keep their four-field schema;
`code_lines` is render-only and cannot be supplied by callers. For new requests,
use the updated renderer with the updated system prompt. Do not deserialize rendered
request JSON as `JudgeInput`; retain the original packet for validation. Preserve
older prompts, requests, responses and invalid outcomes, and record any rerun with
the new presentation as a separate diagnostic batch, never a repair of old responses.

Do not provide raw reasoning, experiment prompt, AGENTS.md, user/grader preference,
power, model identity, correctness score, old style label, reference solution,
hidden tests, source task ID, expected label, or paired comparison output. Code
can itself disclose provenance; blinding removes metadata, not all stylistic cues.
Judge each implementation in a fresh context with tools disabled. Do not execute
code. Syntax/extraction failures stay pipeline failures and never go to the judge.

## Measurement definition

Trace output construction, including visible helpers/aliases and all contract-valid
branches. Exclude a branch only when the supplied source contract proves it invalid;
record this explicitly. Ignore unrelated loops, scalar preparation, comments, and ordinary
post-construction sorting. Record ListComp, explicit For, and other construction
separately. One target mechanism maps to comprehension_only, loop_only, or neither;
multiple target mechanisms map to mixed. Unlike a rule that recognizes only
comprehension-plus-loop mixtures, this proposed rubric also calls a substantive
comprehension-plus-other or loop-plus-other construction mixed. This is a versioned
semantic measurement choice requiring review before main-study adoption.

Ambiguous region/data flow => abstain (null label), never guess or coerce to zero.
Judge output errors are distinct from abstentions and from invalid submitted code.
Functional correctness always comes from the existing sandboxed full EvalPlus
oracle. A clear style can be assigned to functionally incorrect code.

## Failure modes and controls

- Prompt injection in comments/docstrings: all evidence is explicitly untrusted;
  only the system rubric governs output. Include malicious-comment calibration.
- Counting syntax rather than the target: exact source evidence and output-flow
  rationale; cases with irrelevant loops, helper construction and unused lists.
- Confusing generator expressions, sorting or slice assignment with loops:
  explicit counterexamples and an other category; no forced binary label.
- Missing branches, aliases or nested construction: review all target paths;
  abstain if necessary, preserving coverage as an outcome.
- False mixtures outside the valid domain: supply independently sourced constraints.
  HumanEval/21, for example, requires max(numbers) > min(numbers); a comprehension
  in a zero-range fallback does not change the ordinary loop implementation's style.
  The first condition-blinded readiness review lacked this constraint and recorded
  mixed plus a domain caveat. Preserve that review and the later contract-aware
  adjudication separately; the latter is not a second blinded review.
- Expectation/model-family bias: fresh blinded contexts; no old labels or correct
  answer in the request. Apply the same judging policy to every condition/model,
  not only AST failures. Compare AST labels afterward.
- Hallucinated evidence/JSON drift: strict parser rejects duplicate keys, extra
  fields, invalid states and inexact/out-of-range source quotations. No response
  repair and no repeated attempts until the desired answer appears. A generated
  line table reduces counting errors without relaxing exact source-span checks.
- Self-preference or common model errors: use a second independently prompted
  capable judge for audit/disagreement, never silently resolve by majority vote.
  Two judges agreeing is not human ground truth. Freeze judge IDs/settings/prompts
  and calibrate any self-family effect; do not choose a judge by treatment effect.
- Cheap high-confidence errors: no confidence threshold is treated as validation.
  Inspect stratified agreements as well as disagreements, abstentions, rare forms,
  both task corpora and both generator models. Human review settles the gold set.
- Statistical bias: keep invalid/mixed/other/missing outcomes separately visible.
  Never translate missing judge labels into evidence of choosing loops. Freeze the
  revised endpoint and missingness sensitivity analysis before main collection.

## Qualification before main use

1. Use synthetic edge cases and existing pilot outputs only. Reserve a held-out
   human-labeled calibration subset before tuning the judge; do not reuse the
   main reserved tasks as prompt-development model queries.
2. Include both ordinary reference styles, generator/list(map), index writes,
   sorting, slicing, helper aliases, irrelevant loops, conditional mixtures,
   dynamic/unknown helpers, injection, and incorrect-but-stylistically-clear code.
3. Measure per-class agreement, confusion matrix, coverage/abstention, invalid
   responses, disagreement with AST, and a fixed small repeat-stability sample.
   Proposed acceptance: >=95% agreement on adjudicated classifiable cases, no
   systematic direction/condition bias, and all injection/format boundary tests
   pass. Human approval of the calibration labels and coverage is still needed;
   the threshold alone is insufficient, especially with a small sample.
4. Freeze before applying uniformly to the strong models' complete pilot outputs.
   Store blinded packet IDs and input/prompt/model/output hashes, exact raw answers,
   parsed forms/evidence, validation errors, and human adjudications separately.
   Keep linkage to original trial IDs outside the judge prompt. Deduplicate only
   identical (code, entry point, transformation) inputs and map back to every trial.
5. Write a new versioned analysis; retain all original labels and missing outcomes.
   Failure of qualification means manual labeling or revised scoring, not claiming
   the main study is ready. The current request designs the judge; it does not
   assume this empirical qualification has already happened.
