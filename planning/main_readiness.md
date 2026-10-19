# Main collection readiness

**Main collection started with user authorization on 2026-10-10.**

Live counts and operational state are recorded in
`runs/main/main-32-v1/operations/status.json`; batch events and launch provenance
are in the same directory. Two independent model lanes alternate bounded
generation and judging batches. Started requests are never automatically repeated.
New batches pause on provider/scoring infrastructure errors, missing usage records,
or a conservative recorded-usage estimate reaching $250. This assistant-selected
operational threshold is not a user-specified or provider-enforced billing cap;
already-running batches can finish beyond it. Creating `operations/STOP` prevents
further batches after current work finishes. Ambiguous judgments remain available
for blinded adjudication. The eight confirmation tasks remain untouched.
After two judge responses included an undeclared evidence field, the user authorized
resumption with individual judge validation errors deferred until the end. The
`resume-v2` operational controller counts normally completed responses with local
JSON/schema/evidence validation errors as unresolved, without pausing collection.
Provider failures and the other pause checks remain blocking. Original judge
responses/results, the first launch record, and the original controller are retained;
no requests are retried and no frozen scientific code or prompts are changed.
Following one generation timeout and one judge timeout, the user also authorized
continuing through small amounts of request errors/timeouts. The `resume-v3`
controller advances past isolated generation, scoring, and judge failures, retaining
all started requests and missing results for later review without retries. It pauses
new batches when either stage of either model has five failed/incomplete attempts
among its latest 48 starts. Functional test failures are not infrastructure errors.
Session/source-integrity checks, explicit STOP, and the spending guard still apply.
The earlier controllers and launch records are preserved. A finished collection
means all assignments were attempted, not that every request returned a response.
After 17 Together server errors in one batch caused another pause, the user
explicitly requested resumption. The next unstarted GPT-OSS assignment returned a
response and completed scoring. `resume-v4` resets the operational recent-error
window at launch, while preserving cumulative error counts and all prior records.
The same five-in-48 safeguard applies to new starts; prior requests are not retried.
The original `analysis-v1` queue stopped on the later pause. The provider-amended
analysis described below replaces that queue and retains the same missingness rule.

## Fireworks continuation readiness (2026-10-11 UTC)

At the provider cutoff, GPT-OSS had 2,354 started requests (2,307 saved/scored
responses and 47 server errors); Sonnet had 2,112 started requests (2,111
saved/scored responses and one timeout). The user authorized a Fireworks pilot and
resumption with Fireworks conditional on success. The prospective pilot gates
passed: **24/24 strict comprehension, 72/72 full correctness, 72/72 valid blinded
judgments**, with all 96 generations normally completed. No confirmation task was
used. Public provenance is in `manifests/fireworks_provider_amendment.json`.

The continuation freezes the **2,254 originally unstarted GPT-OSS assignments**
under `runs/main/main-32-v1/fireworks-continuation/`. Previous Together attempts,
including failures, are retained. Sonnet continues its original cohort. Prompts,
sampling, sandbox, judge and original scientific source bindings are unchanged;
the separate new source/profile manifest records the provider change.

The `resume-v6-fireworks` operational controller counts original plus continuation
records for GPT-OSS, excludes the provider pilot, and retains the five-in-48
failure safeguard and no-retry policy. The earlier controller's archived bytes
match its launch hash. A new completion waiter runs `run_fireworks.py --phase
analyze` into `analysis-provider-amended-v1/`, with row-level provider provenance
and separate descriptive conflict counts. Provider differs by collection period;
do not interpret pooled GPT-OSS results as a single-provider effect. Missing
outcomes still withhold primary inference.

Verification after the additive runner change: **585 tests passed**; nine Docker
integration tests were skipped in this unit run (the original passing Docker
preflight remains bound and the 72 new responses were actually scored in that
sandbox). Ruff and Pyright passed, including the run-local controller. Offline
controller checks covered combined counts, failure accounting and provider routing.
Known configured credentials were absent from checked committable files and the
Git index; `.env` remains ignored. `.env.example` contains only an empty
`FIREWORKS_API_KEY` placeholder.

Frozen cohort: `runs/main/main-32-v1/manifest.json`. Public summary: `manifests/main_preparation.json`.

## Frozen design

- 32 exploratory tasks: 12 HumanEval and 20 MBPP.
- Eight source tasks locked for confirmation: three HumanEval and five MBPP.
- GPT-OSS-120B at temperature 0.7, top-p 1; Sonnet 5.5 provider-managed sampling.
- Three consequences × four preference combinations × 12 repetitions per task/model.
- 9,216 coding requests; at most 9,216 separate Sonnet style judgments.
- One user message and one response, no tools or actual training/selection.

## Verification

- All 40 source-linked reference pairs retain passing full-suite/parity evidence. The user approved task suitability.
- Frozen prompts and task/source/settings hashes checked; 384 cells per model, each with 12 distinct repetition IDs and identical within-cell prompts.
- No source-task overlap with the pilot; no confirmation assignments generated.
- Complete EvalPlus input constraints are included in blinded judge packets.
- Mock end-to-end generation, scoring, judging, bounded resumption, interrupted judge handling and blinded adjudication passed.
- Analysis rejects duplicate/incomplete factorial assignments, retains missing outcomes, implements task bootstrap and Holm adjustment, and preserves functional failures in the behavioral denominator.
- Empty-cohort analysis verified all 9,216 outcomes remain missing and produces no primary p-values. Its `preflight-empty-not-results/` directory is a pipeline check, not experiment results.
- 580 tests passed. Nine Docker integration tests passed separately. Pyright: zero errors. Ruff lint and formatting passed.
- Targeted scan: no known configured credential/pod-host values found in checked committable files and main artifacts; `.env` and local SSH settings remain ignored. This does not establish absence of every possible secret.

## Cost and launch

Pilot-usage projection: **$117.76** for generation plus judging. A scenario with twice the token usage is **$235.51**. These are estimates, not worst-case bounds or enforced dollar caps.

Rates checked on 2026-10-10: GPT-OSS $0.15 input/$0.60 output and Sonnet $2 input/$10 output per million tokens. [Together pricing](https://www.together.ai/pricing); [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing). No cache discounts are assumed; cache writes and usage differences can affect actual billing.

Preparation made **zero model API calls**. It required no new GPU provisioning.
The user subsequently authorized the main experiment. The scientific runner bounds
request counts and batch sizes; the separate operational controller monitors the
usage threshold described above. The existing pod cost, any extra audits/retries,
and the later confirmation study are excluded.

Commands and the exact analysis/adjudication policy are in [main_study_protocol.md](main_study_protocol.md). After each live batch, inspect the saved result statuses and session failure artifacts: process exit alone is not proof that all requests succeeded. Started requests are preserved and not automatically repeated.

## Locked confirmation tasks

`HumanEval/117`, `HumanEval/17`, `HumanEval/22`, `Mbpp/265`, `Mbpp/436`, `Mbpp/456`, `Mbpp/728`, `Mbpp/754`
