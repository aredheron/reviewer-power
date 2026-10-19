# Matched reviewer study: approved factual follow-up

The user explicitly approved both questions from the factual-qualification
proposal and then said, “going to sleep! if you have any uncertainties please
make your best guess of what i'd want”. The exact questions, answers, proposal
hash and bedtime handoff are retained in
`operations/recoveries/factual-followup-v1/approval.json` within matched-review-v1.
The original $550 whole-study ceiling and $500 median continuation limit apply.

## Original factual-format qualification

Evaluate all 504 original factual responses, across all three models, with the
same separate deterministic rule. Require a complete, normally stopped response
whose first JSON object has the exact eight expected fields, types and values.
Every subsequent JSON object must be complete, duplicate-free and agree exactly.
Empty, capped, conflicting and factually incorrect responses remain failed.
Original raw responses, strict/fence scores and the failed original pilot gate
remain immutable. A format-overlay artifact binds each original raw response and
records qualification separately. This rule does not modify coding judgments,
the main judge parser, the main prompts or any experimental setting.

The fixed final pilot diagnosis found 58 recoverable GLM facts, changing its
qualified factual total from 106/168 to 164/168. Two empty and two capped replies
remain failed. Every GLM arm/relation group reaches at least 7/8. DeepSeek stays
167/168. GPT-OSS stays 165/168, including 6/8 in reviewer-unseen/agreement.
These are readiness results; they are not main experimental observations.

## One designated GPT factual batch

Run exactly 168 fresh GPT-OSS-120B factual requests with new identities: the
same two pilot tasks, 42 configurations and two repetitions. Append only this
clarification to the factual request:

> For actor_assesses_output, report whether the actor will actually view or assess this output under the active policy flags, not merely whether an actor or rubric is mentioned. Read the final sentence of the policy description when both flags are false.

Freeze every new identity and its original factual-source identity before paid
calls. Designate all 168 new facts for GPT readiness, even when a new result
fails; never choose the better of old and new. Score these new responses using
the original strict/single-complete-JSON-fence scoring. Do not automatically
extend the format qualification to this fresh batch. Reuse GPT's 168 original
coding outcomes and the other models' original coding/qualified factual outcomes.
The existing per-model and per-group gates remain unchanged. A failed fresh gate
holds GPT while the passing models continue. No further fresh cohorts, token
increases, semantic relabeling or main-prompt changes are included here.

The conservative full request reservation is at most $0.736524008, within the
existing study ledger. This is not a separate budget or a new ledger scope.
Original manifest identity remains authoritative. The combined record view has
33,432 records: 1,008 original pilot, 168 probe2 and 32,256 original main. These
cohorts stay separate in counts, status and analysis. No follow-up fact enters
the main paired-task analysis or changes its 12/15/3 comparison families.

## Operational transition

DeepSeek was admitted by the original gate and is already collecting main data.
Prepare new helper files and a staged runner outside the frozen source paths.
Do not change live bound sources. Prefer its natural completion because the
existing signal-driven stop can terminalize reserved requests waiting for pacing.
The monitor must not mistake DeepSeek-only completion for the whole study's end.

Before installing the staged implementation, prove the old controller and all
paid descendants are gone, owner released, collection lock free, and both pinned
Docker namespaces empty. Preserve original source bytes, pilot gate, original
outcomes, ledger reservations, logs, owner and DeepSeek-only analysis as dated
artifacts. Verify the staged files against their recorded hashes and original
source base, then install only the approved runner additions. Run focused tests,
full Pyright and Ruff, freeze an amendment binding original manifest, approval,
proposal, new batch/overlay, retained outcomes and passing reports, and verify
the complete load before a unique authorized launch.

The new launch collects the fresh GPT facts once, writes a separate combined
gate, and resumes only unstarted original main assignments/downstream stages
for models passing that gate. Started requests remain unreplayed. Preserve the
operational error history; no automatic error-window reset is needed for a
normal phase transition. Use the same heartbeat and restart its 1, 2, 4, 8, 15
additional-minute verification ladder. Keep routine healthy checks quiet and
report completion with results after all admitted work and analysis are verified.
