# Explicit generation recovery after restored Fireworks credits

This addendum applies only to `runs/style-preferences/syntax-preferences-v1`.
It records the user's new direction: “my bad, i've added credits. please continue
and rerun any generations that failed.” The exact message,
frozen eligibility list, prior outcomes, and operational checks are bound in
`operations/recoveries/generation-rerun-v1/` before the recovery is launched.
Earlier protocols, approvals, qualification rules, and failed gates remain
unchanged and preserved.

## One fresh attempt for the audited current failures

The recovery selects the terminal main-generation failures present at this
pause, independently of style labels, correctness, or treatment effects. The
preparation snapshot identifies 20 targets: 18 started attempts without saved
responses and two saved responses terminated at the generation token limit.
The frozen recovery records and their original generation evidence, rather than
a moving query over later outcomes, define the authorized cohort.

Each target receives exactly one fresh assignment ID in phase
`generation_rerun_v1`, linked by `source_assignment_id` to its original main
assignment. Model, provider, public task, prompt messages, condition,
generation settings, and token limit are copied exactly. Scoring and judging
of these new responses use the same approved pipeline and strict style parsers.
No prompt repair, increased token limit, fresh pilot, or replacement task is
part of this approval.

The original 10,080 main assignments, their request IDs, saved or missing
outputs, errors, incomplete responses, labels, and reservations remain intact.
The 3,360 retired fourth-repetition assignments remain unstarted. Ordinary
continuation admits only the remaining unstarted original assignments and
unstarted downstream stages. These specifically approved fresh generation
IDs are an additive exception to the usual no-replay rule; they do not make
existing request IDs dispatchable again. The existing transport still retries
only explicitly transient rejected HTTP 429 responses. A failed fresh recovery
attempt stays failed; it is not recursively eligible for another attempt.

Judge-only failures on otherwise completed generations are outside this
generation-recovery cohort and retain their unresolved outcomes. No labels are
inferred from code or from an earlier judgment to fill a failed new attempt.

## Spending and restart safeguards

All fresh generation, scoring, and judging work remains in the original flat
ledger scope under the unchanged $400 whole-study ceiling. Existing paid usage
and outstanding reservations are retained, including reservations for rejected,
unsent, and uncertain old requests. New requests receive separate reservations;
restored provider credit is not an increase in the study's spending authority.

Before launch, verify that the old controller and all paid descendants have
finished, the collection lock is released, and the same pinned Docker sandbox
is clean. Bind the new approval, exact cohort, old sources, retained outcomes,
passing tests, source amendment, and any reviewed failure-window reset. A reset
must preserve every historical failure in its audit. Preserve old owner and
launch records and create a unique new authorized launch.

Human STOP, immutable source/report checks, strict judgment validation,
scientific episode limits, verified numeric provider caps, cooldowns, and the
five-failed-stages-in-48 guard remain binding. Track new stopping episodes by
root cause and block after three same or uncertain-cause episodes. The credit
restoration permits this concrete recovery, not repeated unchanged billing
tests or an unbounded restart loop.

## Original and recovered analyses

Keep `analysis-v2/` as the original selected-main analysis. Save the recovery
view separately in `analysis-v3/`, with exactly one row for each of the same
10,080 original selected assignments and explicit links to the frozen recovery
manifest and fresh attempts. Every targeted cell uses its designated fresh
attempt in this view, including when that attempt fails; do not choose whichever
attempt has a more favorable result. Preserve the original outcome and recovery
lineage alongside the replacement fields.

Report original attempts, added attempts, saved responses, scored responses,
resolved labels, terminal unresolved labels, and missing results separately.
The recovered analysis retains the three-repetition design, model/provider
provenance, task-cluster uncertainty, missingness bounds, and 20-contrast Holm
family. The recovery creates no additional independent main observations.

The local dashboard may prefer the verified recovery view once it is available.
It must verify matching original assignment identities and recovery bindings,
display the recovery provenance, and avoid adding both attempts to its
denominators. Older studies and their saved analyses remain unchanged.
