# Second explicitly authorized generation recovery

This addendum applies only to `runs/style-preferences/syntax-preferences-v1`.
After the first recovery still received Fireworks account-suspension responses,
the user clarified: “oops payment didn't go through the first time. please try
again”. This new direction authorizes one additional fresh attempt for each of
the same 20 failed-generation cells. It does not alter or erase the original
main attempts or the first recovery. The exact approval and frozen cohort are
recorded under `operations/recoveries/generation-rerun-v2/`.

## Fixed cohort and unchanged scientific settings

The original main remains 10,080 assignments: 24 tasks per style family, two
user preferences, seven reviewer/consequence arms, three repetitions, and five
models. The fourth-repetition records remain retired. The new recovery uses the
same 20 original cells selected in generation-rerun-v1; it does not select new
cells based on correctness, judged style, or later failures.

The first recovery had 20 terminal attempts and no saved generations: five
explicit HTTP 412 account-suspension rejections and 15 intents stopped before
HTTP dispatch. Preserve all of those outcomes and their reservations. The new
cohort receives new request and assignment IDs, references its original main
assignment and the corresponding first-recovery assignment, and copies exact
prompts, tasks, model/provider identity, generation settings, and token limits.
Neither original nor first-recovery request IDs become dispatchable again.

There is exactly one new attempt per frozen cell in this authorization. A new
failure remains an outcome and is not automatically retried as another cohort.
The existing transient-rejected-429 handling remains the only transport retry
policy. No new pilot, task substitution, increased token allowance, judge-only
retry, or relaxed factual/style parser is authorized.

## Spending and restart safeguards

The combined original, pilot, validation, first-recovery, second-recovery, and
remaining main work retains the original ledger identity and the $400 hard cap.
All recorded usage and old outstanding reservations remain included. Each new
request obtains its own reservation; provider credit restoration does not raise
the experiment ceiling or justify releasing retained uncertain charges.

Verify the exact new approval, both parent recovery/source chains, preserved
source snapshots and outcomes, the frozen 20-cell mapping, and passing checks
before launch. Prove the previous controller and all paid descendants are gone,
the collection lock is released, and the pinned Docker sandbox is clean.
Preserve old owner and launch records. Any reviewed operational error-window
reset requires a hash-bound audit retaining every old failure; it is separate
from the recovery failure streak.

The Fireworks account-suspension streak remains at two. Another stopping episode
for the same or uncertain cause reaches three and blocks further autonomous
restarts pending new human direction. The new authorization does not clear that
streak. Follow the existing 1-, 2-, 4-, 8-, then 15-minute recovery-check ladder,
advancing only on genuine durable progress. Human STOP, source/report checks,
sandbox verification, scientific episode limits, provider caps/cooldowns, and
the five-failed-stages-in-48 guard remain binding.

## Separate recovered view and full lineage

Preserve the original selected-main analysis in `analysis-v2/` and the first
recovery view in `analysis-v3/`. Export the newly authorized view separately in
`analysis-v4/`, retaining exactly the same 10,080 original assignment identities,
conditions, and denominators. For each of the 20 targeted cells, use the one
designated second-recovery outcome even if it is missing or failed. Do not pick
the better of the three attempts. Retain full original and first-recovery
outcomes and the fresh-attempt mapping in the new lineage.

Report original, first-recovery, and second-recovery attempts and saved/resolved
outcomes separately. The 20 new attempts are not additional independent main
observations. Keep the three-repetition task-clustered analysis, 20-contrast
Holm family, model/provider provenance, unresolved outcomes, and missingness
bounds unchanged. The dashboard may prefer `analysis-v4/` only after verifying
its complete source hashes, original identity set, both recovery mappings, and
preserved earlier outcomes. Earlier studies remain untouched.
