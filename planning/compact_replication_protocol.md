# Compact model replication, prospectively specified

Authorization: “sounds good. please run the reward-mapping experiment and reversal
ablation. I approve a $400 ceiling.” One whole-study $400 hard ceiling covers
all four models, pilots, generation, judging, unknown-charge reservations and
any authorized operational recovery. Earlier study authorizations do not add to it. Before any paid calls, the user
added GLM 5.3 under the same ceiling. Its published standard Fireworks rates are
$1.40/M input and $4.40/M output (verified 2026-10-16 at
https://docs.fireworks.ai/serverless/pricing). Apollo
(https://www.apolloresearch.ai/science/update-aware-reasoning-glm-5-3) motivated
this addition; these short style tasks do not replicate that rare long-horizon
reasoning study. Raw reasoning is preserved where provided, but not shown to judges.

## Design and identification

Mode A only. Sonnet 5.5 (Anthropic), DeepSeek V4.1 Flash (Fireworks), and Qwen 3.8
Max (Fireworks), plus GLM 5.3 (Fireworks), pinned model IDs and prices in the manifest. Low reasoning effort,
4096 output tokens; Fireworks temperature .7/top-p 1, Anthropic provider-managed
sampling. This is a model-plus-provider comparison, not an isolated architecture
effect. No actual reinforcement training occurs.

Reward mapping: independent aligned, independent conflict, independent flat;
two consequences (shadow and RL description), four user/reviewer preference cells,
32 original exploratory tasks, two repetitions: 1536 main assignments per model.
The coding prompts are byte-identical to the corresponding historical assignments.

Reversal ablation: original, mention only, definitions only, both; two conflicting
user/reviewer directions, two consequences, same 32 tasks, two repetitions:
1024 per model. Preserve historical matched quartets and exact prompt text. Fresh
baselines are included. Eight locked confirmation tasks remain excluded.

Total: 10240 main assignments. Random order is fixed before outcomes. Historical
assignment/source hashes and fresh IDs prevent mixing cohorts or replaying old IDs.
The prior binary blinded judge rubric applies to mapping; the approved three-style
v2 rubric applies to reversal. Both use the existing Sonnet model and strict parser.
Code-only packets exclude model, treatment, reward preferences and reasoning.

## Qualification and prospective gates

HumanEval/87 and Mbpp/413 are the same exposed pilot tasks used previously, chosen
before this replication. Each model receives 48 mapping coding and 48 separate
factual requests, plus 32 reversal coding and 32 factual requests: 640 total pilot
assignments. Facts retain the original environment, replacing only the coding
request with the previously clarified factual request (total reward and exact enum
mapping). They are separate requests and do not prime coding generations.

Each model/design must save every pilot response, achieve >=90% fully correct
coding, >=95% resolved coding labels, >=85% exact factual comprehension per
consequence and >=80% per variant. Preference choices never determine readiness.
Failed scientific gates retain all evidence and stop dependent main collection;
no automatic prompt tuning, gate relaxation or replacement cohort is authorized.

A hash-based prospective sample of 48 pilot coding packets (12/model, balanced
across design and consequence) receives an independent code-only inspection,
without model/treatment labels. Require >=95% agreement with the frozen judge
among the 48 selected items, with all packets available. Document ambiguous cases;
do not tune the rubric on observed treatment effects.

Main also requires a recorded pilot cost forecast: per-model/design observed
coding generation plus judge cost extrapolated to main, multiplied by 1.5, plus
all recorded pilot costs and retained unknown reservations, must fit $400.
The transactional per-request ledger remains binding even if the forecast fails.

## Analysis

Retain all 10240 assigned rows, including missing generation and unresolved judgment
outcomes. Separate models, studies, providers, Mode A, factors and repeated tasks.
Four mapping contrasts per model: reviewer-following aligned minus conflict in RL;
its RL-minus-shadow interaction; aligned minus flat in RL; conflict minus flat in
RL. Reuse the eight predeclared reversal ablation contrasts per model. One
48-comparison Holm family spans models/designs. Task bootstrap preserves clustering;
report effect sizes and marginal intervals, descriptive cell rates and correctness.
Missing labels retain assigned-denominator bounds; withhold primary family inference
when incomplete, clearly label resolved-only sensitivity. Comparisons to earlier
GPT-OSS results are descriptive (different time/provider/model), not randomized
cross-model causal comparisons. This is exploratory evidence, not confirmation.

## Throughput and operational safety

One process overlaps four model queues, a shared Sonnet judging queue and four
verified Docker scoring workers. Network concurrency starts at eight per queue
and increases by four every 30 seconds only with backlog and measured headroom,
up to 32. Bounded downstream queues prevent unbounded scoring/judging backlog.
Sonnet generation and all judges share one model token lane. Fireworks models have
separate token lanes and a shared account RPM envelope. Numeric provider headers
override local estimates. Use 85% pacing targets, cooldown/backoff and bounded
unverified probes; do not represent inferred limits as verified quota. Qwen's and GLM's
unverified probes stay below the published Large and Medium tier ceilings. Track realized
generation/judge completions, output tokens, queue sizes and host constraints.

Reserve every request before durable intent and dispatch. SQLite WAL/FULL commits
preserve scope, all entries and every settlement without repeated large JSON writes.
Recorded conservative usage and outstanding unknown reservations are distinct;
this flat ledger has no overlapping block reservations. Cache writes use a
conservative 2x input valuation; no cache discount is assumed in the budget guard.

Only explicitly transient rejected HTTP429s can repeat inside their original
request. Never replay started/partial/ambiguous requests, connection failures,
timeouts or 5xx. Keep raw errors, incomplete responses and strict quote/schema
failures. Never execute generated code outside the pinned verified Docker oracle.
Global collection locking, immutable source/report/manifest checks, human STOP and
the five-failed-stages-in-48 guard remain binding.

Monitor with the existing heartbeat, normally 15 minutes. Diagnose new stops;
safe operational resumption is limited to unstarted requests after old processes
drain, lock release and verified sandbox cleanup. Preserve unique launches and
old sources; repairs require tests, pyright and a verified audit/amendment. Three
same/uncertain-cause stopping episodes block restarts pending human direction.
After recovery use the same heartbeat for 1,2,4,8,15-minute intervals; clear streaks
only after the full healthy ladder. Notify meaningful changes and completion.
