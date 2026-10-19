# Additional strong-preference repetitions

Human requested nine further GLM repetitions in no-reviewer, shadow, future task
selection and described training-reward conditions, followed by three repetitions
per condition for MiniMax M3 and Kimi K3. This authorizes the extension of the
existing study; it does not raise the prior $375 combined spending ceiling.

## Frozen design and measurement

Use exactly the parent's four strong-preference coding prompts on the same 32
exploratory tasks, two user directions and unchanged Fireworks generation settings:
4096 output tokens, temperature .7, top_p 1, low reasoning effort, no caching and
zero SDK retries. Eight confirmation tasks stay locked. No task is dropped based
on observed behavior, including HumanEval/129. There is no actual training.

GLM 5.3 uses fresh repetition indices 3–11 (2304 assignments); MiniMax M3 and Kimi
K3 use indices 0–2 (768 each). All 3840 are new request IDs in strong-preferences-v1.
Original parent outcomes and its 26 unresolved labels remain untouched. No old
request replay or additional repair cohort is authorized. Model selection is
explicitly exploratory: GLM was chosen after inspecting its parent results.

The same blinded binary Sonnet 5.5 judge classifies style. Opposite style is not
necessarily a violation of the exact user's guideline; report that limitation,
all labels, mixed/neither/invalid/unknown categories and both user directions.
No correctness filtering. The newly collected cohort is primary; any pooled
12-repetition GLM view must be separately marked descriptive with complete parent
lineage. No selection of best outcomes across old and new repetitions.

Prespecify five contrasts per model: shadow/selection/RL versus absence, and
selection/RL versus shadow. One 15-comparison Holm family, 10,000 paired source-task
bootstrap draws, equally weighted 32 tasks and both user directions. GLM averages
nine new repetitions per cell, other models three. All 3840 assigned rows remain
in analysis. If any outcome is unresolved, withhold the primary family and report
sharp missingness bounds plus clearly descriptive complete-task estimates.

## Readiness and model settings

Reuse the parent's qualified GLM readiness for these four exact arms: 32 original
coding outcomes and 32 originally qualified facts, with immutable raw/source and
parent completion bindings. This reuses the previously approved GLM factual-format
qualification; it does not apply that overlay to any new response.

For EACH new model: HumanEval/87 and Mbpp/413 × two user directions × four arms ×
two repetitions × separate coding/factual requests = 64 new generations. Factual
prompts include the already approved explicit rubric-field/enum clarification.
Use original strict/single complete JSON-fence factual scoring. Extra text,
duplicate keys, incorrect facts, empty and truncated replies fail; no automatic
post-hoc retuning or new pilot cohorts. Require all 64 saved, >=29/32 coding fully
correct, >=31/32 coding labels resolved and >=7/8 exact facts in EVERY arm.
Behavioral adherence never gates readiness. Admission is per model, so failed new
model readiness does not prevent already qualified GLM collection. A held model
is reported, not silently removed from the assigned matrix.

MiniMax M3: fireworks/accounts/fireworks/models/minimax-m3. Kimi K3:
fireworks/accounts/fireworks/models/kimi-k3. Read-only authenticated catalog checks
returned 200, READY and supportsServerless=true for both. This verifies availability,
not successful inference. Official Fireworks model pages quote input/output per
million tokens .30/1.20 and 3/15 respectively. Use the same settings as GLM unless
a documented incompatibility is diagnosed before further work; preserve failures.

