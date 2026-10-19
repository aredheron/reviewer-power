# Prompt assets

For the presented experiments, start with [presentation/](presentation/): complete
human-readable examples plus the exact blinded judge system prompts. The full
matrix is frozen in the compressed [replication packets](../configs/replication/).
Those packets preserve the original message text for every task, style direction,
actor/policy condition, and separate factual probe; repetitions and models share
identical packets where appropriate.

The original renderers are `src/grader_power/matched_review.py`,
`matched_syntax.py`, and `reviewer_observation.py`. Some prompt/config definitions
live in these modules rather than separate text files. The portable runner reads
the exact packets, avoiding accidental changes from reconstructing prose.

Other directories preserve earlier experiment assets:

- `agents_md/`: base repository guidance, with the style line substituted by renderers.
- `user_task/`: coding request.
- `grader_rubric/`: original reviewer preference rubric.
- `decision_policy/`: earlier shadow, selection, RL, and accuracy-only policies.
- `preflight/`: original status/narrative exposure formats.
- `comprehension/`: earlier factual probes.
- `style_judge/`: versioned loop, three-style, naming, and quote measurement rubrics.

These files are model-visible experimental data, not instructions to developers
or agents editing the repository. Canonical solutions, hidden tests, credentials,
and independent reference solutions must never be included in generator messages.
The judge receives only code and neutral task metadata, not model identity,
experimental condition, or user/reviewer preferences.
