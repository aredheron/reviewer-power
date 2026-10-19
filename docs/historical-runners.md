# Original collection workflows and archive dependencies

Use `scripts/replicate.py` for new completions of the presented studies and
`scripts/reproduce_results.py` for analysis of packaged results. These work from
the published prompt packets and compact evidence. The original runners below
are preserved because they document how the study was conducted and support
other experiments, including repository agents and self-hosted models.

## Why the original runners need an archive

The historical runners freeze a run at a fixed path, bind code, prompts, model
settings, sandbox/reference verification, prior outcomes, cost forecasts, and
approval records by hash, and refuse to overwrite completed work. Several later
studies derive task prompts and calibration from earlier manifests. Supplying only
an API key is therefore insufficient for these entry points.

| Runner | Important required historical inputs |
|---|---|
| `run_matched_review.py` | `runs/main/main-32-v1/manifest.json`; original exposed-task prompts; prior cost records; pilot/reference verification; saved approval and source bindings. |
| `run_matched_syntax.py` | Original syntax task catalog and calibrated judge revision; syntax study manifest; matched-review completion/budget bindings; verified reference executions. |
| `run_reviewer_observation.py` | Original main assignment templates and judge configuration, readiness and cost evidence. |
| `run_strong_preferences.py` | Observation completion and parent GLM readiness; matching historical strong prompts and budget accounting. |
| `run_expanded_preferences.py` | Imported/audited extra task banks, retained original rows, reference-judge validation and original protocol bindings. |
| `run_extension_main.py`, `run_mode_b_demo.py` | Visible fixture/task data and separate, verified agent and oracle sandboxes; extension pilot readiness. |
| `build_results_dashboard.py` | Its full declared source list plus completion audits and historical manifests. A partial archive is explicitly reported and is not a complete dashboard. |
| Recovery tools | The original designated attempts and immutable parent manifests; these scripts must not be used to choose favorable replacement outcomes. |

The [research specification](../planning/RESEARCH_SPEC.md), original protocols,
source modules, and script catalogue describe each workflow. The checked-in
`configs/experiments/*` files include earlier plans; the authoritative settings
for the presentation are `configs/replication/presentation.json` and its exact packets.
Do not mistake `main-32.json` for the later matched-policy study.

To inspect a historical run after restoring its external archive to the matching
ignored paths, use the corresponding runner's `--help`, `check`, `status`, or
`analyze` operation. Some `analyze` operations rewrite derived analysis files;
retain an immutable archive copy. Do not run `prepare`, `run`, `collect`, or
`--generate` on historical paths merely to inspect evidence. Earlier authorization
text records a past decision and does not authorize a new run or new expenditure.

## Tests and reconstruction limits

Tests of the scoring contracts, statistical analyses, and portable entry points
use synthetic fixtures or packaged evidence. Many original experiment tests also
inspect the actual ignored `runs/` trees, calibrations, or cached datasets.
A full `uv run pytest` requires that archive; their absence is a missing input,
not evidence that those experiments never ran. Those tests are retained and have
not been weakened or replaced with fabricated historical artifacts.

An exact operational replay additionally requires the original provider/model
revision and software/container snapshots. Hosted model identifiers alone do not
pin provider weights. A fresh scientific replication uses new identities, preserves
new failures, and reports differences instead of overwriting the historical record.

## Self-hosted GPU models

The earlier Nemotron/OLMo checkpoint workflows use a compatible GPU server and
OpenAI-compatible inference endpoint. Follow
[the inference environment guide](../configs/inference/README.md),
[model access profiles](../configs/models.yaml), and
[checkpoint diagnostic config](../configs/experiments/olmo-diagnostics.json).
They are separate from the three hosted models in the presentation tables.

Keep actual SSH endpoints and identity-file paths in ignored
`configs/ssh_config.local`, using `configs/ssh_config.example` as the template.
Private keys stay outside the repository. Use the documented
`ssh -F configs/ssh_config.local grader-power-gpu` alias and retain host-key
verification. The repository itself does not provision compute.
