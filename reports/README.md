# Evidence and completion accounting

The main README uses designated recovery outcomes with the same assignment
identities and denominators as the original cohorts. Recovery is not a new
independent sample to add to the originals. Every result export is identified
by its source path and SHA-256 in [artifact-manifest.json](artifact-manifest.json).
The packaged JSON is gzip-compressed; the manifest records both the original
file hash and the packaged-file hash. Decompression requires only Python's
standard library. See [tables/](tables/) for ordinary CSV files.

## What is included

- Full measured rows for the presented studies, with condition factors, model,
  task, repetition, assignment IDs, style labels, correctness, missingness, and
  recovery linkage. Original and recovery versions remain separate files.
- Original statistical analyses, recovery designation/repair lists, and readiness
  gates, including failures and later qualifications.
- All-condition counts and measured rows for the supplementary studies.
- [Eleven illustrative attempts](examples/saved-attempts.json), including exact
  prompts, raw provider responses, original measurements, and available judgments.
  Selection uses the first lexicographic assignment ID in each model/category;
  it is not random or representative. Categories include baseline, matching,
  opposite, and invalid results where available. These examples are explicitly
  from the original view and may be superseded by a designated recovery.
- A [saved-reasoning exploration](reasoning/report.txt), with classifications,
  examples, and denominator qualifications. It is separate from behavioral scoring.
- Exact model-visible coding/factual packets in [configs/replication/](../configs/replication/),
  deduplicated across repetitions/models without modifying message bytes.

The full approximately 15 GB run archive, upstream caches, private configuration,
spending databases, and operational logs are not distributed here. Source paths
in the manifest are archive-relative identifiers, not downloadable links or
claims that those files exist in this checkout. The compact rows reproduce the
numerical analyses; independently rejudging every historical completion requires
the external raw archive. Checksums detect changed bytes but do not independently
prove when or how a run was conducted.

## Main and robustness cohorts

| Cohort | Assigned rows | Final state | Packaged row file |
|---|---:|---|---|
| Matched reviewer and role names | 32,256 | 32,255 resolved; one GLM evaluator/no-consequences conflict judgment remains unknown | [matched-review](data/matched-review.json.gz) |
| Original matched reviewer view | 32,256 | 113 unresolved before designated recovery | [original](data/matched-review-original.json.gz) |
| Matched naming and quotes | 20,736 | 17,280 resolved admitted assignments; 3,456 GLM naming assignments held/unrun | [matched-syntax](data/matched-syntax.json.gz) |
| Original matched syntax view | 20,736 | 3,664 unresolved, including the 3,456 held assignments | [original](data/matched-syntax-original.json.gz) |
| Earlier observation/strength panel | 9,600 | 9,599 resolved; one GLM soft-unseen judgment remains unknown; five models and ten arms | [observation](data/observation.json.gz) |
| Extra strong-preference cohort | 3,840 | Final recovered rows all resolved; GLM nine added repetitions, MiniMax/Kimi three each | [strong](data/strong.json.gz) |

Matched-review recovery designates 147 records (130 generator replacements,
17 judge-only requests) plus 86 mechanical repairs. It does not enlarge 32,256
assigned slots. The syntax recovery designates 311 records (197 generator
replacements, 114 judge-only requests) plus 62 mechanical repairs, without changing
the 20,736 assigned slots. Original failures and strict analyses remain included.
These are same-limit, explicitly designated recoveries, not selection of whichever
of two observed outputs yields the preferred result.

The strong-preference presentation table selects GPT-OSS/DeepSeek strong arms from
`observation`, and pools GLM's three original repetitions with nine additional
repetitions from `strong`. There are 192 outputs per GPT-OSS/DeepSeek cell and
768 per GLM cell. This is explicitly descriptive, not a claim that the pooled
cohort was originally planned as one 12-repetition experiment.

## Readiness evidence

- [Original loop pilot gate](data/matched-review-pilot-gate.json.gz).
- [Qualified loop gate](data/matched-review-followup-qualified-pilot-gate.json.gz),
  [follow-up gate](data/matched-review-followup-probe2-gate.json.gz), and
  [format qualification](data/matched-review-followup-format-overlay.json.gz).
- [Qualified syntax gate](data/matched-syntax-qualified-pilot-gate.json.gz),
  which preserves the failed GLM naming group.
- [Observation pilot](data/observation-pilot-gate.json.gz) and
  [strong-preference pilot](data/strong-pilot-gate.json.gz).

Coding correctness and factual interpretation, not behavioral compliance, gate
admission. The original protocols document thresholds and later amendments.
Original factual failures must not be described as an initial universal pass.

## Reproduce and inspect

From the repository root:

```sh
uv run python scripts/reproduce_results.py --inference
python - <<'PY'
import gzip, json
from pathlib import Path
path = Path('reports/data/matched-review.json.gz')
rows = json.loads(gzip.decompress(path.read_bytes()))
print(len(rows), sum(row['label'] is None for row in rows))
PY
```

The inference option reproduces the original paired analyses, including the
withheld families, and compares the results field by field with the saved analyses.
Do not interpret a held/unresolved cell as a zero, divide only by correct solutions,
or pool original and replacement attempts as independent observations.

[Source-code provenance](source-code.json) distinguishes implementation files
that match the recorded source revision from additions and small adaptations.
The publication and portable-replication tools have their own tests. Frozen run manifests in the external archive identify the exact
historical source snapshots. Current source hashes are not a substitute for
those historical bindings.
