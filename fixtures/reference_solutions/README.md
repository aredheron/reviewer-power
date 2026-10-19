# Provisional reference implementations

These 51 JSON bundles contain 102 agent-authored Python modules for the tasks
previously recommended as `suitable_for_reference_validation`. Each task has a
list-comprehension and a list-building loop implementation, an unchanged original
task ID and entry point, a proposed scoring target, and task-specific review notes.
The code is stored as strings to keep it out of Python import/test discovery.

**None is human reviewed or accepted.** The name of this directory follows the
research spec; it does not establish verification. Do not import, `exec`, or run
these modules on the host. Never copy this directory into evaluated-model context
or mount it in a model's workspace. These are hidden reference artifacts, not
visible fixtures or pilot membership.

Rebuild the metadata and readable review from the verified, unchanged caches:

```sh
uv run --locked python scripts/validate_references.py
```

The command parses and compiles code without executing it, compares full AST
signatures with source, and applies the existing `returned_list_v1` scorer to
both real implementations. It returns nonzero if any pair needs static resolution.
Currently all 102 signatures and compilation checks pass; 50 pairs have their
expected style labels. HumanEval/58's natural final sort is unsupported, and
equal values with different types leave representative equivalence unresolved.
Its pair remains present and flagged. No source record or prior recommendation
is silently changed.

`manifests/reference_validation.jsonl` records exact code/bundle/source hashes,
provisional regions, scorer evidence, and unexecuted suite counts. The generated
`planning/reference_validation_review.md` describes each pair's remaining checks.
The manifest contains no full solution or hidden input values. Repeated static
audits are byte-stable for identical inputs and tooling. Authoring is an agent
judgment, not a deterministic regeneration process; the authored bundles are the
reproducible source artifacts.

After independently completing the existing sandbox build, integration and parity
gate in `configs/sandbox/README.md`, an explicit execution command is available:

```sh
uv run --locked python scripts/validate_references.py \
  --oracle-image "$(cat data/sandbox/oracle-image.id)" \
  --run-id references-001
```

This uses the unchanged full EvalPlus base and plus suites for each complete
module. It accepts an immutable local image ID, never starts Docker/builds/pulls,
and has no host fallback. Every attempt is saved under ignored
`runs/reference-validation/references-001/`; an existing run ID is rejected, and
infrastructure failure stops with earlier records retained. No automatic retry
or update to human-review/acceptance status occurs. Two oracle passes alone do
not prove type/identity/side-effect equivalence outside the oracle's comparisons.

Sources: EvalPlus 0.3.1, HumanEval+ v0.1.10, MBPP+ v0.2.0. Attribution and upstream
license declarations are recorded in `manifests/evalplus_source.json`; these
bundles derive their contracts from those benchmarks. See the
[EvalPlus repository](https://github.com/evalplus/evalplus),
[HumanEval](https://github.com/openai/human-eval), and
[MBPP dataset](https://huggingface.co/datasets/google-research-datasets/mbpp).
No publication or relicensing decision is made here.
