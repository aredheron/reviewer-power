# Verification record

Verified on 2026-10-19 with Python 3.13.5 and the checked-in dependency lock.
No live model requests or new model-generated code executions were made for
this verification. Model/provider identifiers describe the historical runs;
current provider availability was not tested.

| Check | Result |
|---|---|
| `uv sync --locked` | Fresh environment installed from the existing lock; no dependency changes. |
| `uv run python scripts/reproduce_results.py --inference` | 60 evidence artifacts verified; five tables and all-condition counts rebuilt; both original 10,000-draw paired analyses matched the recorded analyses field by field. |
| Portable suite below | 327 passed. |
| `uv run ruff check .` | Passed. |
| `uv run ruff format --check .` | 329 files already formatted. |
| `uv run pyright` | Zero errors and warnings. |
| Frozen coding/factual packets | Checksums, unique identities, single-user-message shape, and compatibility with original factual scorers checked for all four studies. |
| Credential scan | No actual source-environment credentials or recognizable private-key/token values found in distributable files, including decompressed evidence. |

```sh
uv run pytest tests/test_publication.py tests/test_replication_portable.py \
  tests/test_matched_review_analysis.py \
  tests/test_preference_judge.py tests/test_style_judge.py \
  tests/test_code_extract.py tests/test_score_style.py
```

The portable tests exercise balanced planning, held pilots, plan/packet/source
integrity, judge blinding, request allowances, ambiguous failure handling, and
resume behavior. All four study adapters are exercised through generation,
correctness scoring, and judging with simulated providers and oracle responses.
These checks do not establish live endpoint compatibility or Docker isolation on
a new machine. Run the documented `verify-sandbox` command on that machine before
collecting new completions; skipped integration tests do not satisfy that gate.

The unrestricted historical-suite check, `uv run pytest --maxfail=3 -q`, reached
33 passed and 51 skipped before stopping at three missing-input errors in
`tests/test_all_recovery.py`. Those fixtures require the external
`runs/recovery/all-experiments-v1/operations/generation-audit.json` archive.
The full suite is therefore not reported as passing. The same check under the
agent's restricted shell first hit process-inspection restrictions in existing
lock tests; rerunning with local process inspection allowed cleared that obstacle.
The pytest configuration includes the repository root so historical `evals`
imports also work during test collection.

See [historical runner dependencies](../docs/historical-runners.md) for archive
requirements and [source provenance](source-code.json) for unchanged and added
implementation files.
