# Full EvalPlus evaluator

Status on 2026-10-10: Docker Desktop is available, the image has been built, and
all nine opt-in integration tests passed. The HumanEval/7 example's loop and
comprehension references passed all 4 base and 178 plus inputs. Both references
and a deliberately wrong answer agreed with EvalPlus's official checker inside
the isolated container. The subsequent 11-task pilot preparation verified both references for every
selected task, with 8,002 input checks matching the official checker. Human
suitability approval and exact pilot membership are recorded separately. This
does not certify unselected tasks or adversarial grading integrity.
Local evidence is in `runs/mode-a-verification/` and
`data/sandbox/mode-a-example-verification.json`; reproduce these gates on a new host.

## Build and pin

Build using only this directory as context. Its allowlist `.dockerignore` sends
only `Dockerfile` and `runner.py`; do not build from the repository root.

```sh
mkdir -p data/sandbox
docker build --iidfile data/sandbox/oracle-image.id configs/sandbox
export GRADER_POWER_ORACLE_IMAGE="$(cat data/sandbox/oracle-image.id)"
GRADER_POWER_DOCKER_TESTS=1 uv run pytest tests/integration/test_oracle_docker.py -v -rs \
  --junitxml=runs/mode-a-verification/docker-tests.xml
```

Execution accepts **only the immutable local `sha256:...` image ID**, never a tag,
and uses `--pull=never`. Record that ID with each attempt and retain/export the
built image for reproduction. The build recipe uses Python 3.12.12, EvalPlus
0.3.1, NumPy 2.5.3, and psutil 7.2.2 (the latter three match `uv.lock`). It installs
only the dependencies of EvalPlus's execution APIs; model clients and dataset
download dependencies are deliberately absent. The worker checks Python and
EvalPlus versions at runtime. The Python build tag can be republished: rebuilding
is not a promise of identical bytes. No image ID is invented or certified in this
repository as a universal constant; the actual immutable build ID is a
required deployment input. Image builds may download public packages, but
**evaluation has networking disabled**.

The integration tests require Linux containers with cgroup v2. They separately
exercise correct, base-only-correct, syntax-invalid, nonterminating, exception,
and abrupt-exit submissions; a broken canonical oracle; a host wall deadline;
effective cgroup/network/UID/privilege restrictions; and removal after each case.
Without explicit opt-in they skip. With opt-in but no daemon, they skip with an
explicit unmet-prerequisite reason; a missing image ID fails when Docker exists.

## Trusted boundary and semantics

`src/grader_power/datasets/oracle.py` accepts a trusted `LoadedDataset`, source
name/ID, and a **complete file/function implementation**. It stages only that
task and the exact submission in a new temporary directory outside the research
repository. Native inputs use host-to-container pickle so tuples and complex
numbers retain their meaning; never accept pickle supplied by a model. The host
reads only JSON result metadata back. Source caches are neither mounted nor
modified. No credentials, repository directories, or full corpus are mounted.

The **trusted evaluator container** receives one task's canonical solution and
full base/plus tests. It is separate from model-visible workspaces and agent
tools. Never mount its input directory in a generation sandbox or expose its
payload to a model. Submitted Python necessarily runs inside the evaluator with
the checker: EvalPlus is not an in-process secrecy or anti-cheating boundary
against deliberately introspective Python. Docker isolates the host; it does
not prove adversarial grading integrity. Use only the reviewed image, keep raw
submissions for audit, and do not treat this adapter as an adversarial-code
certification system.

The installed 0.3.1 APIs were inspected before implementation:

- `evaluate.get_groundtruth` uses `trusted_exec` on `prompt + canonical_solution`,
  with MBPP's output-not-None exceptions. The worker uses that same computation
  inside Docker, with no shared ground-truth cache.
- `evaluate.check_correctness` calls `eval.untrusted_check` on the complete
  `solution` separately for base and plus inputs.
- `untrusted_check` supervises `unsafe_execute`, which contains all special
  HumanEval/MBPP comparisons and tolerance rules. It collapses runtime exceptions
  and per-input timeouts into `fail`, and premature child exits into `timeout`.
  This worker calls the **unchanged `unsafe_execute`** and retains its full-detail
  behavior, but adds an exception-observing callable wrapper and records process
  exit codes. It preserves the function metadata/signature, source text, inputs,
  comparisons, and default timing rule: max(1 second, 4 × canonical time), with
  the upstream full-suite process cap of min(60, sum of limits) + 3 seconds.

Result statuses are `pass`, `wrong_answer`, `syntax_error`, `timeout`, `crash`,
`oracle_error`, and `infrastructure_error`. The two suite detail vectors and
counts must support a pass. Unhandled submitted exceptions/early exits are
crashes, whereas failed comparisons are wrong answers. A broken canonical
solution is an oracle failure. Outer deadlines can interrupt canonical work as
well as submissions and are reported as incomplete timeouts without attributing
blame. Docker/protocol/isolation and cleanup failures are infrastructure errors.
An OOM kill is a crash. No failed/timed-out/incomplete outcome is full correctness.

Only explicit `mode="full"` is supported. There is no quick mode, prompt append,
sanitization, signature rewrite, automatic retry, or host execution fallback.
Malformed/empty suites (including the known `Mbpp/793` source anomaly) are
reported as oracle errors without repairing the record or deciding eligibility.

## Limits and remaining gate

Each attempt gets a fresh non-root container, no network, one CPU, 512 MiB memory
with no additional swap, 64 processes, dropped capabilities, no new privileges,
read-only root and task mount, and a 128 MiB temporary filesystem. Docker's
default seccomp policy stays enabled. Proxy environment variables are explicitly
cleared. The default outer wall deadline is 180 seconds (caller-configurable
from 1–600); Docker control operations have separate 15-second bounds. Logs are
bounded by Docker rotation and are used only as a JSON transport, not published.
Limits are engineering defaults, not scientific task eligibility decisions;
some full upstream tasks may exceed them and must be reported as failures.

Before executing any code, the adapter inspects the created container and rejects
missing isolation controls. `docker rm --force` runs in `finally`, including
create/start/wait/protocol failures. Unverified removal overrides a pass and
reports the container name for manual cleanup. A daemon failure can prevent
cleanup; no application can claim removal in that case. Abrupt host termination
also requires checking labeled leftovers with:

```sh
docker ps -a --filter label=grader-power.oracle=true
```

After Docker integration succeeds, compare a small set of source tasks against
official `untrusted_check`/`check_correctness` inside the same isolated limits.
The adapter's explicit `verify_upstream=True` option performs this comparison
inside the same container and requires identical suite pass decisions and detail
vectors. All 22 references for the frozen 11-task pilot have passed this gate;
unselected tasks still need validation. The earlier Mode A example also checked a
known wrong answer. Pilot evidence is in `data/sandbox/pilot-task-verification.json`;
`manifests/tasks_pilot.jsonl` links exact source, fixture, reference, and review
provenance. Reproduce the nine integration tests and reference checks on a new host.
