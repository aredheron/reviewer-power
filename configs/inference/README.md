# Nemotron inference environment

This GPU environment is separate from the project's CPU/Mac `uv` environment.
The tested target is Linux x86_64, Python 3.12.3, an A100 80 GB, and NVIDIA driver
580.126.16. `requirements.txt` pins vLLM 0.11.0, Transformers 4.57.1, and
Hugging Face Hub 0.35.3; vLLM requires PyTorch 2.8.0. Other hardware requires its
own compatibility check. Do not install these packages in the local project venv.

Use a private persistent directory chosen for your host. Copy only the public
requirements and model profiles to it, not the repository's `.env` or SSH config.
On a PyTorch template, create an environment with system-site access, preserving
the system environment; uv may still install matching dependencies in the venv.

```sh
# Set INFERENCE_DIR to your own persistent directory first.
uv venv --system-site-packages --python python3 "$INFERENCE_DIR/venv"
uv pip install --python "$INFERENCE_DIR/venv/bin/python" \
  -r "$INFERENCE_DIR/requirements.txt"
uv pip freeze --python "$INFERENCE_DIR/venv/bin/python" \
  > "$INFERENCE_DIR/resolved-packages.txt"
```

The three direct pins are a build recipe, not a complete transitive lock. Retain
the resolved package inventory alongside each server's logs and hardware details.
The root `uv.lock` pins the client and smoke-test environment separately.

## Checkpoints

The public `configs/models.yaml` records immutable Hugging Face repository
revisions and the SFT subdirectory. Download final's root model/tokenizer files
and only `Nemotron-Cascade-8B-SFT/*` from the intermediate repository. The latter
repository's root is not itself a loadable SFT checkpoint. Keep model weights and
resolved local paths in ignored/local storage. Public downloads do not require a
Hugging Face token; use `token=False` with `snapshot_download` to avoid forwarding
an unrelated token.

For example, in the serving environment:

```python
from huggingface_hub import snapshot_download

snapshot = snapshot_download(
    "nvidia/Nemotron-Cascade-8B-Intermediate-ckpts",
    revision="9b379bd8f8ecb1feb19825efefe319bd30d8a223",
    allow_patterns=["Nemotron-Cascade-8B-SFT/*"],
    token=False,
)
# Serve snapshot + "/Nemotron-Cascade-8B-SFT", not snapshot itself.
```

Extract the final checkpoint's template from its pinned `tokenizer_config.json`
and use that same file for both checkpoints (approved on 2026-10-10):

```python
import hashlib
import json

# Set final_directory and template_path to pathlib.Path objects for your paths.
template = json.loads((final_directory / "tokenizer_config.json").read_text())[
    "chat_template"
]
assert hashlib.sha256(template.encode()).hexdigest() == (
    "1d357aa3a95cc57d98140d0ecbff42761eb749ef73eb9766535878259887b4d8"
)
template_path.write_text(template)
```

## Serve and connect

Run one checkpoint at a time, preserving the same precision, context limit,
tokenizer, and serving flags. Before switching, ensure the previous process has
stopped and no other work is using the GPU. Set `MODEL_DIRECTORY` to the downloaded
checkpoint and `SERVED_NAME` to `nemotron-final` or `nemotron-sft`. Set
`CHAT_TEMPLATE` to the extracted final template above. Set
`VLLM_API_KEY` privately in the server's environment; never put its value in a
command argument, public file, or log.

```sh
"$INFERENCE_DIR/venv/bin/vllm" serve "$MODEL_DIRECTORY" \
  --served-model-name "$SERVED_NAME" \
  --chat-template "$CHAT_TEMPLATE" \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 --max-model-len 8192 --max-num-seqs 4 \
  --gpu-memory-utilization 0.50 --generation-config vllm --enforce-eager
```

The context/concurrency settings are bounded access-test choices, not frozen pilot
settings. Use a process supervisor or detached process with private logs. Do not
log the environment. Confirm readiness through `/v1/models`, including the served
name, before making a synthetic request.

From the research repository on the client machine:

```sh
ssh -F configs/ssh_config.local -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:8000:127.0.0.1:8000 grader-power-gpu
```

The local ignored `.env` needs `VLLM_BASE_URL=http://127.0.0.1:8000/v1` and the
matching `VLLM_API_KEY`. Both service and tunnel bind only to localhost. Do not
disable SSH host-key verification or expose an unauthenticated public port.
Historical private access records were kept in ignored
`runs/access-checks/remote-runtime.local.json` and `ssh-tunnel.local.json`.
They are not included. Record your own launch settings and process identity
locally; check endpoint readiness before reuse or shutdown. The server and tunnel must be restarted after
their respective host processes stop.

The smoke profiles pass `enable_thinking: false` to the shared final chat
template. This is an explicit access-test choice only. The pinned final and SFT
tokenizer files **do not contain the same template**, despite the model card's
description: final inserts a default system message and a `/no_think` user suffix;
SFT omits that default message and prefills an empty `<think>` block instead.
The pinned `tokenizer.json` files are byte-identical. The shared template removes
this known input-format difference; it does not establish that this format is
optimal for SFT. Preserve the override and template hash in experiment records.
Separately select and record thinking, sampling, token limits, and retry policy.
Inference generates code as text; functional evaluation remains a separate
verified Docker sandbox requirement.

## OLMo diagnostic pair

The separate `configs/experiments/olmo-diagnostics.json` pins two public revisions
of `allenai/Olmo-3.1-32B-Think`. Their native `chat_template.jinja` files are
byte-identical and prefill `<think>`; preserve this behavior. Serve one checkpoint
at a time under `olmo-early` or `olmo-late`, with its matching pinned snapshot and
native template. Do not use the Nemotron template override. The existing vLLM
0.11.0 supports the model architecture but has no OLMo reasoning parser. Its
DeepSeek parser is incompatible with OLMo's ordinary-text reasoning markers;
do not pass `--reasoning-parser deepseek_r1` for this model.

The diagnostic serving settings are BF16, `--max-model-len 12288`,
`--max-num-seqs 1`, `--gpu-memory-utilization 0.92`,
`--generation-config vllm`, and `--enforce-eager`, without a server reasoning
parser. Keep the same authenticated localhost binding and SSH tunnel. The client
retains the untouched response as `provider-output.json`, then requires exactly
one `</think>` terminator and separates reasoning and final content into Inspect
content items saved in `raw.json`. Missing/ambiguous boundaries remain explicit
failures. The original probe and code scorers consume final content only. Record actual
launch arguments, model revisions, template hashes, software, and hardware before
interpreting the results. Do not infer readiness merely from a live process.

Each checkpoint has roughly 65 GB of weights. Preserve existing checkpoints and
private connection settings; no provider provisioning or public endpoint is
needed. Validate GPU memory and storage on the actual host before loading.
