"""Small verified Docker workspace for Inspect Mode B tools, never host execution.

The existing reviewed oracle image supplies Python, but no oracle data is staged
here. Only explicit visible files enter a fresh tmpfs. Invariant context files
are root-owned; the model runs as uid 65534 and can modify only src/ and /tmp.
"""

import asyncio
import json
import os
import re
import selectors
import shlex
import subprocess
import time
import uuid
from pathlib import PurePosixPath
from typing import Any, Literal, overload

from inspect_ai.util import ExecResult, SandboxEnvironment, SandboxEnvironmentConfigType
from pydantic import Field

from grader_power.config import Contract


class EpisodeLimits(Contract):
    """Recorded engineering ceilings; smaller values are useful for limit tests."""

    max_model_turns: int = Field(default=8, ge=1, le=16)
    max_tool_calls: int = Field(default=24, ge=1, le=48)
    episode_seconds: int = Field(default=180, ge=1, le=360)
    command_seconds: int = Field(default=10, ge=1, le=10)
    max_file_bytes: int = Field(default=65536, ge=128, le=65536)
    max_tool_output_bytes: int = Field(default=8192, ge=128, le=8192)
    max_model_input_bytes: int = Field(default=49152, ge=128, le=98304)

    @property
    def container_lifetime_seconds(self) -> int:
        """Retain the old lifetime or allow 60 seconds beyond a longer episode."""
        return max(240, self.episode_seconds + 60)

    def to_dict(self) -> dict[str, Any]:
        """Return all resolved ceilings for the shared assignment freeze."""
        return self.model_dump(mode="json")


class SandboxError(RuntimeError):
    """Sandbox/control failure with no host diagnostics or credential values."""


class SandboxLimit(SandboxError):
    """A bounded process was stopped; preserve any already observed output."""

    def __init__(self, kind: str, output: str = "") -> None:
        super().__init__(kind)
        self.kind = kind
        self.output = output


class WorkspaceFileError(SandboxError):
    """A missing, malformed, or unreadable final file, distinct from Docker failure."""

    def __init__(self, status: str) -> None:
        super().__init__(status)
        self.status = status


# The container stops even if its host supervisor is killed. Normal episode
# deadlines are stricter; this includes bounded setup/teardown allowance.
KEEPALIVE = "import time; time.sleep(240)"
TMPFS = {
    "/workspace": "rw,noexec,nosuid,size=16m,mode=755,uid=0,gid=0",
    "/workspace/src": "rw,noexec,nosuid,size=16m,mode=755,uid=65534,gid=65534",
    "/tmp": "rw,noexec,nosuid,size=128m,mode=1777",
}
PROXIES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)

# These trusted helpers are interpreted ONLY inside the inspected container.
# JSON stdin transports text literally; no model text enters a host shell.
INITIALIZE = r"""
import json, os, pathlib, sys
request = json.load(sys.stdin)
root = pathlib.Path('/workspace')
for name, text in request['files'].items():
    if name.startswith('src/'):
        continue
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))
    path.chmod(0o444)
print(json.dumps({'initialized': True}))
"""
FILE_OPERATION = r"""
import json, os, pathlib, stat, sys
request = json.load(sys.stdin)
root = pathlib.Path('/workspace')
descriptors = []
try:
    path = root / request['path']
    for part in [path, *path.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('symlink')
    if not path.resolve().is_relative_to(root):
        raise ValueError('outside workspace')
    if request['action'] in ('write', 'read'):
        # Resolve each component relative to an already-open directory. O_NOFOLLOW
        # prevents a concurrent model process from swapping in an escaping link.
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(parent)
        parts = pathlib.PurePosixPath(request['path']).parts
        for part in parts[:-1]:
            if request['action'] == 'write':
                try:
                    os.mkdir(part, dir_fd=parent)
                except FileExistsError:
                    pass
            parent = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            descriptors.append(parent)
        flags = os.O_NOFOLLOW | os.O_NONBLOCK
        flags |= (os.O_WRONLY | os.O_CREAT) if request['action'] == 'write' else os.O_RDONLY
        descriptor = os.open(parts[-1], flags, 0o644, dir_fd=parent)
        descriptors.append(descriptor)
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError('not a regular file')
        if request['action'] == 'write':
            os.ftruncate(descriptor, 0)
            data = request['content'].encode('utf-8')
            while data:
                data = data[os.write(descriptor, data):]
            result = {'written': True}
        else:
            data = bytearray()
            while len(data) <= request['limit']:
                chunk = os.read(descriptor, min(8192, request['limit'] + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
            if len(data) > request['limit']:
                raise ValueError('file size limit')
            result = {'text': data.decode('utf-8')}
    else:
        paths = []
        for directory, dirs, files in os.walk(root, followlinks=False):
            for name in sorted(dirs + files):
                paths.append(str((pathlib.Path(directory) / name).relative_to(root)))
                if len(paths) > 128:
                    raise ValueError('file count limit')
        result = {'paths': sorted(paths)}
    print(json.dumps(result, ensure_ascii=False))
except FileNotFoundError:
    print(json.dumps({'error': 'file_missing'}))
except (ValueError, UnicodeError):
    print(json.dumps({'error': 'file_invalid'}))
except OSError:
    print(json.dumps({'error': 'file_io_error'}))
finally:
    for descriptor in reversed(descriptors):
        os.close(descriptor)
"""


def relative_path(value: str) -> str:
    """Reject absolute, traversal, control-character, and noncanonical tool paths."""
    if not isinstance(value, str):
        raise TypeError("Path must be text")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("Use a canonical path relative to /workspace")
    return value


def visible_projection(
    files: list[dict[str, str]], target: str, *, direct_style_control: bool = False
) -> dict[str, str]:
    """Accept the frozen fixture surface, including explicit source-only controls.

    Direct-style capability controls have exactly README and target source; their
    style request is supplied directly by the shared runner. Ordinary episodes
    still require AGENTS.md and condition-appropriate paired review files.
    """
    relative_path(target)
    if not target.startswith("src/") or not target.endswith(".py"):
        raise ValueError("Target must be a Python file below src/")
    allowed = {
        target,
        "README.md",
        "AGENTS.md",
        ".ci/style-review.yaml",
        ".ci/selection-policy.yaml",
    }
    result: dict[str, str] = {}
    for item in files:
        if set(item) != {"path", "content"}:
            raise ValueError("Visible files require only path and content")
        name, content = item["path"], item["content"]
        if name not in allowed or name in result:
            raise ValueError("Unexpected or duplicate visible file")
        if not isinstance(content, str):
            raise TypeError("Visible content must be text")
        result[name] = content
    if direct_style_control:
        if set(result) != {target, "README.md"}:
            raise ValueError("Direct-style controls require exactly target and README")
    elif target not in result or "README.md" not in result or "AGENTS.md" not in result:
        raise ValueError("Visible fixture lacks target, README, or AGENTS.md")
    if (".ci/style-review.yaml" in result) != (".ci/selection-policy.yaml" in result):
        raise ValueError("Rubric and policy must be supplied together")
    return result


def verify_container(
    info: dict[str, Any], image: str, limits: EpisodeLimits | None = None
) -> None:
    """Reject any effective configuration outside the no-host-mount boundary."""
    host, config = info["HostConfig"], info["Config"]
    mounts = info["Mounts"]
    env = dict(value.split("=", 1) for value in config["Env"])
    allowed_env = {
        "PATH",
        "LANG",
        "GPG_KEY",
        "PYTHON_VERSION",
        "PYTHON_SHA256",
        "PYTHONDONTWRITEBYTECODE",
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        *PROXIES,
    }
    valid = (
        info["Image"] == image
        and host["NetworkMode"] == "none"
        and host["NanoCpus"] == 1_000_000_000
        and host["Memory"] == 536870912
        and host["MemorySwap"] == 536870912
        and host["PidsLimit"] == 64
        and host["ReadonlyRootfs"] is True
        and host["Privileged"] is False
        and host["CapDrop"] == ["ALL"]
        and host["SecurityOpt"] == ["no-new-privileges"]
        and host["PidMode"] == ""
        and host["IpcMode"] == "private"
        and not host.get("Devices")
        and not host.get("DeviceRequests")
        and not host.get("Binds")
        and not host.get("VolumesFrom")
        and host["Tmpfs"] == TMPFS
        and all(m["Type"] == "tmpfs" and m["Destination"] in TMPFS for m in mounts)
        and len(mounts) <= 3
        and config["User"] == "65534:65534"
        and config["WorkingDir"] == "/workspace"
        and config["Entrypoint"] == ["python"]
        and config["Cmd"]
        == [
            "-I",
            "-c",
            f"import time; time.sleep({(limits or EpisodeLimits()).container_lifetime_seconds})",
        ]
        and set(env) <= allowed_env
        and all(env.get(key) == "" for key in PROXIES)
    )
    if not valid:
        raise SandboxError("Effective Docker isolation verification failed")


class DockerWorkspace:
    """One bounded episode container; no tools are available until verification."""

    def __init__(
        self, image: str, files: dict[str, str], limits: EpisodeLimits
    ) -> None:
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("Use an immutable local sha256 image ID")
        if any(len(text.encode()) > limits.max_file_bytes for text in files.values()):
            raise ValueError("Initial file exceeds the frozen file limit")
        self.image, self.files, self.limits = image, dict(files), limits
        self.name = "grader-power-agent-" + uuid.uuid4().hex
        self.ready = False
        self.created = False
        self.cleanup_verified = False
        self.partial_files: dict[str, str] = {}

    def _call(
        self,
        args: list[str],
        *,
        data: bytes | None = None,
        timeout: float = 15,
        limit: int = 262144,
    ) -> tuple[int, str]:
        """Bound Docker client output in memory and elapsed time without a shell."""
        with subprocess.Popen(
            ["docker", *args],
            stdin=subprocess.PIPE if data is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ) as process:
            assert process.stdout is not None
            output = bytearray()
            deadline = time.monotonic() + timeout
            try:
                with selectors.DefaultSelector() as selector:
                    selector.register(process.stdout, selectors.EVENT_READ)
                    pending = memoryview(data or b"")
                    if process.stdin is not None:
                        os.set_blocking(process.stdin.fileno(), False)
                        selector.register(process.stdin, selectors.EVENT_WRITE)
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            raise SandboxLimit(
                                "command_timeout",
                                output.decode("utf-8", errors="replace"),
                            )
                        ready = selector.select(min(remaining, 0.1))
                        if not ready:
                            continue
                        if process.stdin is not None and any(
                            key.fileobj is process.stdin for key, _ in ready
                        ):
                            if pending:
                                pending = pending[
                                    os.write(process.stdin.fileno(), pending[:4096]) :
                                ]
                            if not pending:
                                selector.unregister(process.stdin)
                                process.stdin.close()
                        if not any(key.fileobj is process.stdout for key, _ in ready):
                            continue
                        chunk = os.read(
                            process.stdout.fileno(), min(4096, limit - len(output) + 1)
                        )
                        if not chunk:
                            break
                        output.extend(chunk)
                        if len(output) > limit:
                            raise SandboxLimit(
                                "tool_output_limit",
                                output[:limit].decode("utf-8", errors="replace"),
                            )
                code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
                return code, output.decode("utf-8", errors="replace")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()

    def _checked(self, args: list[str], *, data: bytes | None = None) -> str:
        code, output = self._call(args, data=data)
        if code != 0:
            raise SandboxError("Docker control operation failed")
        return output

    def start(self) -> None:
        """Create, inspect, then start and initialize; model code cannot run earlier."""
        args = [
            "create",
            "--name",
            self.name,
            "--pull=never",
            "--label=grader-power.agent=true",
            "--network=none",
            "--cpus=1",
            "--memory=512m",
            "--memory-swap=512m",
            "--pids-limit=64",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--ipc=private",
            "--user=65534:65534",
            "--workdir=/workspace",
            "--entrypoint=python",
            *(f"--env={key}=" for key in PROXIES),
            *(f"--tmpfs={path}:{options}" for path, options in TMPFS.items()),
            self.image,
            "-I",
            "-c",
            f"import time; time.sleep({self.limits.container_lifetime_seconds})",
        ]
        self.created = True  # Ambiguous create failures still require cleanup.
        self._checked(args)
        verify_container(
            json.loads(self._checked(["inspect", self.name]))[0],
            self.image,
            self.limits,
        )
        self._checked(["start", self.name])
        self._checked(
            ["exec", "--user=0:0", "-i", self.name, "python", "-I", "-c", INITIALIZE],
            data=json.dumps({"files": self.files}, ensure_ascii=False).encode(),
        )
        for path, content in self.files.items():
            if path.startswith("src/"):
                result = self._checked(
                    ["exec", "-i", self.name, "python", "-I", "-c", FILE_OPERATION],
                    data=json.dumps(
                        {
                            "action": "write",
                            "path": path,
                            "content": content,
                            "limit": self.limits.max_file_bytes,
                        },
                        ensure_ascii=False,
                    ).encode(),
                )
                if json.loads(result) != {"written": True}:
                    raise SandboxError("Initial source could not be staged")
        self.ready = True

    def file_operation(
        self, action: str, path: str = ".", content: str = ""
    ) -> dict[str, Any]:
        """Read/list or write src/ through a trusted in-container helper."""
        if not self.ready:
            raise SandboxError("Workspace is not verified and ready")
        relative_path(path)
        if action not in ("read", "write", "list"):
            raise ValueError("Unsupported file operation")
        if action == "write" and not path.startswith("src/"):
            raise ValueError("Only src/ is writable; repository guidance is read-only")
        if len(content.encode()) > self.limits.max_file_bytes:
            raise ValueError("Content exceeds the file limit")
        output = self._checked(
            ["exec", "-i", self.name, "python", "-I", "-c", FILE_OPERATION],
            data=json.dumps(
                {
                    "action": action,
                    "path": path,
                    "content": content,
                    "limit": self.limits.max_file_bytes,
                },
                ensure_ascii=False,
            ).encode(),
        )
        result = json.loads(output)
        if "error" in result:
            raise WorkspaceFileError(result["error"])
        return result

    def command(self, command: str) -> dict[str, Any]:
        """Run a shell command only in the verified container, with hard bounds.

        On a deadline/output breach, remove the whole container so children of
        the Docker exec process cannot survive. Partial output remains evidence.
        """
        if not self.ready:
            raise SandboxError("Workspace is not verified and ready")
        if (
            not isinstance(command, str)
            or len(command.encode()) > self.limits.max_file_bytes
        ):
            raise ValueError("Command must be text within the file-byte ceiling")
        try:
            code, output = self._call(
                ["exec", self.name, "/bin/sh", "-c", command],
                timeout=self.limits.command_seconds,
                limit=self.limits.max_tool_output_bytes,
            )
            return {"exit_code": code, "output": output}
        except SandboxLimit:
            try:
                for path in self.files:
                    if path.startswith("src/"):
                        self.partial_files[path] = self.file_operation("read", path)[
                            "text"
                        ]
            except (SandboxError, OSError, ValueError):
                # Fail closed even when resource exhaustion prevents capture.
                pass
            finally:
                self.close()
            raise

    def close(self) -> None:
        """Remove only this episode container; cleanup failures must remain failures."""
        self.ready = False
        if self.created and not self.cleanup_verified:
            self._checked(["rm", "--force", self.name])
            self.cleanup_verified = True


class WorkspaceSandbox(SandboxEnvironment):
    """Inspect sandbox interface over the verified, explicitly owned workspace.

    The solver starts this workspace after writing its exclusive attempt marker,
    then closes it in finally. This preserves interrupted-start evidence before
    any sandbox/provider action. No generic Docker compose defaults are used.
    """

    def __init__(self, workspace: DockerWorkspace) -> None:
        super().__init__()
        self.workspace = workspace

    async def exec(
        self,
        cmd: list[str],
        input: str | bytes | None = None,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | None = None,
        timeout: int | None = None,
        timeout_retry: bool = True,
        concurrency: bool = True,
    ) -> ExecResult[str]:
        """Execute only with the frozen user, environment, workdir, and deadline."""
        if (
            input is not None
            or cwd not in (None, "/workspace")
            or env
            or user is not None
        ):
            raise ValueError("Mode B does not allow sandbox environment/user overrides")
        if timeout not in (None, self.workspace.limits.command_seconds):
            raise ValueError("Use the frozen command deadline")
        result = await asyncio.to_thread(self.workspace.command, shlex.join(cmd))
        return ExecResult(
            success=result["exit_code"] == 0,
            returncode=result["exit_code"],
            stdout=result["output"],
            stderr="",
        )

    async def write_file(self, file: str, contents: str | bytes) -> None:
        """Write the intended source surface with descriptor-based path checks."""
        text = contents.decode("utf-8") if isinstance(contents, bytes) else contents
        await asyncio.to_thread(self.workspace.file_operation, "write", file, text)

    @overload
    async def read_file(self, file: str, text: Literal[True] = True) -> str: ...

    @overload
    async def read_file(self, file: str, text: Literal[False]) -> bytes: ...

    async def read_file(self, file: str, text: bool = True) -> str | bytes:
        """Read a bounded regular UTF-8 file, without following symlinks."""
        result = await asyncio.to_thread(self.workspace.file_operation, "read", file)
        content: str = result["text"]
        return content if text else content.encode("utf-8")

    @classmethod
    async def sample_cleanup(
        cls,
        task_name: str,
        config: SandboxEnvironmentConfigType | None,
        environments: dict[str, SandboxEnvironment],
        interrupted: bool,
    ) -> None:
        """Implement Inspect's cleanup contract for these explicit environments."""
        for environment in environments.values():
            if isinstance(environment, cls):
                await asyncio.to_thread(environment.workspace.close)
