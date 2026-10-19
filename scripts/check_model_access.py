"""Perform one explicitly requested live smoke generation; retain an ignored record."""

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from dotenv import dotenv_values

from grader_power.inference import check_model_access, load_access_profiles


def main() -> None:
    """Load credentials without exporting them, then save a private result per call."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "profile", choices=tuple(load_access_profiles(root / "configs/models.yaml"))
    )
    parser.add_argument(
        "--generate", action="store_true", help="Make one potentially billable request"
    )
    args = parser.parse_args()
    profile = load_access_profiles(root / "configs/models.yaml")[args.profile]
    values = {**dotenv_values(root / ".env", interpolate=False), **os.environ}
    profile.resolve_access(values)
    if not args.generate:
        print(
            "Local settings present; no provider request made. Use --generate to test access."
        )
        return
    os.environ["PYTHON_DOTENV_DISABLED"] = "1"
    result = asyncio.run(check_model_access(profile, values))
    directory = root / "runs" / "access-checks"
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stamp}-{args.profile}-{uuid4().hex[:8]}.json"
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    # Prevent an unexpected provider echo from persisting credential values.
    for name, value in values.items():
        if value and name.endswith(("_API_KEY", "_TOKEN")):
            payload = payload.replace(value, "[REDACTED]")
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w"
    ) as file:
        file.write(payload + "\n")
    print(
        json.dumps(
            {"profile": args.profile, "status": result["status"], "record": str(path)}
        )
    )
    raise SystemExit(0 if result["status"] == "generated" else 1)


if __name__ == "__main__":
    main()
