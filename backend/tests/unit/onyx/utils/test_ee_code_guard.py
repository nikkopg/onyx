"""Behavior of the EE switch in a build that ships no ``ee`` package
(``backend/Dockerfile`` with ``INCLUDE_EE=false``)."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[4]

_SCRIPT = """
from onyx.utils.variable_functionality import (
    fetch_versioned_implementation,
    global_version,
    set_is_ee_based_on_env_variable,
)
set_is_ee_based_on_env_variable()
assert not global_version.is_ee_version()
fetch_versioned_implementation("onyx.server.settings.store", "load_settings")
print("community")
"""


def _run_without_ee(
    tmp_path: Path, env_overrides: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    for package in ("onyx", "shared_configs"):
        (tmp_path / package).symlink_to(BACKEND_DIR / package)
    env = {
        key: value
        for key, value in os.environ.items()
        if key
        not in (
            "ENABLE_PAID_ENTERPRISE_EDITION_FEATURES",
            "LICENSE_ENFORCEMENT_ENABLED",
        )
    }
    env.update(env_overrides)
    env["PYTHONPATH"] = str(tmp_path)
    return subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "env_overrides",
    [
        {},
        {
            "ENABLE_PAID_ENTERPRISE_EDITION_FEATURES": "false",
            "LICENSE_ENFORCEMENT_ENABLED": "false",
        },
    ],
)
def test_runs_as_community_edition_without_ee_code(
    tmp_path: Path, env_overrides: dict[str, str]
) -> None:
    result = _run_without_ee(tmp_path, env_overrides)
    assert result.returncode == 0, result.stderr
    assert "community" in result.stdout


@pytest.mark.parametrize(
    "env_overrides",
    [
        {"ENABLE_PAID_ENTERPRISE_EDITION_FEATURES": "true"},
        {"LICENSE_ENFORCEMENT_ENABLED": "true"},
    ],
)
def test_explicit_ee_request_fails_without_ee_code(
    tmp_path: Path, env_overrides: dict[str, str]
) -> None:
    result = _run_without_ee(tmp_path, env_overrides)
    assert result.returncode != 0
    assert "this build has no Enterprise Edition code" in result.stderr
