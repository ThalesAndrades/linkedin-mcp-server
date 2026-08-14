import json
import os
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from linkedin_mcp_server.debug_trace import (
    _safe_source_profile_dir,
    cleanup_trace_dir,
    get_trace_dir,
    mark_trace_for_retention,
    record_page_trace,
    reset_trace_state_for_testing,
)


def setup_function():
    reset_trace_state_for_testing()


def teardown_function():
    reset_trace_state_for_testing()


def test_get_trace_dir_creates_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    trace_dir = get_trace_dir()

    assert trace_dir is not None
    assert trace_dir.exists()
    assert "trace-runs" in str(trace_dir)


def test_cleanup_trace_dir_removes_ephemeral_dir_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = get_trace_dir()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert not trace_dir.exists()


def test_mark_trace_for_retention_keeps_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    trace_dir = mark_trace_for_retention()
    assert trace_dir is not None

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_explicit_trace_dir_is_preserved(monkeypatch, tmp_path):
    trace_dir = tmp_path / "explicit-trace"
    monkeypatch.setenv("LINKEDIN_DEBUG_TRACE_DIR", str(trace_dir))

    resolved = get_trace_dir()
    assert resolved == trace_dir
    trace_dir.mkdir(parents=True, exist_ok=True)

    cleanup_trace_dir()

    assert trace_dir.exists()


def test_trace_mode_off_disables_trace_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setenv("LINKEDIN_TRACE_MODE", "off")

    assert get_trace_dir() is None


@pytest.mark.asyncio
async def test_reset_trace_state_resets_step_counter(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))

    page = MagicMock()
    page.url = "https://www.linkedin.com/feed/"
    page.title = AsyncMock(return_value="LinkedIn")
    page.evaluate = AsyncMock(return_value="Feed")
    locator = MagicMock()
    locator.count = AsyncMock(return_value=0)
    page.locator = MagicMock(return_value=locator)
    page.context.cookies = AsyncMock(return_value=[])
    page.screenshot = AsyncMock()

    await record_page_trace(page, "first")
    trace_dir = get_trace_dir()
    assert trace_dir is not None
    first_payload = json.loads((trace_dir / "trace.jsonl").read_text().splitlines()[0])
    assert first_payload["step_id"] == 1

    reset_trace_state_for_testing()
    monkeypatch.setenv("USER_DATA_DIR", str((tmp_path / "second") / "profile"))

    await record_page_trace(page, "first-again")
    second_trace_dir = get_trace_dir()
    assert second_trace_dir is not None
    second_payload = json.loads(
        (second_trace_dir / "trace.jsonl").read_text().splitlines()[0]
    )
    assert second_payload["step_id"] == 1


def test_safe_source_profile_dir_ignores_generic_env_fallback(monkeypatch):
    monkeypatch.setenv("USER_DATA_DIR", "/tmp/unrelated-user-data")
    monkeypatch.setattr(
        "linkedin_mcp_server.debug_trace.get_source_profile_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    assert _safe_source_profile_dir() == Path("~/.linkedin-mcp/profile").expanduser()


def _make_stale_run(root: Path, name: str, *, age_seconds: float) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    stale = time.time() - age_seconds
    os.utime(run_dir, (stale, stale))
    return run_dir


def test_prune_removes_runs_older_than_max_age(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    root = tmp_path / "trace-runs"
    old_run = _make_stale_run(root, "run-old", age_seconds=20 * 86400)
    fresh_run = _make_stale_run(root, "run-fresh", age_seconds=60)

    current = get_trace_dir()

    assert current is not None and current.exists()
    assert not old_run.exists()
    assert fresh_run.exists()


def test_prune_removes_runs_beyond_recent_cap(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    monkeypatch.setattr("linkedin_mcp_server.debug_trace._PRUNE_KEEP_RECENT_RUNS", 2)
    root = tmp_path / "trace-runs"
    runs = [
        # Oldest first; all younger than the age cap.
        _make_stale_run(root, f"run-{idx}", age_seconds=3600 * (4 - idx))
        for idx in range(4)
    ]

    current = get_trace_dir()

    assert current is not None and current.exists()
    # The two most recent survive; the two oldest are pruned.
    assert [run.exists() for run in runs] == [False, False, True, True]


def test_prune_keeps_run_with_recent_writes_despite_stale_dir_mtime(
    monkeypatch, tmp_path
):
    """A concurrently active process keeps its trace.jsonl fresh even though
    the run dir's own mtime never changes after creation. Age-based pruning
    must key on the newest file inside the run, not the dir mtime."""
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    root = tmp_path / "trace-runs"
    active_run = _make_stale_run(root, "run-active", age_seconds=20 * 86400)
    trace_file = active_run / "trace.jsonl"
    trace_file.write_text("{}\n")
    stale = time.time() - 20 * 86400
    os.utime(active_run, (stale, stale))

    current = get_trace_dir()

    assert current is not None and current.exists()
    assert active_run.exists()
    assert trace_file.exists()


def test_prune_skips_shared_root_when_explicit_dir_is_set(monkeypatch, tmp_path):
    monkeypatch.setenv("USER_DATA_DIR", str(tmp_path / "profile"))
    root = tmp_path / "trace-runs"
    old_run = _make_stale_run(root, "run-old", age_seconds=20 * 86400)
    monkeypatch.setenv("LINKEDIN_DEBUG_TRACE_DIR", str(tmp_path / "explicit"))

    get_trace_dir()

    assert old_run.exists()
