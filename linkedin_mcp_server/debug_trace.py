"""Best-effort trace capture with on-error retention."""

from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Literal

from linkedin_mcp_server.common_utils import secure_mkdir, slugify_fragment
from linkedin_mcp_server.session_state import auth_root_dir, get_source_profile_dir

TraceMode = Literal["off", "on_error", "always"]

_TRACE_COUNTER = itertools.count(1)
_TRACE_DIR: Path | None = None
_TRACE_KEEP = False
_EXPLICIT_TRACE_DIR = False

# Bounded retention for the shared trace root. Retained runs (bug-report
# diagnostics) and runs leaked by crashed sessions are pruned once they age
# out or fall beyond the recent-run cap, so trace-runs/ cannot grow without
# limit across sessions.
_PRUNE_MAX_AGE_DAYS = 14
_PRUNE_KEEP_RECENT_RUNS = 20


def _trace_mode() -> TraceMode:
    raw = os.getenv("LINKEDIN_TRACE_MODE", "").strip().lower()
    if raw in {"off", "false", "0", "no"}:
        return "off"
    if raw in {"always", "keep", "persist"}:
        return "always"
    return "on_error"


def _trace_root() -> Path:
    source_profile = _safe_source_profile_dir()
    root = auth_root_dir(source_profile) / "trace-runs"
    secure_mkdir(root)
    return root


def trace_enabled() -> bool:
    return (
        bool(os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip())
        or _trace_mode() != "off"
    )


def get_trace_dir() -> Path | None:
    global _TRACE_DIR, _EXPLICIT_TRACE_DIR

    explicit = os.getenv("LINKEDIN_DEBUG_TRACE_DIR", "").strip()
    if explicit:
        _EXPLICIT_TRACE_DIR = True
        if _TRACE_DIR is None:
            _TRACE_DIR = Path(explicit).expanduser().resolve()
        return _TRACE_DIR

    if _trace_mode() == "off":
        return None

    if _TRACE_DIR is None:
        root = _trace_root()
        _TRACE_DIR = Path(
            tempfile.mkdtemp(
                prefix="run-",
                dir=root,
            )
        ).resolve()
        _prune_stale_trace_runs(root, current=_TRACE_DIR)
    return _TRACE_DIR


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _run_last_activity(path: Path) -> float:
    """Newest mtime of the run dir or anything inside it.

    A directory's own mtime only changes when entries are added or removed,
    so a long-lived process appending to ``trace.jsonl`` would look stale by
    dir mtime alone. Any process still writing traces keeps its newest file
    fresh, which shields concurrently active runs from age-based pruning.
    """
    newest = _mtime(path)
    try:
        for child in path.rglob("*"):
            newest = max(newest, _mtime(child))
    except OSError:
        pass
    return newest


def _prune_stale_trace_runs(root: Path, *, current: Path | None) -> None:
    """Best-effort pruning of old ``run-*`` dirs under the shared trace root.

    Keeps the ``_PRUNE_KEEP_RECENT_RUNS`` most recently active runs and
    drops any run whose last write is older than ``_PRUNE_MAX_AGE_DAYS``.
    The current session's run dir is never touched. Explicit
    ``LINKEDIN_DEBUG_TRACE_DIR`` locations are user-managed, live outside
    this root, and are never pruned. Tracing is best-effort by design: if a
    pruned run belonged to a process that is somehow still alive, its next
    ``record_page_trace`` recreates the dir and keeps writing.
    """
    try:
        candidates = [
            path
            for path in root.glob("run-*")
            if path.is_dir() and path.resolve() != current
        ]
    except OSError:
        return

    by_activity = sorted(
        ((_run_last_activity(path), path) for path in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    now = time.time()
    max_age_seconds = _PRUNE_MAX_AGE_DAYS * 86400
    for index, (last_activity, path) in enumerate(by_activity):
        if index < _PRUNE_KEEP_RECENT_RUNS and now - last_activity <= max_age_seconds:
            continue
        try:
            shutil.rmtree(path)
        except OSError:
            continue


def mark_trace_for_retention() -> Path | None:
    global _TRACE_KEEP
    trace_dir = get_trace_dir()
    if trace_dir is not None:
        secure_mkdir(trace_dir)
        _TRACE_KEEP = True
    return trace_dir


def should_keep_traces() -> bool:
    return _EXPLICIT_TRACE_DIR or _TRACE_KEEP or _trace_mode() == "always"


def cleanup_trace_dir() -> None:
    global _TRACE_DIR, _TRACE_KEEP, _EXPLICIT_TRACE_DIR

    trace_dir = _TRACE_DIR
    if trace_dir is None or should_keep_traces():
        return
    try:
        shutil.rmtree(trace_dir)
    except OSError:
        return
    _TRACE_DIR = None
    _TRACE_KEEP = False
    _EXPLICIT_TRACE_DIR = False


def reset_trace_state_for_testing() -> None:
    global _TRACE_COUNTER, _TRACE_DIR, _TRACE_KEEP, _EXPLICIT_TRACE_DIR
    _TRACE_COUNTER = itertools.count(1)
    _TRACE_DIR = None
    _TRACE_KEEP = False
    _EXPLICIT_TRACE_DIR = False


def _slugify_step(step: str) -> str:
    return slugify_fragment(step)


def _safe_source_profile_dir() -> Path:
    try:
        return get_source_profile_dir()
    except Exception:
        return Path("~/.linkedin-mcp/profile").expanduser()


async def record_page_trace(
    page: Any, step: str, *, extra: dict[str, Any] | None = None
) -> None:
    """Persist a screenshot and basic page state when trace capture is enabled."""
    trace_dir = get_trace_dir()
    if trace_dir is None:
        return

    secure_mkdir(trace_dir)
    screenshot_dir = trace_dir / "screens"
    secure_mkdir(screenshot_dir)
    step_id = next(_TRACE_COUNTER)
    slug = _slugify_step(step) or "step"

    try:
        title = await page.title()
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        title = f"<error: {exc}>"

    try:
        body_text = await page.evaluate("() => document.body?.innerText || ''")
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        body_text = f"<error: {exc}>"

    if not isinstance(body_text, str):
        body_text = ""

    try:
        remember_me = (await page.locator("#rememberme-div").count()) > 0
    except Exception:  # pragma: no cover - best effort diagnostics
        remember_me = False

    try:
        cookies = await page.context.cookies()
    except Exception:  # pragma: no cover - best effort diagnostics
        cookies = []

    linkedin_cookie_names = sorted(
        {
            cookie["name"]
            for cookie in cookies
            if "linkedin.com" in cookie.get("domain", "")
        }
    )

    screenshot_path = screenshot_dir / f"{step_id:03d}-{slug}.png"
    screenshot: str | None = None
    try:
        await page.screenshot(path=str(screenshot_path), full_page=True)
        screenshot = str(screenshot_path)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        screenshot = f"<error: {exc}>"

    payload = {
        "step_id": step_id,
        "step": step,
        "url": getattr(page, "url", ""),
        "title": title,
        "remember_me": remember_me,
        "body_length": len(body_text),
        "body_marker": " ".join(body_text.split())[:200],
        "linkedin_cookie_names": linkedin_cookie_names,
        "screenshot": screenshot,
        "extra": extra or {},
    }

    trace_jsonl = trace_dir / "trace.jsonl"
    try:
        fd = os.open(str(trace_jsonl), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    except FileExistsError:
        pass
    with trace_jsonl.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=True) + "\n")
