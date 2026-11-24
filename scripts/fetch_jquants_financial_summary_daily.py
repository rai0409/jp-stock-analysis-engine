import argparse
import contextlib
import errno
import fcntl
import json
import os
import re
import shutil
import stat
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

# PROJECT_DIR must be resolved at import time: it locates the ``.env`` file
# itself, so it can only come from the process environment (or the default).
PROJECT_DIR = Path(
    os.getenv(
        "JP_STOCK_ANALYSIS_PROJECT_DIR",
        str(Path(__file__).resolve().parents[1]),
    )
)
LOCAL_STORE = PROJECT_DIR / "data/jquants_price_store"


def get_universe_file() -> Path:
    # Read lazily so a value supplied through ``.env`` (loaded in main()) is honoured.
    return Path(
        os.getenv(
            "JP_STOCK_ANALYSIS_UNIVERSE_FILE",
            str(LOCAL_STORE / "topix1000_usable_tickers.csv"),
        )
    )


# Import-time snapshot, kept only for backward compatibility with importers.
# main() calls get_universe_file() after the environment has been loaded.
UNIVERSE_FILE = get_universe_file()

JST = timezone(timedelta(hours=9))

RAW_NAME = "financial_summary_raw.csv"
COVERAGE_NAME = "financial_summary_coverage.csv"
STATE_NAME = "financial_summary_auto_state.json"
RUN_LOG_NAME = "financial_summary_auto_runs.jsonl"

# Failure reasons that are never retried and therefore never carried over.
NON_RETRYABLE_REASONS = frozenset(
    {"OUT_OF_SUBSCRIPTION_DATE_RANGE", "PLAN_NOT_AVAILABLE"}
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Failure classes
#
# Each class has its own exit code, so a log reader (or systemd) can tell a missing
# drive from a corrupt file from a busy store. Exit code 1 stays what it always was for
# the two pre-existing configuration errors (missing API key, missing universe file).
# ---------------------------------------------------------------------------
EXIT_STORE_UNAVAILABLE = 3
EXIT_INPUT_UNREADABLE = 4
EXIT_PROGRESS_REGRESSION = 5
EXIT_STORE_CHANGED = 6
EXIT_BUSY = 75  # EX_TEMPFAIL: another process holds the store; try again later

# errno values meaning "the storage behind this path is not working", as opposed to
# "this file does not exist". On WSL these are what a detached or failing /mnt/<drive>
# looks like (EIO "Input/output error", ENODEV "No such device", ...).
UNAVAILABLE_ERRNOS = frozenset(
    code
    for code in (
        errno.EIO,
        errno.ENODEV,
        errno.ENXIO,
        errno.ESTALE,
        errno.ENOTCONN,
        errno.ETIMEDOUT,
        errno.EHOSTDOWN,
        getattr(errno, "EREMOTEIO", None),
    )
    if code is not None
)

ALLOW_REGRESSION_ENV = "JQUANTS_FIN_SUMMARY_ALLOW_PROGRESS_REGRESSION"
LOCK_FILE_ENV = "JQUANTS_FIN_SUMMARY_LOCK_FILE"


class StoreError(Exception):
    kind = "STORE_ERROR"
    exit_code = 1

    def __init__(self, message: str, kind: str | None = None):
        super().__init__(message)
        if kind:
            self.kind = kind


class StoreUnavailableError(StoreError):
    kind = "STORE_UNAVAILABLE"
    exit_code = EXIT_STORE_UNAVAILABLE


class StoreInputError(StoreError):
    kind = "INPUT_UNREADABLE"
    exit_code = EXIT_INPUT_UNREADABLE


class ProgressRegressionError(StoreError):
    kind = "PROGRESS_REGRESSION"
    exit_code = EXIT_PROGRESS_REGRESSION


class StoreChangedError(StoreError):
    kind = "STORE_CHANGED_DURING_RUN"
    exit_code = EXIT_STORE_CHANGED


class BusyError(StoreError):
    kind = "BUSY"
    exit_code = EXIT_BUSY


def oserror_to_store_error(exc: OSError, path, kind: str) -> StoreError:
    """Tell a failing mount apart from an ordinary unreadable file."""
    if exc.errno in UNAVAILABLE_ERRNOS:
        return StoreUnavailableError(
            f"{path}: {exc.strerror or exc} (errno {exc.errno}). The drive or mount behind "
            "this path is not working. Nothing was changed."
        )
    return StoreInputError(f"{path}: {exc}", kind=kind)


def stat_or_none(path, kind: str = "INPUT_UNREADABLE"):
    """os.stat() that returns None only for "does not exist".

    Path.exists() would also say False for some error codes and True/raise for others;
    here a mount failure is never mistaken for a missing file.
    """
    try:
        return os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError as exc:
        raise oserror_to_store_error(exc, path, kind) from exc


def check_store_reachable(path: Path) -> None:
    """Fail clearly for a dangling symlink or an unreachable directory.

    Creates nothing. A plainly missing local store is still created later, as before.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise oserror_to_store_error(exc, path, "STORE_UNAVAILABLE") from exc
    if stat.S_ISLNK(st.st_mode):
        target = os.path.realpath(path)
        if stat_or_none(target, "STORE_UNAVAILABLE") is None:
            raise StoreUnavailableError(
                f"{path} is a symlink to {target}, which does not exist. Is the drive behind "
                "it mounted? Nothing was changed."
            )
    else:
        stat_or_none(path, "STORE_UNAVAILABLE")


def describe_mount(path) -> str:
    """'<realpath> on <mountpoint> (<fstype>)' from /proc/self/mountinfo, for the journal."""
    try:
        real = os.path.realpath(path)
    except OSError:
        real = str(path)
    best_mp, best_fs = "/", "?"
    try:
        with open("/proc/self/mountinfo", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 7 or "-" not in parts:
                    continue
                mp = parts[4].replace("\\040", " ")
                inside = real == mp or real.startswith(mp.rstrip("/") + "/")
                if inside and len(mp) >= len(best_mp):
                    best_mp, best_fs = mp, parts[parts.index("-") + 1]
    except OSError:
        pass
    return f"{real} on {best_mp} ({best_fs})"


# ---------------------------------------------------------------------------
# Single-writer lock shared by the daily job and the backfill script.
# It lives under .cache/ in the project (local disk, git-ignored), not on the data
# drive, so it keeps working when that drive misbehaves.
# ---------------------------------------------------------------------------
def lock_file_path() -> Path:
    override = os.getenv(LOCK_FILE_ENV)
    if override:
        return Path(override)
    return PROJECT_DIR / ".cache" / "jquants" / "financial_summary_store.lock"


class StoreLock:
    def __init__(self, role: str, path: Path | None = None):
        self.role = role
        self.path = path or lock_file_path()
        self._fh = None

    def holder(self) -> str:
        try:
            text = self.path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return "unknown holder"
        return text or "unknown holder"

    def acquire(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fh = open(self.path, "a+", encoding="utf-8")
        except OSError as exc:
            raise StoreError(f"cannot open lock file {self.path}: {exc}", "LOCK_FAILED") from exc
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            fh.close()
            if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                raise BusyError(
                    f"another financial-summary process is running ({self.holder()}); "
                    f"lock file {self.path}. Nothing was changed."
                ) from exc
            raise StoreError(f"cannot lock {self.path}: {exc}", "LOCK_FAILED") from exc
        stamp = datetime.now(JST).isoformat(timespec="seconds")
        fh.seek(0)
        fh.truncate()
        fh.write(f"role={self.role} pid={os.getpid()} since={stamp}\n")
        fh.flush()
        self._fh = fh

    def release(self) -> None:
        if self._fh is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            self._fh.close()
        self._fh = None


# ---------------------------------------------------------------------------
# Writes: temp file in the same directory, flushed to disk, checked, then renamed over
# the target. A write that fails half way (EIO, ENODEV, disk full) leaves the old file
# untouched instead of a truncated one.
# ---------------------------------------------------------------------------
def _discard(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        _discard(tmp)
        raise oserror_to_store_error(exc, path, "WRITE_FAILED") from exc


def atomic_write_csv(df: pd.DataFrame, path: Path, verify: bool = True) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as f:
            df.to_csv(f, index=False)
            f.flush()
            os.fsync(f.fileno())
        if verify and len(df):
            check = pd.read_csv(tmp, dtype=str)
            if check.shape != df.shape:
                raise StoreInputError(
                    f"{path}: read-back check failed (wrote {df.shape}, read {check.shape}). "
                    "The existing file was not replaced.",
                    kind="WRITE_VERIFY_FAILED",
                )
        os.replace(tmp, path)
    except StoreError:
        _discard(tmp)
        raise
    except (OSError, pd.errors.ParserError) as exc:
        _discard(tmp)
        if isinstance(exc, OSError):
            raise oserror_to_store_error(exc, path, "WRITE_FAILED") from exc
        raise StoreInputError(f"{path}: {exc}", kind="WRITE_VERIFY_FAILED") from exc


def append_line(path: Path, line: str) -> None:
    try:
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
    except OSError as exc:
        raise oserror_to_store_error(exc, path, "WRITE_FAILED") from exc


def file_fingerprint(path: Path):
    st = stat_or_none(path)
    return None if st is None else (st.st_size, st.st_mtime_ns)


def get_external_candidates() -> list[str]:
    raw = os.getenv("JQUANTS_EXTERNAL_STORE_DIRS") or os.getenv("JQUANTS_EXTERNAL_STORE_DIR", "")
    return [item.strip() for item in raw.split(os.pathsep) if item.strip()]


# Import-time snapshot, kept only for backward compatibility with importers.
# choose_store_dir() re-reads the environment so that values loaded from ``.env``
# by main() are used.
EXTERNAL_CANDIDATES = get_external_candidates()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def is_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test = path / ".write_test"
        test.write_text("ok")
        test.unlink()
        return True
    except OSError as exc:
        # A failing drive must not silently push the data to a different store.
        if exc.errno in UNAVAILABLE_ERRNOS:
            raise oserror_to_store_error(exc, path, "STORE_UNAVAILABLE") from exc
        return False
    except Exception:
        return False


def choose_store_dir(candidates: list[str] | None = None) -> tuple[Path, str]:
    if candidates is None:
        candidates = get_external_candidates()
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw).expanduser()
        # Same fallback rule as before (a missing parent means "not this candidate"), but an
        # I/O error on the parent is reported instead of being read as "missing".
        if stat_or_none(p.parent, "STORE_UNAVAILABLE") is None:
            continue
        if is_writable_dir(p):
            return p, "external"

    try:
        LOCAL_STORE.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        if isinstance(exc, FileExistsError) and os.path.islink(LOCAL_STORE):
            raise StoreUnavailableError(
                f"{LOCAL_STORE} is a symlink to {os.path.realpath(LOCAL_STORE)}, which does not "
                "exist. Is the drive behind it mounted? Nothing was changed."
            ) from exc
        raise oserror_to_store_error(exc, LOCAL_STORE, "STORE_UNAVAILABLE") from exc
    return LOCAL_STORE, "local"


def ensure_external_has_existing_data(store: Path, store_type: str) -> None:
    if store_type != "external":
        return

    for name in [RAW_NAME, COVERAGE_NAME]:
        src = LOCAL_STORE / name
        dst = store / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def normalize_code_to_ticker(code):
    s = str(code).strip()
    if s.endswith("0") and len(s) == 5 and s[:4].isdigit():
        return s[:4]
    if s.endswith("0") and len(s) == 5 and s[:3].isdigit() and s[3].isalpha():
        return s[:4]
    return s


def extract_records(data):
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("data"), list):
        return data["data"]
    for value in data.values():
        if isinstance(value, list):
            return value
    return []


def parse_subscription_window(message: str):
    m = re.search(r"(\d{4}-\d{2}-\d{2})\s*~\s*(\d{4}-\d{2}-\d{2})", message)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def fetch_one_date(target_date: str, api_key: str, max_attempts: int = 3):
    headers = {
        "x-api-key": api_key,
        "User-Agent": "jp-stock-analysis-engine-fin-summary-auto-backfill",
    }
    url = "https://api.jquants.com/v2/fins/summary?" + urlencode({"date": target_date})
    req = urllib.request.Request(url, headers=headers)

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as res:
                data = json.loads(res.read().decode("utf-8"))
            return {
                "ok": True,
                "records": extract_records(data),
                "attempt": attempt,
                "reason": "OK",
                "error": None,
            }

        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            cov_start, cov_end = parse_subscription_window(body)

            if e.code == 400 and cov_start and cov_end:
                return {
                    "ok": False,
                    "records": [],
                    "attempt": attempt,
                    "reason": "OUT_OF_SUBSCRIPTION_DATE_RANGE",
                    "error": {
                        "type": "HTTPError",
                        "code": e.code,
                        "body": body[:1000],
                        "subscription_start": cov_start,
                        "subscription_end": cov_end,
                    },
                }

            if e.code == 403:
                return {
                    "ok": False,
                    "records": [],
                    "attempt": attempt,
                    "reason": "PLAN_NOT_AVAILABLE",
                    "error": {
                        "type": "HTTPError",
                        "code": e.code,
                        "body": body[:1000],
                    },
                }

            return {
                "ok": False,
                "records": [],
                "attempt": attempt,
                "reason": "HTTP_ERROR",
                "error": {
                    "type": "HTTPError",
                    "code": e.code,
                    "body": body[:1000],
                },
            }

        except Exception as e:
            last_error = {
                "type": type(e).__name__,
                "error": repr(e),
            }
            if attempt < max_attempts:
                time.sleep(30 * attempt)

    return {
        "ok": False,
        "records": [],
        "attempt": max_attempts,
        "reason": "NETWORK_OR_DNS_RETRY_EXHAUSTED",
        "error": last_error,
    }


def get_today_jst():
    return datetime.now(JST).date()


def failure_reason(info) -> str:
    if isinstance(info, dict):
        return info.get("reason") or info.get("type") or ""
    return ""


def load_retryable_failures(state: dict) -> dict:
    """Previously failed dates that must be retried, sorted by date."""
    previous = state.get("failed_dates_carryover")
    if not isinstance(previous, dict):
        return {}
    retryable = {
        d: info
        for d, info in previous.items()
        if failure_reason(info) not in NON_RETRYABLE_REASONS
    }
    return dict(sorted(retryable.items()))


def iso_date_or_none(value):
    if not isinstance(value, str) or not ISO_DATE_RE.match(value):
        return None
    try:
        datetime.fromisoformat(value)
    except ValueError:
        return None
    return value


def merge_failed_carryover(
    previous_carryover: dict, run_results: dict, failed_dates: dict
) -> dict:
    """Carryover for the next run.

    Dates processed in this run are decided by this run's outcome (success clears
    them, a retryable failure replaces them, a non-retryable outcome drops them).
    Dates not processed in this run (e.g. cut by the per-run limit or by an early
    stop) keep their previous entry.
    """
    processed = set(run_results)
    merged = {d: info for d, info in previous_carryover.items() if d not in processed}
    for d, info in failed_dates.items():
        if failure_reason(info) not in NON_RETRYABLE_REASONS:
            merged[d] = info
    return dict(sorted(merged.items()))


def compute_last_successful_query_date(previous, successful_query_dates):
    """Monotonic watermark: never moves backwards and only advances on API success."""
    candidates = [
        d for d in [previous, *successful_query_dates] if iso_date_or_none(d)
    ]
    return max(candidates) if candidates else None


# ---------------------------------------------------------------------------
# Reading the store: strict. A file that exists but cannot be read is an error, never
# "no progress yet" -- silently starting over is how an old date window gets fetched again.
# ---------------------------------------------------------------------------
def read_state(state_path: Path) -> dict:
    """{} only when the file does not exist."""
    if stat_or_none(state_path, "STATE_UNREADABLE") is None:
        return {}
    try:
        text = state_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise oserror_to_store_error(exc, state_path, "STATE_UNREADABLE") from exc
    except UnicodeDecodeError as exc:
        raise StoreInputError(f"{state_path}: {exc}", kind="STATE_UNREADABLE") from exc
    try:
        state = json.loads(text)
    except ValueError as exc:
        raise StoreInputError(
            f"{state_path}: not valid JSON ({exc}). The file was NOT reset. Restore it from a "
            "backup; or, only if you have checked that the raw CSV is complete, move the file "
            "aside yourself so progress is derived from the raw CSV.",
            kind="STATE_UNREADABLE",
        ) from exc
    if not isinstance(state, dict):
        raise StoreInputError(
            f"{state_path}: expected a JSON object, found {type(state).__name__}",
            kind="STATE_UNREADABLE",
        )
    return state


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Empty frame when the file is missing or holds no rows; an error when it is unreadable."""
    if stat_or_none(path, "RAW_UNREADABLE") is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        # A previous run that saved no rows leaves an empty file behind.
        return pd.DataFrame()
    except OSError as exc:
        raise oserror_to_store_error(exc, path, "RAW_UNREADABLE") from exc
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        raise StoreInputError(
            f"{path}: cannot be parsed as CSV ({exc}). It was not modified.",
            kind="RAW_UNREADABLE",
        ) from exc


def latest_disc_date_from_df(df: pd.DataFrame):
    if len(df) == 0 or "DiscDate" not in df.columns:
        return None
    s = df["DiscDate"].dropna().astype(str)
    s = s[s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    if len(s) == 0:
        return None
    return s.max()


def get_latest_disc_date(raw_path: Path):
    return latest_disc_date_from_df(read_raw_csv(raw_path))


def read_last_run_log_entry(path: Path):
    """Last parsable JSON object of the run log (a torn last line is skipped), or None."""
    if stat_or_none(path, "RUN_LOG_UNREADABLE") is None:
        return None
    last = None
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    last = obj
    except OSError as exc:
        raise oserror_to_store_error(exc, path, "RUN_LOG_UNREADABLE") from exc
    return last


@dataclass
class StoreInputs:
    state: dict
    raw: pd.DataFrame
    raw_fingerprint: tuple | None
    last_log: dict | None


def load_store_inputs(raw_path: Path, state_path: Path, run_log_path: Path | None = None):
    fingerprint = file_fingerprint(raw_path)
    return StoreInputs(
        state=read_state(state_path),
        raw=read_raw_csv(raw_path),
        raw_fingerprint=fingerprint,
        last_log=read_last_run_log_entry(run_log_path) if run_log_path else None,
    )


def _parse_timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def find_progress_problems(inputs: StoreInputs) -> list[str]:
    """Signs that the state or raw CSV was rolled back or truncated.

    The run log is append-only and the state file is written together with its last line,
    so the state can never legitimately be older than the log, and the raw CSV can never
    have fewer rows / an older newest disclosure than the state recorded.
    """
    state, raw, last_log = inputs.state, inputs.raw, inputs.last_log or {}
    problems = []

    state_ts = _parse_timestamp(state.get("last_run_at_jst"))
    log_ts = _parse_timestamp(last_log.get("last_run_at_jst"))
    if state_ts and log_ts and state_ts.tzinfo and log_ts.tzinfo and state_ts < log_ts:
        problems.append(
            "the state file is older than the last run-log entry "
            f"(state {state.get('last_run_at_jst')} < log {last_log.get('last_run_at_jst')})"
        )

    state_wm = iso_date_or_none(state.get("last_successful_query_date"))
    log_wm = iso_date_or_none(last_log.get("last_successful_query_date"))
    if log_wm and (state_wm is None or state_wm < log_wm):
        problems.append(
            f"last_successful_query_date in the state ({state_wm}) is behind the last run-log "
            f"entry ({log_wm})"
        )

    rows_saved = state.get("rows_saved_total")
    if isinstance(rows_saved, int) and not isinstance(rows_saved, bool) and len(raw) < rows_saved:
        problems.append(
            f"the raw CSV has {len(raw)} rows but the state recorded {rows_saved}"
        )

    saved_latest = iso_date_or_none(state.get("latest_disc_date_saved"))
    raw_latest = latest_disc_date_from_df(raw)
    if saved_latest and (raw_latest is None or raw_latest < saved_latest):
        problems.append(
            f"the newest disclosure in the raw CSV ({raw_latest}) is older than the state "
            f"recorded ({saved_latest})"
        )
    return problems


def compute_anchor(latest_disc_date, last_successful_query_date):
    candidates = []
    for d in [latest_disc_date, last_successful_query_date]:
        if d:
            try:
                candidates.append(datetime.fromisoformat(d).date())
            except Exception:
                pass
    return max(candidates) if candidates else None


def build_target_dates_from(state: dict, raw: pd.DataFrame, today=None):
    today = today or get_today_jst()
    pending = list(load_retryable_failures(state))
    anchor = compute_anchor(
        latest_disc_date_from_df(raw), state.get("last_successful_query_date")
    )
    if anchor:
        start = anchor + timedelta(days=1)
    else:
        start = datetime.fromisoformat("2024-04-05").date()

    incremental = []
    d = start
    while d <= today:
        incremental.append(d.isoformat())
        d += timedelta(days=1)

    seen = set()
    result = []
    for d in pending + incremental:
        if d not in seen:
            seen.add(d)
            result.append(d)
    return result


def build_target_dates(raw_path: Path, state_path: Path, today=None):
    return build_target_dates_from(read_state(state_path), read_raw_csv(raw_path), today)


def load_universe(path: Path) -> pd.DataFrame:
    try:
        universe = pd.read_csv(path, dtype=str)
    except OSError as exc:
        raise oserror_to_store_error(exc, path, "UNIVERSE_UNREADABLE") from exc
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise StoreInputError(f"{path}: {exc}", kind="UNIVERSE_INVALID") from exc
    if "ticker" not in universe.columns:
        raise StoreInputError(
            f"{path}: required column 'ticker' is missing (columns: {list(universe.columns)})",
            kind="UNIVERSE_INVALID",
        )
    if len(universe) == 0:
        raise StoreInputError(f"{path}: the universe file has no rows", kind="UNIVERSE_INVALID")
    if universe["ticker"].isna().any():
        raise StoreInputError(
            f"{path}: {int(universe['ticker'].isna().sum())} row(s) have an empty ticker",
            kind="UNIVERSE_INVALID",
        )
    universe["ticker"] = universe["ticker"].astype(str)
    return universe


MAX_DATES_ENV = "JQUANTS_FIN_SUMMARY_MAX_DATES_PER_RUN"


def print_dry_run(
    *, store, store_type, universe_file, universe, inputs, problems, allowed, target_dates, lock
):
    state, raw = inputs.state, inputs.raw
    print("== dry-run: no API call, nothing is written ==")
    print("store:", store, f"({store_type})")
    print("store_mount:", describe_mount(store))
    print("universe:", universe_file, f"rows={len(universe)}")
    print(
        "state:",
        {
            "present": bool(state),
            "last_run_at_jst": state.get("last_run_at_jst"),
            "last_successful_query_date": state.get("last_successful_query_date"),
            "latest_disc_date_saved": state.get("latest_disc_date_saved"),
            "rows_saved_total": state.get("rows_saved_total"),
            "failed_dates_carryover": sorted(state.get("failed_dates_carryover") or {}),
        },
    )
    raw_info = {"rows": int(len(raw)), "columns": int(len(raw.columns))}
    if len(raw) and "query_date" in raw.columns:
        raw_info["min_query_date"] = raw["query_date"].min()
        raw_info["max_query_date"] = raw["query_date"].max()
    raw_info["max_DiscDate"] = latest_disc_date_from_df(raw)
    print("raw:", raw_info)
    last_log = inputs.last_log or {}
    print(
        "run_log_last:",
        {
            "last_run_at_jst": last_log.get("last_run_at_jst"),
            "last_successful_query_date": last_log.get("last_successful_query_date"),
        },
    )
    anchor = compute_anchor(
        latest_disc_date_from_df(raw), state.get("last_successful_query_date")
    )
    print("anchor (next start = anchor + 1 day):", anchor)
    if problems:
        verdict = (
            "PROBLEMS (override active, would continue)" if allowed else "PROBLEMS (run would stop)"
        )
        print("progress_check:", verdict)
        for p in problems:
            print("  -", p)
    else:
        print("progress_check: OK")
    cap = int(os.getenv(MAX_DATES_ENV, "20"))
    shown = target_dates[:cap]
    if shown:
        print(f"target_dates (cap {cap}): {shown[0]} .. {shown[-1]} ({len(shown)} dates)")
    else:
        print(f"target_dates (cap {cap}): none")
    print("lock:", lock)


def parse_args(argv):
    ap = argparse.ArgumentParser(
        description="Daily J-Quants Financial Summary fetch (run without options as before)."
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print what this process sees and which dates it would fetch; no API call, no write",
    )
    ap.add_argument(
        "--allow-progress-regression",
        action="store_true",
        help=(
            "continue even if the state/raw CSV look rolled back "
            f"(or set {ALLOW_REGRESSION_ENV}=1)"
        ),
    )
    return ap.parse_args(argv)


def run(args) -> int:
    load_env_file(PROJECT_DIR / ".env")

    api_key = os.getenv("JQUANTS_API_KEY")
    if not api_key and not args.dry_run:
        raise SystemExit("JQUANTS_API_KEY: MISSING")

    universe_file = get_universe_file()
    check_store_reachable(universe_file.parent)
    if stat_or_none(universe_file, "UNIVERSE_UNREADABLE") is None:
        print(
            "hint: the universe file is gone. If financial_summary_coverage.csv is intact, "
            "scripts/recover_jquants_universe.py can rebuild it (it never overwrites).",
            file=sys.stderr,
        )
        raise SystemExit(f"universe file not found: {universe_file}")

    store, store_type = choose_store_dir()
    check_store_reachable(store)
    ensure_external_has_existing_data(store, store_type)

    lock = StoreLock("daily")
    lock_status = "acquired"
    try:
        lock.acquire()
    except BusyError as exc:
        if not args.dry_run:
            raise
        lock_status = f"HELD by another process ({lock.holder()})"
        print(f"BUSY (dry-run continues read-only): {exc}", file=sys.stderr)
    try:
        return _run_locked(args, api_key, universe_file, store, store_type, lock_status)
    finally:
        lock.release()


def _run_locked(args, api_key, universe_file, store, store_type, lock_status) -> int:
    raw_out = store / RAW_NAME
    coverage_out = store / COVERAGE_NAME
    state_out = store / STATE_NAME
    run_log_out = store / RUN_LOG_NAME

    universe = load_universe(universe_file)
    usable = set(universe["ticker"])

    inputs = load_store_inputs(raw_out, state_out, run_log_out)
    previous_state = inputs.state
    previous_carryover = load_retryable_failures(previous_state)

    problems = find_progress_problems(inputs)
    allowed = args.allow_progress_regression or os.getenv(ALLOW_REGRESSION_ENV) == "1"
    if problems and not allowed:
        if not args.dry_run:
            raise ProgressRegressionError(
                "the stored progress looks rolled back or truncated, so fetching would repeat "
                "old dates and could overwrite newer data: "
                + "; ".join(problems)
                + f". Nothing was changed. Restore the store from a backup, or if you have "
                f"verified it is intended, set {ALLOW_REGRESSION_ENV}=1."
            )
    elif problems:
        print("WARNING: progress check overridden:", "; ".join(problems), file=sys.stderr)

    target_dates = build_target_dates_from(previous_state, inputs.raw)

    max_dates_per_run = int(os.getenv(MAX_DATES_ENV, "20"))
    target_dates = target_dates[:max_dates_per_run]

    if args.dry_run:
        print_dry_run(
            store=store,
            store_type=store_type,
            universe_file=universe_file,
            universe=universe,
            inputs=inputs,
            problems=problems,
            allowed=allowed,
            target_dates=target_dates,
            lock=lock_status,
        )
        return 0 if (allowed or not problems) else ProgressRegressionError.exit_code

    all_rows = []
    failed_dates = {}
    skipped_dates = {}
    empty_dates = []
    run_results = {}
    subscription_start = None
    subscription_end = None

    print("store:", store)
    print("store_type:", store_type)
    print("store_mount:", describe_mount(store))
    print("target_dates:", target_dates)

    for target_date in target_dates:
        print("fetch", target_date)

        result = fetch_one_date(target_date, api_key, max_attempts=3)
        run_results[target_date] = {
            "ok": result["ok"],
            "attempt": result["attempt"],
            "reason": result["reason"],
            "raw_count": len(result["records"]),
            "error": result["error"],
        }

        if not result["ok"]:
            err = result["error"] or {}
            reason = result["reason"]

            if reason == "OUT_OF_SUBSCRIPTION_DATE_RANGE":
                subscription_start = err.get("subscription_start")
                subscription_end = err.get("subscription_end")
                skipped_dates[target_date] = {
                    "reason": reason,
                    "detail": err,
                }
                print("  skipped:", reason, subscription_start, subscription_end)
                break

            failed_dates[target_date] = {
                "reason": reason,
                "detail": err,
            }
            print("  failed:", reason, err)
            time.sleep(20)
            continue

        records = result["records"]

        if not records:
            empty_dates.append(target_date)
            print("  records: 0")
            time.sleep(20)
            continue

        usable_added = 0
        for r in records:
            r = dict(r)
            r["query_date"] = target_date
            r["ticker"] = normalize_code_to_ticker(r.get("Code", ""))
            if r["ticker"] in usable:
                all_rows.append(r)
                usable_added += 1

        print("  raw_records:", len(records), "usable_added:", usable_added)
        time.sleep(20)

    new_df = pd.DataFrame(all_rows)

    # The raw CSV must still be what this run started from. A writer that does not take the
    # lock (an older copy of this script, a manual edit) must not be silently overwritten.
    if file_fingerprint(raw_out) != inputs.raw_fingerprint:
        raise StoreChangedError(
            f"{raw_out} changed while this run was fetching. Nothing was written; run again."
        )

    old_df = inputs.raw
    if len(old_df.columns) == 0:
        df = new_df
    else:
        df = pd.concat([old_df, new_df], ignore_index=True)

    if len(df):
        dedupe_keys = [c for c in ["DiscNo", "Code", "DiscDate", "DocType"] if c in df.columns]
        if dedupe_keys:
            df = df.drop_duplicates(dedupe_keys)
        atomic_write_csv(df, raw_out)
    else:
        atomic_write_csv(pd.DataFrame(), raw_out, verify=False)

    got = set(df["ticker"].astype(str)) if len(df) and "ticker" in df.columns else set()

    coverage = universe.copy()
    coverage["has_any_financial_summary"] = coverage["ticker"].isin(got)
    atomic_write_csv(coverage, coverage_out)

    failed_dates_carryover = merge_failed_carryover(
        previous_carryover, run_results, failed_dates
    )

    successful_query_dates = [
        d for d, r in run_results.items()
        if r.get("ok") is True
    ]

    last_successful_query_date = compute_last_successful_query_date(
        previous_state.get("last_successful_query_date"),
        successful_query_dates,
    )

    state = {
        "endpoint": "/v2/fins/summary",
        "query_mode": "date",
        "last_run_at_jst": datetime.now(JST).isoformat(),
        "store_type": store_type,
        "store_dir": str(store),
        "target_dates_last_run": target_dates,
        "successful_query_dates_last_run": successful_query_dates,
        "last_successful_query_date": last_successful_query_date,
        "rows_added_last_run_before_dedup": int(len(new_df)),
        "rows_saved_total": int(len(df)),
        "unique_tickers_saved": int(len(got)),
        "coverage_missing_count": int(len(usable - got)),
        "latest_disc_date_saved": latest_disc_date_from_df(df),
        "failed_dates_last_run": failed_dates,
        "failed_dates_count_last_run": int(len(failed_dates)),
        "failed_dates_carryover": failed_dates_carryover,
        "failed_dates_carryover_count": int(len(failed_dates_carryover)),
        "skipped_dates_last_run": skipped_dates,
        "skipped_dates_count_last_run": int(len(skipped_dates)),
        "empty_dates_last_run": empty_dates,
        "run_results": run_results,
        "subscription_start_detected": subscription_start,
        "subscription_end_detected": subscription_end,
        "output": str(raw_out),
        "coverage_output": str(coverage_out),
        "api_key_status": "PRESENT",
        "secret_included": False,
        "note": (
            "Only raw J-Quants fields are saved. "
            "No PER/PBR/ROE/margins/growth/yield are calculated."
        ),
    }

    atomic_write_text(state_out, json.dumps(state, ensure_ascii=False, indent=2))
    append_line(run_log_out, json.dumps(state, ensure_ascii=False) + "\n")

    print("\n== done ==")
    print("store_type:", store_type)
    print("store_dir:", store)
    print("rows_added_last_run_before_dedup:", len(new_df))
    print("rows_saved_total:", len(df))
    print("unique_tickers_saved:", len(got))
    print("coverage_missing_count:", len(usable - got))
    print("latest_disc_date_saved:", state["latest_disc_date_saved"])
    print("failed_dates_count_last_run:", len(failed_dates))
    print("failed_dates_carryover_count:", len(failed_dates_carryover))
    print("skipped_dates_count_last_run:", len(skipped_dates))
    print("state:", state_out)
    return 0


def main(argv=None) -> int:
    args = parse_args([] if argv is None else argv)
    try:
        return run(args)
    except StoreError as exc:
        print(f"{exc.kind}: {exc}", file=sys.stderr)
        raise SystemExit(exc.exit_code) from None


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
