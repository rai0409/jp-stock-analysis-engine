from __future__ import annotations

import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path


SCRIPT = Path("scripts/run_jquants_daily_bars_incremental_safe.sh").resolve()


def _write_store(tmp_path: Path) -> tuple[Path, Path, Path]:
    store = tmp_path / "store"
    store.mkdir(exist_ok=True)
    price_file = store / "prices_daily_bars.csv"
    price_file.write_text(
        "ticker,date,adj_close,turnover_value,volume\n"
        "7203,2026-03-24,100,1000,10\n",
        encoding="utf-8",
    )
    universe = store / "topix1000_usable_tickers.csv"
    universe.write_text("ticker\n7203\n", encoding="utf-8")
    return store, price_file, universe


def _run_wrapper(tmp_path: Path, fetch_script: Path, *, today: str = "2026-07-02"):
    store, price_file, universe = _write_store(tmp_path)
    env = {
        **os.environ,
        "PROJECT_DIR": str(Path.cwd()),
        "STORE_DIR": str(store),
        "PRICE_FILE": str(price_file),
        "UNIVERSE_FILE": str(universe),
        "CACHE_DIR": str(tmp_path / "cache"),
        "FETCH_BIN": str(fetch_script),
        "SAFE_STATE_FILE": str(store / "daily_bars_safe_state.json"),
        "LOG_DIR": str(tmp_path / "logs"),
        "JQUANTS_DAILY_BARS_SAFE_TODAY": today,
        "JQUANTS_DAILY_BARS_SAFE_PROBE_TTL_HOURS": "20",
    }
    return subprocess.run(
        [str(SCRIPT)],
        cwd=Path.cwd(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_safe_wrapper_converts_coverage_failure_to_wait_exit_zero(tmp_path):
    store = tmp_path / "store"
    fetch = tmp_path / "fake_fetch.sh"
    fetch.write_text(
        "#!/usr/bin/env bash\n"
        "cat > \"$STORE_DIR/daily_bars_fetch_state.json\" <<'JSON'\n"
        "{\"failed_dates\":{\"2026-03-25\":\"INVALID_ARGUMENT: date outside plan coverage\"}}\n"
        "JSON\n"
        "echo 'Daily bars store updated: rows=1, tickers=1, added=0, failed_dates=1'\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fetch.chmod(0o755)

    result = _run_wrapper(tmp_path, fetch)

    assert result.returncode == 0
    assert "coverage_wait" in result.stdout
    safe_state = (store / "daily_bars_safe_state.json").read_text(encoding="utf-8")
    assert "coverage_wait" in safe_state
    assert "2026-03-25" in safe_state


def test_safe_wrapper_keeps_unexpected_failure_exit_two(tmp_path):
    fetch = tmp_path / "fake_fetch.sh"
    fetch.write_text(
        "#!/usr/bin/env bash\n"
        "cat > \"$STORE_DIR/daily_bars_fetch_state.json\" <<'JSON'\n"
        "{\"failed_dates\":{\"2026-03-25\":\"Unauthorized: API key rejected\"}}\n"
        "JSON\n"
        "echo 'Daily bars store updated: failed_dates=1'\n"
        "exit 2\n",
        encoding="utf-8",
    )
    fetch.chmod(0o755)

    result = _run_wrapper(tmp_path, fetch)

    assert result.returncode == 2
    assert "unexpected_fetch_failure" in result.stderr


def test_safe_wrapper_recent_coverage_wait_skips_probe(tmp_path):
    fetch = tmp_path / "fake_fetch.sh"
    fetch.write_text("#!/usr/bin/env bash\necho should-not-run\nexit 2\n", encoding="utf-8")
    fetch.chmod(0o755)

    first = _run_wrapper(tmp_path, fetch)
    assert first.returncode == 2

    store = tmp_path / "store"
    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    (store / "daily_bars_safe_state.json").write_text(
        "{\n"
        f'  "generated_at": "{generated_at}",\n'
        '  "status": "coverage_wait",\n'
        '  "candidate_start_date": "2026-03-25",\n'
        '  "date_max": "2026-03-24"\n'
        "}\n",
        encoding="utf-8",
    )
    second = _run_wrapper(tmp_path, fetch)
    assert second.returncode == 0
    assert "recent_coverage_wait" in second.stdout
    assert "should-not-run" not in second.stdout
