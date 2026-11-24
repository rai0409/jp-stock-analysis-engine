#!/usr/bin/env python3
"""Rebuild topix1000_usable_tickers.csv from financial_summary_coverage.csv -- if, and only
if, the rebuild can be shown to be equivalent. Standard library only; read-only unless it
creates the missing universe file.

Why this is legitimate (and where it stops):
  * The daily job builds the coverage file with ``coverage = universe.copy()`` and then adds
    one column, ``has_any_financial_summary`` (see fetch_jquants_financial_summary_daily.py).
    So the coverage file is the universe, row for row and column for column, plus that flag.
  * The consumers need exactly the columns ticker, name, date, new_index_category
    (jquants_listed_master._read_universe) and a ticker column (load_universe_file).
  * What cannot be proven: that the lost file had no other difference (byte layout, quoting,
    values pandas turns into empty cells), or that nothing changed between the last daily
    run that wrote the coverage file and the loss. The result is therefore a
    *content-equivalent reconstruction*. If you know the SHA-256 of the original (for example
    the "universe_sha256" recorded in the other financial-summary pipeline's state file),
    pass --expect-sha256: a match proves byte identity, a mismatch writes nothing.

Rules:
  * Never overwrites. An existing output file is left alone (identical content -> "already
    restored", different content -> refusal).
  * Every check must pass before anything is written; --dry-run runs the checks only.
  * A provenance file <output>.recovery_<timestamp>.json is written next to the result.

Exit codes: 0 ok, 2 validation failed, 3 refused (output exists with different content) or
store problem, 4 I/O error.

Examples:
    python scripts/recover_jquants_universe.py --dry-run
    python scripts/recover_jquants_universe.py --expect-rows 974
    python scripts/recover_jquants_universe.py --expect-sha256 <hex> --cross-check-file prices.csv
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import os
import re
import sys
from datetime import UTC, date, datetime
from pathlib import Path

UNIVERSE_NAME = "topix1000_usable_tickers.csv"
COVERAGE_NAME = "financial_summary_coverage.csv"
STATE_NAME = "financial_summary_auto_state.json"
UNIVERSE_COLUMNS = ["ticker", "name", "date", "new_index_category"]
COVERAGE_COLUMNS = [*UNIVERSE_COLUMNS, "has_any_financial_summary"]
TICKER_RE = re.compile(r"^\d{3}[0-9A-Z]$")  # 4 chars; new JPX codes carry a letter, e.g. 130A
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
FLAG_VALUES = {"True", "False"}
UNAVAILABLE_ERRNOS = {errno.EIO, errno.ENODEV, errno.ENXIO, errno.ESTALE, errno.ENOTCONN}

EXIT_OK, EXIT_INVALID, EXIT_REFUSED, EXIT_IO = 0, 2, 3, 4


class Refused(Exception):
    pass


class Invalid(Exception):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def default_store_dir() -> Path:
    project = os.getenv("JP_STOCK_ANALYSIS_PROJECT_DIR") or str(Path(__file__).resolve().parents[1])
    return Path(project) / "data" / "jquants_price_store"


def read_bytes_checked(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise Invalid([f"{path}: file not found"]) from None
    except OSError as exc:
        what = "drive/mount problem" if exc.errno in UNAVAILABLE_ERRNOS else "unreadable"
        raise OSError(f"{path}: {what}: {exc}") from exc


def parse_coverage(data: bytes, path: Path) -> list[list[str]]:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise Invalid([f"{path}: not UTF-8 ({exc})"]) from None
    return [row for row in csv.reader(io.StringIO(text, newline="")) if row]


def normalize_ticker(value: str) -> str:
    token = value.strip()
    if len(token) == 5 and token.endswith("0") and token[:4].isdigit():
        return token[:4]
    if token.isdigit() and len(token) < 4:
        return token.zfill(4)
    return token


def read_cross_tickers(path: Path) -> set[str]:
    tickers: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        lowered = {h.lower(): h for h in (reader.fieldnames or [])}
        col = next((lowered[c] for c in ("ticker", "code", "symbol") if c in lowered), None)
        if col is None:
            raise Invalid([f"{path}: needs a ticker/code/symbol column"])
        for row in reader:
            value = normalize_ticker(row.get(col) or "")
            if value:
                tickers.add(value)
    return tickers


def validate(
    rows: list[list[str]],
    *,
    expect_rows: int | None,
    allow_missing_fields: bool,
    state: dict | None,
) -> tuple[list[list[str]], dict]:
    """Returns (universe rows without header, report). Raises Invalid with every problem."""
    errors: list[str] = []
    if not rows:
        raise Invalid(["the coverage file is empty"])
    header, body = rows[0], rows[1:]
    if header != COVERAGE_COLUMNS:
        raise Invalid(
            [f"coverage header is {header}, expected exactly {COVERAGE_COLUMNS}; refusing to guess"]
        )
    if not body:
        raise Invalid(["the coverage file has a header but no rows"])

    seen: dict[str, int] = {}
    bad_width, bad_ticker, padded, dupes = [], [], [], []
    empty = {c: 0 for c in UNIVERSE_COLUMNS[1:]}
    bad_dates, bad_flags = [], []
    dates: set[str] = set()
    flags = {"True": 0, "False": 0}
    out_rows: list[list[str]] = []

    for lineno, row in enumerate(body, start=2):
        if len(row) != len(header):
            bad_width.append(lineno)
            continue
        ticker, name, day, category, flag = row
        if ticker != ticker.strip() or ticker == "":
            padded.append(lineno)
        if not TICKER_RE.match(ticker.strip()):
            bad_ticker.append((lineno, ticker))
        if ticker in seen:
            dupes.append((lineno, ticker, seen[ticker]))
        seen.setdefault(ticker, lineno)
        for col, value in (("name", name), ("date", day), ("new_index_category", category)):
            if value.strip() == "":
                empty[col] += 1
        if day.strip():
            if not DATE_RE.match(day):
                bad_dates.append((lineno, day))
            else:
                try:
                    date.fromisoformat(day)
                    dates.add(day)
                except ValueError:
                    bad_dates.append((lineno, day))
        if flag in FLAG_VALUES:
            flags[flag] += 1
        else:
            bad_flags.append((lineno, flag))
        out_rows.append([ticker, name, day, category])

    if bad_width:
        errors.append(f"rows with the wrong number of fields at lines {bad_width[:5]}")
    if padded:
        errors.append(f"tickers empty or with surrounding whitespace at lines {padded[:5]}")
    if bad_ticker:
        errors.append(f"tickers not matching ^\\d{{3}}[0-9A-Z]$: {bad_ticker[:5]}")
    if dupes:
        errors.append(f"duplicate tickers (line, ticker, first line): {dupes[:5]}")
    if bad_dates:
        errors.append(f"dates not YYYY-MM-DD: {bad_dates[:5]}")
    if bad_flags:
        errors.append(f"has_any_financial_summary not True/False: {bad_flags[:5]}")
    if not allow_missing_fields and any(empty.values()):
        errors.append(
            f"empty values in {empty} (rerun with --allow-missing-fields only if you accept that)"
        )
    if expect_rows is not None and len(out_rows) != expect_rows:
        errors.append(f"{len(out_rows)} rows, expected {expect_rows}")

    state_check = "not available"
    if state is not None:
        want_true, want_false = (
            state.get("unique_tickers_saved"),
            state.get("coverage_missing_count"),
        )
        if (want_true, want_false) == (flags["True"], flags["False"]):
            state_check = f"consistent with the daily state ({want_true} True / {want_false} False)"
        else:
            state_check = (
                f"MISMATCH: coverage has {flags['True']} True / {flags['False']} False, the state "
                f"recorded {want_true} / {want_false}"
            )
            errors.append(
                state_check + " -- the coverage file is not the one that state describes "
                "(use --skip-state-check only if a later backfill changed it)"
            )

    if errors:
        raise Invalid(errors)
    report = {
        "rows": len(out_rows),
        "unique_tickers": len(seen),
        "distinct_dates": sorted(dates),
        "empty_values": empty,
        "flags": flags,
        "state_check": state_check,
    }
    return out_rows, report


def render(out_rows: list[list[str]]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(UNIVERSE_COLUMNS)
    writer.writerows(out_rows)
    return buf.getvalue().encode("utf-8")


def create_new_file(path: Path, data: bytes) -> None:
    """Create-only: fails if the path already exists; removes its own partial file on error."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        if path.read_bytes() != data:
            raise OSError(errno.EIO, "read-back after write did not match", str(path))
    except BaseException:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Rebuild the universe CSV from the coverage CSV (validated, never overwrites)."
    )
    ap.add_argument("--store-dir", type=Path, help="default: <project>/data/jquants_price_store")
    ap.add_argument("--coverage-file", type=Path, help="default: <store>/" + COVERAGE_NAME)
    ap.add_argument("--output-file", type=Path, help="default: <store>/" + UNIVERSE_NAME)
    ap.add_argument("--state-file", type=Path, help="default: <store>/" + STATE_NAME)
    ap.add_argument("--skip-state-check", action="store_true")
    ap.add_argument("--expect-rows", type=int)
    ap.add_argument("--expect-sha256", help="SHA-256 of the ORIGINAL file; proves byte identity")
    ap.add_argument(
        "--cross-check-file",
        type=Path,
        help="CSV with a ticker/code column (e.g. prices_daily_bars.csv) to compare against",
    )
    ap.add_argument(
        "--require-subset",
        action="store_true",
        help="with --cross-check-file: fail if that file has tickers missing from the rebuild",
    )
    ap.add_argument("--allow-missing-fields", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    store = args.store_dir or default_store_dir()
    coverage_path = args.coverage_file or store / COVERAGE_NAME
    output_path = args.output_file or Path(
        os.getenv("JP_STOCK_ANALYSIS_UNIVERSE_FILE") or store / UNIVERSE_NAME
    )
    state_path = args.state_file or store / STATE_NAME

    try:
        data = read_bytes_checked(coverage_path)
        rows = parse_coverage(data, coverage_path)

        state = None
        if not args.skip_state_check and state_path.exists():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise Invalid([f"{state_path}: cannot be used as evidence ({exc})"]) from None
            if not isinstance(state, dict):
                raise Invalid([f"{state_path}: not a JSON object"])

        out_rows, report = validate(
            rows,
            expect_rows=args.expect_rows,
            allow_missing_fields=args.allow_missing_fields,
            state=state,
        )
        payload = render(out_rows)
        digest = hashlib.sha256(payload).hexdigest()
        report["output_sha256"] = digest

        errors = []
        if args.expect_sha256:
            if digest == args.expect_sha256.lower():
                report["byte_identity"] = "PROVEN (matches --expect-sha256)"
            else:
                errors.append(
                    f"sha256 of the rebuild is {digest}, not the expected {args.expect_sha256}: "
                    "the original differs in content or layout; nothing was written"
                )
        else:
            report["byte_identity"] = "NOT PROVEN (no --expect-sha256 given)"

        if args.cross_check_file:
            other = read_cross_tickers(args.cross_check_file)
            mine = {r[0] for r in out_rows}
            report["cross_check"] = {
                "file": str(args.cross_check_file),
                "tickers_in_file": len(other),
                "in_file_not_in_rebuild": len(other - mine),
                "in_rebuild_not_in_file": len(mine - other),
            }
            if args.require_subset and other - mine:
                errors.append(
                    f"{len(other - mine)} ticker(s) in {args.cross_check_file} are not in the "
                    f"rebuild, e.g. {sorted(other - mine)[:5]}"
                )
        if errors:
            raise Invalid(errors)
    except Invalid as exc:
        print("VALIDATION FAILED -- nothing was written:", file=sys.stderr)
        for e in exc.errors:
            print("  -", e, file=sys.stderr)
        return EXIT_INVALID
    except OSError as exc:
        print(f"I/O ERROR -- nothing was written: {exc}", file=sys.stderr)
        return EXIT_IO

    print(
        json.dumps(
            {"source": str(coverage_path), "target": str(output_path), **report},
            ensure_ascii=False,
            indent=2,
        )
    )

    if output_path.exists():
        if output_path.read_bytes() == payload:
            print("already restored: the output file exists with identical content; nothing to do")
            return EXIT_OK
        print(
            f"REFUSED: {output_path} already exists with different content. It is never "
            "overwritten; compare it with the rebuild yourself.",
            file=sys.stderr,
        )
        return EXIT_REFUSED
    if args.dry_run:
        print("dry-run: all checks passed; nothing written")
        return EXIT_OK

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    sidecar = output_path.with_name(f"{output_path.name}.recovery_{stamp}.json")
    provenance = {
        "kind": "universe_recovered_from_coverage",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "source_file": str(coverage_path),
        "source_sha256": hashlib.sha256(data).hexdigest(),
        "output_file": str(output_path),
        "claim": "content-equivalent reconstruction; see the caveats in the script docstring",
        **report,
    }
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        create_new_file(output_path, payload)
        create_new_file(
            sidecar, (json.dumps(provenance, ensure_ascii=False, indent=2) + "\n").encode()
        )
    except FileExistsError:
        print(
            f"REFUSED: {output_path} appeared while writing; it was not touched.", file=sys.stderr
        )
        return EXIT_REFUSED
    except OSError as exc:
        print(f"I/O ERROR while writing: {exc}", file=sys.stderr)
        return EXIT_IO
    print(f"restored: {output_path}\nprovenance: {sidecar}")
    return EXIT_OK


def main() -> int:
    return run(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
