"""Deterministic tests for local real-price CSV preparation.

No network, no randomness, no real price data: every input is a tiny fake
fixture built inline. These tests only exercise schema validation,
normalization, coverage checks, and the CLI smoke -- never predictive
validation.
"""

from __future__ import annotations

from datetime import date

import pytest

from jp_stock_analysis.cli import main
from jp_stock_analysis.errors import DataValidationError
from jp_stock_analysis.validation.price_prep import normalize_ticker, prepare_price_csv


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _read(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_ticker_date_close_input(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n3928,2026-03-28,1010\n3928,2026-03-31,1020\n",
    )
    out = tmp_path / "out.csv"
    result = prepare_price_csv(src, out, ["3928"], date(2026, 3, 28))
    assert _read(out) == [
        "ticker,date,close",
        "3928,2026-03-28,1010",
        "3928,2026-03-31,1020",
    ]
    assert result.total_rows_written == 2
    assert result.rows_per_ticker == {"3928": 2}
    assert result.rows_after_from_date == {"3928": 2}


def test_ohlcv_input_takes_close_only(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,open,high,low,close,volume\n"
        "4107,2026-03-28,5000,5100,4950,5050,9000\n",
    )
    out = tmp_path / "out.csv"
    prepare_price_csv(src, out, ["4107"], date(2026, 3, 28))
    assert _read(out) == ["ticker,date,close", "4107,2026-03-28,5050"]


def test_localcode_date_close_input(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "LocalCode,Date,Close\n4264,2026/03/28,808\n4264,20260331,815\n",
    )
    out = tmp_path / "out.csv"
    prepare_price_csv(src, out, ["4264"], date(2026, 3, 28))
    assert _read(out) == [
        "ticker,date,close",
        "4264,2026-03-28,808",
        "4264,2026-03-31,815",
    ]


def test_code_header_capitalized(tmp_path):
    src = _write(tmp_path / "in.csv", "Code,Date,Close\n3928,2026-03-28,1010\n")
    out = tmp_path / "out.csv"
    prepare_price_csv(src, out, ["3928"], date(2026, 3, 28))
    assert _read(out)[1] == "3928,2026-03-28,1010"


def test_dot_t_suffix_normalization(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n7203.T,2026-03-28,2500\n",
    )
    out = tmp_path / "out.csv"
    result = prepare_price_csv(src, out, ["7203.T"], date(2026, 3, 28))
    # both the requested ticker and the input rows lose the .T suffix
    assert _read(out)[1] == "7203,2026-03-28,2500"
    assert result.tickers == ["7203"]


def test_alphanumeric_listing_code_preserved(tmp_path):
    src = _write(tmp_path / "in.csv", "ticker,date,close\n286a,2026-03-28,1500\n")
    out = tmp_path / "out.csv"
    prepare_price_csv(src, out, ["286A"], date(2026, 3, 28))
    assert _read(out)[1] == "286A,2026-03-28,1500"


def test_missing_required_ticker_fails(tmp_path):
    src = _write(tmp_path / "in.csv", "ticker,date,close\n3928,2026-03-28,1010\n")
    out = tmp_path / "out.csv"
    with pytest.raises(DataValidationError, match="absent from input"):
        prepare_price_csv(src, out, ["3928", "4107"], date(2026, 3, 28))


def test_missing_required_column_fails(tmp_path):
    src = _write(tmp_path / "in.csv", "ticker,date\n3928,2026-03-28\n")
    out = tmp_path / "out.csv"
    with pytest.raises(DataValidationError, match="missing required column"):
        prepare_price_csv(src, out, ["3928"], date(2026, 3, 28))


def test_insufficient_rows_after_from_date_fails(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n"
        "3928,2026-03-26,1000\n"  # before from-date: does not count
        "3928,2026-03-28,1010\n"
        "3928,2026-03-31,1020\n",
    )
    out = tmp_path / "out.csv"
    with pytest.raises(DataValidationError, match="insufficient rows"):
        prepare_price_csv(src, out, ["3928"], date(2026, 3, 28), min_rows_after=3)
    # below the threshold the output file is not produced
    assert not out.exists()


def test_rows_before_from_date_kept_but_not_counted(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n3928,2026-03-26,1000\n3928,2026-03-28,1010\n",
    )
    out = tmp_path / "out.csv"
    result = prepare_price_csv(src, out, ["3928"], date(2026, 3, 28), min_rows_after=1)
    assert result.rows_per_ticker == {"3928": 2}
    assert result.rows_after_from_date == {"3928": 1}
    assert _read(out) == [
        "ticker,date,close",
        "3928,2026-03-26,1000",
        "3928,2026-03-28,1010",
    ]


def test_non_numeric_close_fails(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n3928,2026-03-28,not-a-number\n",
    )
    out = tmp_path / "out.csv"
    with pytest.raises(DataValidationError, match="non-numeric close"):
        prepare_price_csv(src, out, ["3928"], date(2026, 3, 28))


def test_comma_grouped_close_is_numeric(tmp_path):
    src = _write(tmp_path / "in.csv", 'ticker,date,close\n4107,2026-03-28,"51,015"\n')
    out = tmp_path / "out.csv"
    prepare_price_csv(src, out, ["4107"], date(2026, 3, 28))
    assert _read(out)[1] == "4107,2026-03-28,51015"


def test_deterministic_ordering_and_ticker_filtering(tmp_path):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n"
        "4107,2026-03-31,5060\n"
        "3928,2026-03-31,1020\n"
        "9999,2026-03-28,1\n"  # not requested: dropped
        "4107,2026-03-28,5050\n"
        "3928,2026-03-28,1010\n",
    )
    out = tmp_path / "out.csv"
    result = prepare_price_csv(src, out, ["3928", "4107"], date(2026, 3, 28))
    assert _read(out) == [
        "ticker,date,close",
        "3928,2026-03-28,1010",
        "3928,2026-03-31,1020",
        "4107,2026-03-28,5050",
        "4107,2026-03-31,5060",
    ]
    assert "9999" not in result.rows_per_ticker


def test_unparseable_date_fails(tmp_path):
    src = _write(tmp_path / "in.csv", "ticker,date,close\n3928,March 28 2026,1010\n")
    out = tmp_path / "out.csv"
    with pytest.raises(DataValidationError, match="unparseable date"):
        prepare_price_csv(src, out, ["3928"], date(2026, 3, 28))


def test_normalize_ticker_helper():
    assert normalize_ticker(" 7203.t ") == "7203"
    assert normalize_ticker("286a") == "286A"
    assert normalize_ticker(None) == ""


def test_cli_smoke(tmp_path, capsys):
    src = _write(
        tmp_path / "in.csv",
        "ticker,date,close\n3928,2026-03-28,1010\n3928,2026-03-31,1020\n",
    )
    out = tmp_path / "out.csv"
    code = main(
        [
            "prepare-price-csv",
            "--input",
            str(src),
            "--output",
            str(out),
            "--tickers",
            "3928",
            "--from-date",
            "2026-03-28",
            "--min-rows-after",
            "2",
        ]
    )
    assert code == 0
    assert out.exists()
    assert "Prepared 2 rows" in capsys.readouterr().out


def test_cli_smoke_failure_returns_nonzero(tmp_path, capsys):
    src = _write(tmp_path / "in.csv", "ticker,date,close\n3928,2026-03-28,1010\n")
    out = tmp_path / "out.csv"
    code = main(
        [
            "prepare-price-csv",
            "--input",
            str(src),
            "--output",
            str(out),
            "--tickers",
            "3928,4107",
            "--from-date",
            "2026-03-28",
        ]
    )
    assert code == 1
    assert "absent from input" in capsys.readouterr().err


def test_cli_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["prepare-price-csv", "--help"])
    assert exc.value.code == 0
    assert "ticker,date,close" in capsys.readouterr().out
