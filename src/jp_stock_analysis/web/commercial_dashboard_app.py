"""FastAPI app for the J-Quants commercial screening dashboard."""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from jp_stock_analysis.web.commercial_dashboard_data import (
    CLASSIFICATION_FILES,
    get_audit,
    get_daily_summary,
    get_group_summary,
    get_rankings,
    get_snapshot,
    list_downloadable_files,
    resolve_download_file,
)


PACKAGE_DIR = Path(__file__).resolve().parent


def create_app(analysis_dir: str | Path = "analysis/jquants_fin_summary") -> FastAPI:
    app = FastAPI(title="J-Quants Commercial Screening Dashboard")
    app.state.analysis_dir = Path(analysis_dir)
    templates = Jinja2Templates(directory=str(PACKAGE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(PACKAGE_DIR / "static")), name="static")

    @app.get("/")
    async def index(request: Request):
        snapshot, snapshot_warnings = get_snapshot(app.state.analysis_dir)
        audit, audit_warnings = get_audit(app.state.analysis_dir)
        daily = get_daily_summary(app.state.analysis_dir, limit=20, offset=0)
        rankings = {
            classification: get_rankings(app.state.analysis_dir, classification, limit=20, offset=0)
            for classification in [
                "positive_reacted",
                "positive_unreacted",
                "negative_reacted",
                "negative_reversal",
            ]
        }
        context = {
            "request": request,
            "snapshot": snapshot,
            "audit": audit,
            "warnings": [*snapshot_warnings, *audit_warnings, *snapshot.get("warnings", []), *audit.get("warnings", [])],
            "daily_summary": daily.get("rows", []),
            "sector_summary": get_group_summary(app.state.analysis_dir, "sector", limit=20).get("rows", []),
            "market_summary": get_group_summary(app.state.analysis_dir, "market_segment", limit=20).get("rows", []),
            "topix_summary": get_group_summary(app.state.analysis_dir, "topix_size", limit=20).get("rows", []),
            "rankings": rankings,
            "downloads": list_downloadable_files(app.state.analysis_dir),
            "classification_labels": list(CLASSIFICATION_FILES),
        }
        return templates.TemplateResponse(request, "commercial_dashboard.html", context)

    @app.get("/health")
    async def health():
        snapshot, warnings = get_snapshot(app.state.analysis_dir)
        return {
            "status": "ok",
            "analysis_dir": str(app.state.analysis_dir),
            "latest_price_date": snapshot.get("latest_price_date"),
            "latest_disc_date": snapshot.get("latest_disc_date"),
            "total_dashboard_rows": snapshot.get("total_dashboard_rows"),
            "warnings": warnings,
        }

    @app.get("/api/snapshot")
    async def api_snapshot():
        payload, warnings = get_snapshot(app.state.analysis_dir)
        payload.setdefault("warnings", [])
        payload["warnings"] = [*payload.get("warnings", []), *warnings]
        return payload

    @app.get("/api/audit")
    async def api_audit():
        payload, warnings = get_audit(app.state.analysis_dir)
        payload.setdefault("warnings", [])
        payload["warnings"] = [*payload.get("warnings", []), *warnings]
        return payload

    @app.get("/api/daily-summary")
    async def api_daily_summary(
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = Query(500, ge=0, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return get_daily_summary(
            app.state.analysis_dir,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/rankings/{classification}")
    async def api_rankings(
        classification: str,
        q: str | None = None,
        sector: str | None = None,
        market_segment: str | None = None,
        topix_size_category: str | None = None,
        disc_date: str | None = None,
        limit: int = Query(100, ge=0, le=1000),
        offset: int = Query(0, ge=0),
        sort: str | None = None,
        order: str | None = "asc",
    ):
        result = get_rankings(
            app.state.analysis_dir,
            classification,
            q=q,
            sector=sector,
            market_segment=market_segment,
            topix_size_category=topix_size_category,
            disc_date=disc_date,
            limit=limit,
            offset=offset,
            sort=sort,
            order=order,
        )
        if result is None:
            raise HTTPException(status_code=404, detail="unknown classification")
        return result

    @app.get("/api/sector-summary")
    async def api_sector_summary(
        classification: str | None = None,
        disc_date: str | None = None,
        limit: int = Query(500, ge=0, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return get_group_summary(
            app.state.analysis_dir,
            "sector",
            classification=classification,
            disc_date=disc_date,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/market-segment-summary")
    async def api_market_summary(
        classification: str | None = None,
        disc_date: str | None = None,
        limit: int = Query(500, ge=0, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return get_group_summary(
            app.state.analysis_dir,
            "market_segment",
            classification=classification,
            disc_date=disc_date,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/topix-size-summary")
    async def api_topix_summary(
        classification: str | None = None,
        disc_date: str | None = None,
        limit: int = Query(500, ge=0, le=1000),
        offset: int = Query(0, ge=0),
    ):
        return get_group_summary(
            app.state.analysis_dir,
            "topix_size",
            classification=classification,
            disc_date=disc_date,
            limit=limit,
            offset=offset,
        )

    @app.get("/download/{name:path}")
    async def download(name: str):
        path = resolve_download_file(app.state.analysis_dir, name)
        if path is None:
            raise HTTPException(status_code=404, detail="download not found")
        media_type = "application/json" if path.suffix == ".json" else "text/csv"
        return Response(
            path.read_bytes(),
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{path.name}"'},
        )

    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", default="analysis/jquants_fin_summary")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(args.analysis_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="debug" if args.debug else "info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
