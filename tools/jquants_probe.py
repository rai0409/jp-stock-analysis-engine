from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

CODE = os.environ.get("JQUANTS_PROBE_CODE", "7203")
FROM_DATE = os.environ.get("JQUANTS_PROBE_FROM", "2023-01-01")
TO_DATE = os.environ.get("JQUANTS_PROBE_TO", "2023-12-31")

API_BASE = os.environ.get("JQUANTS_API_BASE_URL", "https://api.jquants.com")


@dataclass
class ProbeResult:
    name: str
    status: str
    body_preview: str


def _request(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
    body: dict[str, object] | None = None,
) -> ProbeResult:
    data = None
    request_headers = headers or {}

    if body is not None:
        data = json.dumps(body).encode("utf-8")
        request_headers = {
            "Content-Type": "application/json",
            **request_headers,
        }

    req = urllib.request.Request(
        url,
        data=data,
        headers=request_headers,
        method=method,
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return ProbeResult(
                name=url,
                status=str(resp.status),
                body_preview=text[:500],
            )
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        return ProbeResult(
            name=url,
            status=str(exc.code),
            body_preview=text[:500],
        )
    except Exception as exc:
        return ProbeResult(
            name=url,
            status=f"ERROR:{type(exc).__name__}",
            body_preview=str(exc)[:500],
        )


def _print_result(label: str, result: ProbeResult) -> None:
    print(f"\n=== {label} ===")
    print(f"status={result.status}")
    print("body_preview=")
    print(result.body_preview or "(empty)")


def probe_v2_api_key() -> None:
    api_key = os.environ.get("JQUANTS_API_KEY")
    url = (
        f"{API_BASE}/v2/prices/daily_quotes?"
        + urllib.parse.urlencode({"code": CODE, "from": FROM_DATE, "to": TO_DATE})
    )

    if not api_key:
        print("\n=== v2 x-api-key ===")
        print("SKIP: JQUANTS_API_KEY is not set")
        return

    result = _request(
        "GET",
        url,
        headers={"x-api-key": api_key},
    )
    _print_result("v2 x-api-key", result)


def probe_v2_bearer_api_key() -> None:
    api_key = os.environ.get("JQUANTS_API_KEY")
    url = (
        f"{API_BASE}/v2/prices/daily_quotes?"
        + urllib.parse.urlencode({"code": CODE, "from": FROM_DATE, "to": TO_DATE})
    )

    if not api_key:
        print("\n=== v2 bearer-with-api-key ===")
        print("SKIP: JQUANTS_API_KEY is not set")
        return

    result = _request(
        "GET",
        url,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    _print_result("v2 bearer-with-api-key", result)


def probe_v1_legacy_token_flow() -> None:
    email = os.environ.get("JQUANTS_EMAIL")
    password = os.environ.get("JQUANTS_PASSWORD")

    print("\n=== v1 legacy token flow ===")
    if not email or not password:
        print("SKIP: JQUANTS_EMAIL or JQUANTS_PASSWORD is not set")
        return

    auth_user_url = f"{API_BASE}/v1/token/auth_user"
    auth_user = _request(
        "POST",
        auth_user_url,
        body={"mailaddress": email, "password": password},
    )
    print(f"auth_user status={auth_user.status}")
    print("auth_user body_preview=")
    print(auth_user.body_preview or "(empty)")

    if auth_user.status != "200":
        return

    try:
        refresh_token = json.loads(auth_user.body_preview).get("refreshToken")
    except json.JSONDecodeError:
        print("Could not parse refreshToken")
        return

    if not refresh_token:
        print("No refreshToken in response")
        return

    auth_refresh_url = (
        f"{API_BASE}/v1/token/auth_refresh?"
        + urllib.parse.urlencode({"refreshtoken": refresh_token})
    )
    auth_refresh = _request("POST", auth_refresh_url)
    print(f"auth_refresh status={auth_refresh.status}")
    print("auth_refresh body_preview=")
    print(auth_refresh.body_preview or "(empty)")

    if auth_refresh.status != "200":
        return

    try:
        id_token = json.loads(auth_refresh.body_preview).get("idToken")
    except json.JSONDecodeError:
        print("Could not parse idToken")
        return

    if not id_token:
        print("No idToken in response")
        return

    prices_url = (
        f"{API_BASE}/v1/prices/daily_quotes?"
        + urllib.parse.urlencode({"code": CODE, "from": FROM_DATE, "to": TO_DATE})
    )
    prices = _request(
        "GET",
        prices_url,
        headers={"Authorization": f"Bearer {id_token}"},
    )
    _print_result("v1 daily_quotes bearer-idToken", prices)


def main() -> int:
    print("J-Quants probe")
    print(f"API_BASE={API_BASE}")
    print(f"CODE={CODE}")
    print(f"FROM={FROM_DATE}")
    print(f"TO={TO_DATE}")
    print("Secrets are never printed.")

    probe_v2_api_key()
    probe_v2_bearer_api_key()
    probe_v1_legacy_token_flow()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
