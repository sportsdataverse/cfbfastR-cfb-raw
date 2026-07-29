"""Round-robin Proxy Bonanza rotation for the CFB backfill, with a bandwidth governor.

WHY: ESPN's Core v2 host (``sports.core.api.espn.com``) rate-limits hard under
concurrency -- 429s and 5xx -- and it carries ~99% of the backfill's requests
(the ~220-athlete ``$ref`` roster fan-out per modern game). Spreading those over
50 IPs is what lets the worker count go up.

WHAT IS *NOT* PROXIED: the Site v2 ``summary`` payload is ~400KB in a SINGLE
request. It is expensive in bytes and free in rate pressure, so routing it
through a metered pool would burn quota for no concurrency gain. Hosts in
``DIRECT_HOSTS`` are excluded via ``NO_PROXY``, which ``requests`` honours
alongside ``HTTPS_PROXY`` -- so the split needs no change inside sportsdataverse.

BANDWIDTH IS THE BINDING CONSTRAINT, NOT REQUESTS. Measured wire bytes are
~1.14 MB/game pre-2014 and ~2.88 MB/game post-2014, i.e. ~39.7 GB for a full
2004-2025 run, against a package that holds far less. The governor therefore
polls the package's REMAINING bandwidth (authoritative, not an estimate) and
transparently falls back to direct connections once the reserve is reached, so
the run degrades instead of hard-failing mid-season.

SECRETS: credentials are read from ~/.Renviron at call time and never written to
disk or logged. Only ``ip:port`` is ever printed -- never the proxy URL, which
embeds login:password.
"""

from __future__ import annotations

import itertools
import os
import pathlib
import re
import threading
import time

_RENVIRON_FILES = (
    pathlib.Path.home() / ".Renviron",
    pathlib.Path.home() / "Documents" / ".Renviron",
)

#: Hosts that must NOT go through the metered pool. Site v2 summary and the CDN
#: sidecar are single large requests -- bytes-expensive, rate-cheap.
DIRECT_HOSTS = ("site.api.espn.com", "cdn.espn.com")

_lock = threading.Lock()
_proxies: list[str] | None = None
_cycle: itertools.cycle | None = None
_remaining: int | None = None
_remaining_at: float = 0.0
_disabled = False

_POLL_TTL = float(os.getenv("CFB_PROXY_POLL_TTL", "300"))
_RESERVE_BYTES = float(os.getenv("CFB_PROXY_RESERVE_GB", "1.0")) * 1024**3

# Measured proxied wire bytes per game (Core v2 only -- Site v2 goes direct).
# pre-2014 : core plays 347 KB + roster $ref walk ~65 x 5.64 KB  ~= 744 KB
# 2014+    : core plays 726 KB + roster $ref walk ~220 x 7.62 KB ~= 2436 KB
_BYTES_PRE_2014 = 744 * 1024
_BYTES_POST_2014 = 2436 * 1024
_PARTICIPANTS_ERA = 2014

# LOCAL accounting is the primary control, NOT the provider's counter.
# Proxy Bonanza reports bandwidth in BATCHES, not live: 3.55 MiB pushed through
# a proxy left `bandwidth` byte-identical after 60s, and the value did not move
# across ~50 minutes of real scraping. A governor that waits for that number to
# drop would sail past the quota and only find out when requests start failing.
#
# So each worker gets an equal slice of the remaining budget and meters itself
# against measured per-game costs. No shared state and no locking: a worker
# simply stops proxying once it has spent its own share. Slightly conservative
# (a worker that finishes early leaves its slice unused), which is the right
# direction to err on a metered resource.
_my_budget: float | None = None
_spent: float = 0.0


def _renviron() -> dict[str, str]:
    """Read .Renviron at call time. Values are never logged."""
    vals: dict[str, str] = {}
    for f in _RENVIRON_FILES:
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            m = re.match(r"^([A-Za-z_][\w.]*)\s*=\s*(.*)$", line.strip())
            if m:
                vals.setdefault(m.group(1), m.group(2).strip().strip('"').strip("'"))
    return vals


def _fetch_package() -> dict:
    import requests

    env = _renviron()
    key = env.get("PROXYBONANZA_API_KEY") or env.get("PROXY_KEY")
    pkg = env.get("PROXY_PKG")
    base = env.get("PROXY_ENDPOINT", "https://proxybonanza.com/api/v1/userpackages")
    if not (key and pkg):
        raise RuntimeError("PROXYBONANZA_API_KEY / PROXY_PKG not found in .Renviron")
    r = requests.get(f"{base}/{pkg}.json", headers={"Authorization": key}, timeout=30)
    r.raise_for_status()
    return r.json()["data"]


def _load() -> None:
    """Populate the proxy ring, the baseline budget, and this worker's slice."""
    global _proxies, _cycle, _remaining, _remaining_at, _my_budget
    data = _fetch_package()
    login, password = data["login"], data["password"]
    urls = [
        f"http://{login}:{password}@{p['ip']}:{p['port_http']}"
        for p in data.get("ippacks", [])
        if p.get("active") and p.get("ip") and p.get("port_http")
    ]
    if not urls:
        raise RuntimeError("proxy package returned no active ips")
    # Offset each worker into the ring so 12 processes don't all start on the
    # same IP and serialise onto it.
    offset = os.getpid() % len(urls)
    _proxies = urls
    _cycle = itertools.cycle(urls[offset:] + urls[:offset])
    _remaining = int(data.get("bandwidth") or 0)
    _remaining_at = time.time()

    # The provider reading is a trustworthy STARTING POINT even though it does
    # not update live, so use it to size this worker's slice once.
    workers = max(1, int(os.getenv("CFB_SCRAPE_WORKERS", "1")))
    _my_budget = max(0.0, _remaining - _RESERVE_BYTES) / workers


def _refresh_remaining() -> None:
    """Re-poll remaining bandwidth. Authoritative -- we do not estimate bytes."""
    global _remaining, _remaining_at
    if time.time() - _remaining_at < _POLL_TTL:
        return
    try:
        data = _fetch_package()
        _remaining = int(data.get("bandwidth") or 0)
    except Exception:
        # A failed poll must not take the run down; keep the last reading.
        pass
    _remaining_at = time.time()


def status() -> str:
    """Human-safe status line. Never includes credentials."""
    if _disabled:
        return "proxy: DISABLED (bandwidth reserve reached) -- direct connections"
    if _proxies is None:
        return "proxy: not initialised"
    gb = (_remaining or 0) / 1024**3
    budget = f"{_my_budget / 1024**3:.2f}" if _my_budget is not None else "?"
    return (
        f"proxy: {len(_proxies)} ips, ~{gb:.2f} GiB reported remaining "
        f"(batched, not live); this worker spent {_spent / 1024**3:.2f}/{budget} GiB of its slice"
    )


def apply_to_env(season: int | None = None) -> str | None:
    """Point the next request family at the next proxy in the ring.

    Sets HTTP(S)_PROXY plus NO_PROXY so only the rate-limited Core v2 host is
    routed. ``requests`` re-reads these per request, so mutating them between
    games takes effect immediately for every nested call.

    Returns a masked ``ip:port`` for logging, or ``None`` when running direct.
    """
    global _disabled
    if os.getenv("CFB_PROXY", "0") not in ("1", "true", "yes"):
        return None

    with _lock:
        if _disabled:
            return None
        if _proxies is None:
            try:
                _load()
            except Exception:
                _disabled = True
                return None
        global _spent
        _refresh_remaining()

        # Two independent stop conditions. The local one is what actually fires
        # in practice; the provider one only helps if their counter ever catches
        # up mid-run (or someone else drains the package).
        out_of_slice = _my_budget is not None and _spent >= _my_budget
        provider_low = _remaining is not None and _remaining < _RESERVE_BYTES
        if out_of_slice or provider_low:
            _disabled = True
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                os.environ.pop(k, None)
            return None

        # Charge this game up front: if it dies mid-way we have still spent the
        # bytes it pulled, so pre-charging is the conservative direction.
        _spent += (
            _BYTES_PRE_2014 if (season or 0) < _PARTICIPANTS_ERA else _BYTES_POST_2014
        )
        url = next(_cycle)  # type: ignore[arg-type]

    os.environ["HTTP_PROXY"] = url
    os.environ["HTTPS_PROXY"] = url
    no_proxy = os.getenv("CFB_PROXY_DIRECT_HOSTS") or ",".join(DIRECT_HOSTS)
    os.environ["NO_PROXY"] = no_proxy
    os.environ["no_proxy"] = no_proxy
    # host:port only -- the full URL embeds login:password
    return url.rsplit("@", 1)[-1]


def disabled() -> bool:
    return _disabled
