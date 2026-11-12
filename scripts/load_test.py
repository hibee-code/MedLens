"""Load test of the API and the scan pipeline (open-loop: requests arrive at a set rate whether
or not earlier ones finished, as real users do).

    python scripts/load_test.py --scenario reads --rate 200 --duration 60
    python scripts/load_test.py --scenario scans --rate 5 --duration 120
    python scripts/load_test.py --scenario mixed --rate 50 --duration 120

Scenarios:
  reads   public product lookups and searches (GET /products/{nafdac}, GET /products?query=)
  scans   guest scans: a unique photo and device id per request, then polling until the verdict
  mixed   80% reads, 20% scans

Reported per request type: count, statuses, achieved rate, latency p50/p90/p95/p99/max; for
scans also the share answered within the fast path (200) and the end-to-end time to a verdict;
the scan queue's peak depth (from /metrics). Arrivals that would exceed --max-in-flight are
counted as dropped: the client, not the server, could not keep up.

Run against a stack with rate limits off (they are per IP, and the test comes from one):
MEDLENS_RATELIMIT__ENABLED=false. The test client shares the machine with the stack when run
locally, so local numbers are a floor, not the capacity of a deployment.
"""

# ruff: noqa: S311 - random picks traffic for a load test, nothing secret
import argparse
import asyncio
import contextlib
import io
import json
import random
import statistics
import sys
import time
import uuid
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
STATES = ["NG-LA", "NG-KN", "NG-FC", "NG-RI", "NG-OY", "NG-KD", "NG-EN", "NG-AN"]


@dataclass
class Stats:
    latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    statuses: dict[str, Counter[str]] = field(default_factory=lambda: defaultdict(Counter))
    end_to_end: list[float] = field(default_factory=list)
    fast_path: int = 0
    verdicts: Counter[str] = field(default_factory=Counter)
    dropped: int = 0
    queue_peak: float = 0.0

    def record(self, name: str, status: str, seconds: float) -> None:
        self.statuses[name][status] += 1
        self.latencies[name].append(seconds)


def percentile(values: list[float], share: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(share * len(ordered)))]


def summary(stats: Stats, duration: float) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for name, values in stats.latencies.items():
        ok = sum(n for status, n in stats.statuses[name].items() if status.startswith("2"))
        report[name] = {
            "count": len(values),
            "per_second": round(len(values) / duration, 2),
            "success_rate": round(ok / len(values), 4) if values else None,
            "statuses": dict(stats.statuses[name]),
            "latency_ms": {
                "p50": round(statistics.median(values) * 1000),
                "p90": round(percentile(values, 0.90) * 1000),
                "p95": round(percentile(values, 0.95) * 1000),
                "p99": round(percentile(values, 0.99) * 1000),
                "max": round(max(values) * 1000),
            },
        }
    if stats.end_to_end:
        done = stats.end_to_end
        report["scan_to_verdict_ms"] = {
            "count": len(done),
            "fast_path_share": round(stats.fast_path / len(done), 3),
            "p50": round(statistics.median(done) * 1000),
            "p95": round(percentile(done, 0.95) * 1000),
            "max": round(max(done) * 1000),
            "verdicts": dict(stats.verdicts),
        }
    report["dropped_by_client"] = stats.dropped
    report["scan_queue_peak"] = stats.queue_peak
    return report


def photos(count: int) -> list[bytes]:
    """Variants of a seed pack photo: each differs, so none is answered as a duplicate."""
    base = Image.open(ROOT / "seed" / "test_packs" / "genuine.jpg").convert("RGB")
    rng = random.Random(7)
    made = []
    for index in range(count):
        picture = base.copy()
        draw = ImageDraw.Draw(picture)
        x, y = rng.randrange(0, picture.width - 40), rng.randrange(0, 60)
        draw.rectangle((x, y, x + 30, y + 12), fill=(rng.randrange(256), 90, 90))
        draw.text((10, picture.height - 30), f"lt-{index}", fill=(120, 120, 120))
        buffer = io.BytesIO()
        picture.save(buffer, "JPEG", quality=85)
        made.append(buffer.getvalue())
    return made


def products() -> list[str]:
    import csv

    with (ROOT / "seed" / "products.csv").open(encoding="utf-8") as handle:
        return [row["nafdac_number"] for row in csv.DictReader(handle)]


async def timed(stats: Stats, name: str, call: Any) -> httpx.Response | None:
    started = time.perf_counter()
    try:
        response: httpx.Response = await call
    except httpx.HTTPError as exc:
        stats.record(name, type(exc).__name__, time.perf_counter() - started)
        return None
    stats.record(name, str(response.status_code), time.perf_counter() - started)
    return response


async def read(client: httpx.AsyncClient, stats: Stats, numbers: list[str]) -> None:
    if random.random() < 0.7:
        await timed(
            stats,
            "GET /products/{nafdac}",
            client.get(f"/api/v1/products/{random.choice(numbers)}"),
        )
    else:
        term = random.choice(["paracetamol", "amoxicillin", "chloroquine", "artemether", "emzor"])
        await timed(
            stats, "GET /products?query", client.get("/api/v1/products", params={"query": term})
        )


async def scan(
    client: httpx.AsyncClient, stats: Stats, images: list[bytes], poll_seconds: float
) -> None:
    device = f"loadtest-{uuid.uuid4()}"
    headers = {"X-Device-Id": device}
    started = time.perf_counter()
    response = await timed(
        stats,
        "POST /scans",
        client.post(
            "/api/v1/scans",
            files={"image": ("pack.jpg", random.choice(images), "image/jpeg")},
            data={"state": random.choice(STATES), "latitude": "6.52", "longitude": "3.37"},
            headers=headers,
        ),
    )
    if response is None or response.status_code not in (200, 202):
        return
    body = response.json()
    if response.status_code == 200 and body.get("status") in ("completed", "rejected"):
        stats.fast_path += 1
        stats.end_to_end.append(time.perf_counter() - started)
        stats.verdicts[str((body.get("result") or {}).get("final_verdict"))] += 1
        return
    scan_id = body["id"]
    deadline = started + 120
    while time.perf_counter() < deadline:
        await asyncio.sleep(poll_seconds)
        polled = await timed(
            stats, "GET /scans/{id}", client.get(f"/api/v1/scans/{scan_id}", headers=headers)
        )
        if polled is not None and polled.status_code == 200:
            data = polled.json()
            if data.get("status") in ("completed", "rejected", "failed"):
                stats.end_to_end.append(time.perf_counter() - started)
                stats.verdicts[str(data.get("final_verdict"))] += 1
                return
    stats.verdicts["not finished in 120 s"] += 1


async def watch_queue(client: httpx.AsyncClient, stats: Stats, stop: asyncio.Event) -> None:
    while not stop.is_set():
        try:
            text = (await client.get("/metrics")).text
            for line in text.splitlines():
                if line.startswith('medlens_queue_depth{queue="scans.pipeline"}'):
                    stats.queue_peak = max(stats.queue_peak, float(line.split()[-1]))
        except httpx.HTTPError:
            pass
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(stop.wait(), timeout=2)


async def run(args: argparse.Namespace) -> tuple[Stats, float]:
    stats = Stats()
    numbers = products()
    images = photos(200) if args.scenario != "reads" else []
    limits = httpx.Limits(
        max_connections=args.max_in_flight, max_keepalive_connections=args.max_in_flight
    )
    async with httpx.AsyncClient(base_url=args.base_url, timeout=60, limits=limits) as client:
        stop = asyncio.Event()
        watcher = asyncio.create_task(watch_queue(client, stats, stop))
        in_flight: set[asyncio.Task[None]] = set()
        started = time.perf_counter()
        next_at = started
        while next_at - started < args.duration:
            now = time.perf_counter()
            if next_at > now:
                await asyncio.sleep(next_at - now)
            next_at += random.expovariate(args.rate)  # Poisson arrivals
            if len(in_flight) >= args.max_in_flight:
                stats.dropped += 1
                continue
            kind = args.scenario
            if kind == "mixed":
                kind = "scans" if random.random() < 0.2 else "reads"
            job = (
                scan(client, stats, images, args.poll_seconds)
                if kind == "scans"
                else read(client, stats, numbers)
            )
            task = asyncio.create_task(job)
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)
        elapsed = time.perf_counter() - started
        if in_flight:
            await asyncio.wait(in_flight, timeout=150)
        stop.set()
        await watcher
    return stats, elapsed


def run_process(args: argparse.Namespace) -> tuple[Stats, float]:
    """One client process (an asyncio client tops out at a few hundred requests a second)."""
    return asyncio.run(run(args))


def merge(parts: list[tuple[Stats, float]]) -> tuple[Stats, float]:
    total = Stats()
    for stats, _ in parts:
        for name, values in stats.latencies.items():
            total.latencies[name] += values
        for name, counter in stats.statuses.items():
            total.statuses[name].update(counter)
        total.end_to_end += stats.end_to_end
        total.fast_path += stats.fast_path
        total.verdicts.update(stats.verdicts)
        total.dropped += stats.dropped
        total.queue_peak = max(total.queue_peak, stats.queue_peak)
    return total, max(elapsed for _, elapsed in parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scenario", choices=["reads", "scans", "mixed"], default="reads")
    parser.add_argument("--rate", type=float, default=50.0, help="arrivals per second")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals")
    parser.add_argument("--max-in-flight", type=int, default=500)
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument("--processes", type=int, default=4, help="client processes sharing --rate")
    args = parser.parse_args(argv)
    share = argparse.Namespace(**vars(args))
    share.rate = args.rate / args.processes
    share.max_in_flight = max(1, args.max_in_flight // args.processes)
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        parts = list(pool.map(run_process, [share] * args.processes))
    stats, elapsed = merge(parts)
    report = {
        "scenario": args.scenario,
        "target_rate_per_second": args.rate,
        "client_processes": args.processes,
        "duration_seconds": round(elapsed, 1),
        **summary(stats, elapsed),
    }
    json.dump(report, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
