#!/usr/bin/env python3
"""Resumable GitHub Issues -> Azure Blob ingestion for OpsForge.

The source is the GitHub REST API. A checkpoint is committed only after all
issues and comments in a page have been uploaded. Replays are idempotent.
"""
import argparse
import json
import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse

import requests
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ClientAssertionCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

LOG = logging.getLogger("opsforge.ingest")
API = "https://api.github.com"
BASELINE = "2008-01-01T00:00:00Z"


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def utc_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_json(session: requests.Session, url: str, params: dict | None = None) -> tuple[Any, dict]:
    for attempt in range(10):
        try:
            response = session.get(url, params=params, timeout=(10, 90))
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == 9:
                raise
            delay = min(2 ** attempt + random.random(), 90)
            LOG.warning("Network error %s; retry in %.1fs", exc, delay)
            time.sleep(delay)
            continue
        if response.ok:
            return response.json(), response.links
        if response.status_code in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining")
            reset = response.headers.get("X-RateLimit-Reset")
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = max(0, (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds())
            elif remaining == "0" and reset:
                delay = max(0, int(reset) - time.time()) + 5
            else:
                delay = min(30 * (attempt + 1), 300)
        elif response.status_code in (500, 502, 503, 504):
            delay = min(2 ** attempt + random.random(), 90)
        else:
            raise RuntimeError(f"GitHub API {response.status_code} at {url}: {response.text[:400]}")
        if attempt == 9:
            raise RuntimeError(f"GitHub API retries exhausted: {response.status_code} at {url}")
        LOG.warning("GitHub API %s; retry in %.1fs", response.status_code, delay)
        time.sleep(delay)
    raise RuntimeError("Unreachable")


def paged(session: requests.Session, url: str, params: dict | None = None):
    next_url = url
    while next_url:
        items, links = get_json(session, next_url, params=params)
        if not isinstance(items, list):
            raise TypeError(f"Expected list from {next_url}")
        yield items
        next_url = links.get("next", {}).get("url")
        params = None


def upload_json(container, name: str, data: Any) -> None:
    body = json.dumps(data, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    container.upload_blob(
        name=name,
        data=body,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/json"),
        max_concurrency=2,
    )


def load_checkpoint(container, path: str) -> dict | None:
    try:
        return json.loads(container.download_blob(path).readall())
    except ResourceNotFoundError:
        return None


def comments_for_issue(session: requests.Session, issue: dict) -> list[dict]:
    if not issue.get("comments", 0):
        return []
    result = []
    for page in paged(session, issue["comments_url"], {"per_page": 100}):
        result.extend(page)
    return result


def get_fresh_github_assertion() -> str:
    """Request a fresh short-lived OIDC assertion for Azure token renewal."""
    response = requests.get(
        os.environ["ACTIONS_ID_TOKEN_REQUEST_URL"],
        params={"audience": "api://AzureADTokenExchange"},
        headers={
            "Authorization": f'Bearer {os.environ["ACTIONS_ID_TOKEN_REQUEST_TOKEN"]}'
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["value"]


def run(args: argparse.Namespace) -> None:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required (GitHub Actions provides one automatically)")
    account = os.environ["AZURE_STORAGE_ACCOUNT"]
    container_name = os.environ.get("AZURE_STORAGE_CONTAINER", "github-issues-raw")
    repository = os.environ.get("GITHUB_SOURCE_REPO", "apache/airflow")
    if repository.count("/") != 1:
        raise ValueError("GITHUB_SOURCE_REPO must be owner/repository")
    owner, repo = repository.split("/")
    credential = ClientAssertionCredential(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        func=get_fresh_github_assertion,
    )
    client = BlobServiceClient(f"https://{account}.blob.core.windows.net", credential=credential)
    container = client.get_container_client(container_name)
    # The private container must exist already. No account keys or connection strings.
    container.get_container_properties()

    checkpoint_path = f"_control/{owner}/{repo}/checkpoint.json"
    checkpoint = load_checkpoint(container, checkpoint_path)
    if checkpoint and checkpoint.get("repository") != repository:
        raise ValueError("Checkpoint repository mismatch")

    # A full replay intentionally ignores the checkpoint, but never clears it.
    if args.mode == "full":
        start = parse_time(BASELINE)
    elif checkpoint:
        start = parse_time(checkpoint["watermark"]) - timedelta(minutes=args.overlap_minutes)
    else:
        start = parse_time(BASELINE)
    start = max(start, parse_time(BASELINE))
    LOG.info("Repository=%s mode=%s starting_since=%s", repository, args.mode, utc_iso(start))

    session = requests.Session()
    session.headers.update({
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "Authorization": f"Bearer {token}",
        "User-Agent": "OpsForge-GitHub-Ingestion/1.0",
    })

    url = f"{API}/repos/{owner}/{repo}/issues"
    params = {"state": "all", "sort": "updated", "direction": "asc", "since": utc_iso(start), "per_page": 100}
    deadline = time.monotonic() + args.max_runtime_minutes * 60
    pages = issues_seen = comments_seen = 0
    # Capture a safe upper bound so the next run can include updates arriving during this run.
    run_started = datetime.now(timezone.utc)
    latest = start
    for page in paged(session, url, params):
        # Stop only between complete pages; previous page's checkpoint is durable.
        if time.monotonic() > deadline:
            LOG.warning("Time budget reached; resume on next run")
            break
        page_latest = latest
        for issue in page:
            if "pull_request" in issue:
                continue
            updated = parse_time(issue["updated_at"])
            comments = comments_for_issue(session, issue)
            issue_number = issue["number"]
            # Deterministic snapshot key: safe to overwrite during overlapping replays.
            # The comment payload is refreshed even when the issue timestamp is unchanged.
            key = f"github/{owner}/{repo}/issues/{issue_number}/latest.json"
            upload_json(container, key, {
                "source": "github_rest_api",
                "repository": repository,
                "ingested_at": utc_iso(datetime.now(timezone.utc)),
                "issue": issue,
                "comments": comments,
            })
            issues_seen += 1
            comments_seen += len(comments)
            if updated > page_latest:
                page_latest = updated
        pages += 1
        latest = max(latest, page_latest)
        # Do not advance checkpoint past start of this run. Prevents missing late updates.
        safe_watermark = min(latest, run_started)
        # In full mode, don't move an existing incremental cursor backwards.
        if checkpoint:
            safe_watermark = max(safe_watermark, parse_time(checkpoint["watermark"]))
        checkpoint = {
            "repository": repository,
            "watermark": utc_iso(safe_watermark),
            "last_completed_page_at": utc_iso(datetime.now(timezone.utc)),
            "mode": args.mode,
        }
        upload_json(container, checkpoint_path, checkpoint)
        LOG.info("Completed page=%d issues=%d comments=%d watermark=%s", pages, issues_seen, comments_seen, checkpoint["watermark"])
    LOG.info("Finished: pages=%d issues=%d comments=%d", pages, issues_seen, comments_seen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("incremental", "full"), default="incremental")
    parser.add_argument("--overlap-minutes", type=int, default=60)
    parser.add_argument("--max-runtime-minutes", type=int, default=300)
    args = parser.parse_args()
    if args.overlap_minutes < 1 or not 1 <= args.max_runtime_minutes <= 330:
        parser.error("overlap must be positive and runtime must be 1..330 minutes")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
