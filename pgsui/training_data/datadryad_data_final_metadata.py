#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Harvest Dryad dataset metadata via the Dryad v2 API, with robust backoff.

This script:
1. Queries Dryad's /search endpoint for multiple terms.
2. Paginates through all results per term.
3. Deduplicates datasets across terms.
4. Extracts title, keywords, abstract, DOI, and authors into a CSV.
5. Optionally downloads all public files for each unique dataset.
6. Filters a subset of "genomic" datasets based on keyword/field matching.

Authentication:
- Authorization: Bearer <token>
"""

from __future__ import annotations

import argparse
import logging
import os
import time
import random
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional
from urllib.parse import unquote_plus, urljoin, urlparse

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
LOGGER = logging.getLogger("dryad_harvest")


class RateLimiter:
    """Strict Rate Limiter with absolute spacing."""

    def __init__(self, max_calls_per_minute: int = 20) -> None:
        """
        Args:
            max_calls_per_minute: Max RPM.
        """
        self.interval = 60.0 / float(max_calls_per_minute)
        self.last_call = 0.0

    def acquire(self) -> None:
        """Block until a new request is permitted."""
        now = time.time()
        elapsed = now - self.last_call
        if elapsed < self.interval:
            sleep_time = self.interval - elapsed
            time.sleep(sleep_time)
        self.last_call = time.time()


class DryadClient:
    """Dryad API v2 client with exponential backoff and strict rate limiting."""

    def __init__(
        self,
        base_url: str = "https://datadryad.org/api/v2/",
        per_page: int = 100,
        timeout: int = 60,
        max_requests_per_minute: int = 10,
        user_agent: str = "dryad-harvest-script/2.1 (academic use)",
        api_token: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.per_page = per_page
        self.timeout = timeout

        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

        self.headers = {
            "Accept": "application/json",
            "User-Agent": user_agent,
        }

        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"

        self.rate_limiter = RateLimiter(max_calls_per_minute=max_requests_per_minute)

    def _make_url(self, endpoint_or_url: str | None) -> str:
        if endpoint_or_url is None:
            raise ValueError("Cannot make URL from None")
        if endpoint_or_url.startswith("http"):
            return endpoint_or_url

        root = f"{urlparse(self.base_url).scheme}://{urlparse(self.base_url).netloc}/"
        if endpoint_or_url.startswith("/"):
            return urljoin(root, endpoint_or_url)
        if endpoint_or_url.startswith("api/"):
            return urljoin(root, endpoint_or_url)
        return urljoin(self.base_url, endpoint_or_url)

    def _request(
        self, endpoint_or_url: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """GET JSON with exponential backoff for 429s."""
        url = self._make_url(endpoint_or_url)
        params = params or {}

        attempt = 0
        max_attempts = 5

        while attempt < max_attempts:
            self.rate_limiter.acquire()

            try:
                r = self.session.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=self.timeout,
                )

                if r.status_code == 429:
                    LOGGER.warning(
                        f"Received 429 Too Many Requests. Waiting for {int(r.headers['Retry-After']) + 1} duration..."
                    )

                    if r.headers.get("Retry-After") is not None:
                        time.sleep(int(r.headers["Retry-After"]) + 1)
                        continue
                    else:
                        attempt += 1
                        # Exponential backoff: 5s, 10s, 20s...
                        wait_time = (5 * (2 ** (attempt - 1))) + random.uniform(0, 1)
                        LOGGER.warning(
                            f"429 Too Many Requests. Sleeping {wait_time:.1f}s..."
                        )
                        time.sleep(wait_time)
                        continue

                if r.status_code in {401, 403}:
                    # If this happens, your token is likely bad
                    raise RuntimeError(f"Auth failed (401/403): {r.text}")

                r.raise_for_status()
                return r.json()

            except requests.exceptions.RequestException as e:
                LOGGER.warning(f"Request failed: {e}. Retrying...")
                attempt += 1
                time.sleep(5)

        raise RuntimeError(f"Failed to fetch {url} after {max_attempts} attempts.")

    def search_page(self, term: str, page: int) -> Dict[str, Any]:
        return self._request(
            "search", {"q": term, "page": page, "per_page": self.per_page}
        )

    def term_total(self, term: str) -> int:
        data = self.search_page(term, 1)
        return int(data.get("total", 0))

    def iter_datasets(self, term: str) -> Iterable[Dict[str, Any]]:
        page = 1
        while True:
            data = self.search_page(term, page)
            datasets = data.get("_embedded", {}).get("stash:datasets", [])
            if not datasets:
                break
            yield from datasets
            page += 1

    def get_full_dataset(self, summary_ds: Dict[str, Any]) -> Dict[str, Any]:
        self_href = summary_ds.get("_links", {}).get("self", {}).get("href")
        if not self_href:
            return summary_ds
        try:
            return self._request(self_href)
        except Exception as e:
            LOGGER.warning("Failed to fetch full dataset %s: %s", self_href, str(e))
            return summary_ds

    def iter_files_for_dataset(
        self, ds_full: Dict[str, Any]
    ) -> Iterable[Dict[str, Any]]:
        """
        Robustly find files by checking:
        1. Embedded files (rare but possible)
        2. Direct 'stash:files' link
        3. 'stash:currentVersion' link -> files
        4. 'stash:versions' link -> latest version -> files
        """

        def fetch_files_from_href(href):
            if not href:
                return []
            try:
                # Handle pagination of files
                next_url = href
                while next_url:
                    d = self._request(next_url)
                    yield from d.get("_embedded", {}).get("stash:files", [])
                    next_url = d.get("_links", {}).get("next", {}).get("href")
            except Exception:
                return []

        # Strategy 1: Already embedded?
        embedded = ds_full.get("_embedded", {}).get("stash:files", [])
        if embedded:
            yield from embedded
            return

        # Strategy 2: Direct file link
        files_href = ds_full.get("_links", {}).get("stash:files", {}).get("href")
        if files_href:
            yield from fetch_files_from_href(files_href)
            return

        # Strategy 3: Current Version
        curr_ver_href = (
            ds_full.get("_links", {}).get("stash:currentVersion", {}).get("href")
        )
        if curr_ver_href:
            try:
                ver_data = self._request(curr_ver_href)
                v_files_href = (
                    ver_data.get("_links", {}).get("stash:files", {}).get("href")
                )
                if v_files_href:
                    yield from fetch_files_from_href(v_files_href)
                    return
            except Exception:
                pass

        # Strategy 4: Version List (Find the latest one)
        versions_href = ds_full.get("_links", {}).get("stash:versions", {}).get("href")
        if versions_href:
            try:
                v_list_data = self._request(versions_href)
                versions = v_list_data.get("_embedded", {}).get("stash:versions", [])

                # Sort versions by versionNumber just in case
                def get_vnum(v):
                    return int(v.get("versionNumber", 0))

                if versions:
                    latest = sorted(versions, key=get_vnum)[-1]
                    latest_self = latest.get("_links", {}).get("self", {}).get("href")
                    if latest_self:
                        latest_data = self._request(latest_self)
                        l_files_href = (
                            latest_data.get("_links", {})
                            .get("stash:files", {})
                            .get("href")
                        )
                        if l_files_href:
                            yield from fetch_files_from_href(l_files_href)
                            return
            except Exception:
                pass

        return

    def download_file(
        self,
        file_obj: Dict[str, Any],
        out_dir: Path,
        overwrite: bool = False,
    ) -> Optional[Path]:
        links = file_obj.get("_links", {}) or {}
        download_href = links.get("stash:file-download", {}).get("href") or links.get(
            "stash:download", {}
        ).get("href")

        if not download_href:
            return None

        filename = (
            file_obj.get("path")
            or file_obj.get("filename")
            or file_obj.get("name")
            or "dryad_file"
        )
        filename = Path(str(filename)).name

        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        # This is where overwrite is checked.
        # If overwrite is False and file exists, we return success immediately.
        if out_path.exists() and not overwrite:
            return out_path

        url = self._make_url(download_href)

        self.rate_limiter.acquire()
        try:
            with self.session.get(url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                with open(out_path, "wb") as fout:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fout.write(chunk)
            return out_path
        except Exception as e:
            LOGGER.error(f"Failed download {filename}: {e}")
            return None


def normalize_term(term: str) -> str:
    return unquote_plus(term)


def extract_dataset_row(ds: Dict[str, Any], record_num: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {"DatasetID": record_num}

    identifier = ds.get("identifier")
    doi: Optional[str] = None
    if isinstance(identifier, str):
        doi = identifier[4:] if identifier.lower().startswith("doi:") else identifier

    if doi is None:
        href = ds.get("_links", {}).get("stash:download", {}).get("href")
        if isinstance(href, str):
            doi = "/".join(href.split("/")[-3:])

    row["DOI"] = doi if doi else pd.NA
    row["Title"] = ds.get("title", pd.NA)

    keywords = ds.get("keywords") or []
    if isinstance(keywords, list):
        row["Keywords"] = ",".join([str(k) for k in keywords])
    else:
        row["Keywords"] = str(keywords) if keywords else pd.NA

    row["Abstract"] = ds.get("abstract", pd.NA)

    authors = ds.get("authors") or []
    if isinstance(authors, list) and authors:
        for i, auth in enumerate(authors, start=1):
            if not isinstance(auth, dict):
                continue
            for k, v in auth.items():
                row[f"{k}_{i}"] = v
    else:
        # Fill first author blanks
        for col in ["firstName_1", "lastName_1", "email_1", "affiliation_1"]:
            row[col] = pd.NA

    return row


def filter_genomic_datasets(df: pd.DataFrame) -> pd.DataFrame:
    contains_list = [
        "vcf",
        "phylip",
        "phy",
        "genepop",
        "rad-seq",
        "ddrad",
        "double-digest-rad",
        "genotyping-by-sequencing",
        "stacks",
        "ipyrad",
        "pyrad",
        "radseq",
        "gbs",
    ]
    pattern = "|".join(contains_list)

    # Vectorized string search
    mask = df.apply(
        lambda r: r.astype(str)
        .str.lower()
        .str.contains(pattern.lower(), regex=True)
        .any(),
        axis=1,
    )
    return df[mask].copy()


def should_skip_file(
    file_obj: Dict[str, Any],
    max_size_mb: Optional[float],
    allowed_ext: Optional[List[str]],
) -> tuple[bool, str]:
    """Return (True, reason) if file should be skipped."""
    name = (
        file_obj.get("path") or file_obj.get("filename") or file_obj.get("name") or ""
    )

    if allowed_ext:
        ext = Path(str(name)).suffix.lower()
        # Handle .vcf.gz double extension if needed, but simple suffix matches standard usage
        if not ext:
            return True, "No Extension"
        if ext not in allowed_ext:
            return True, f"Bad Extension ({ext})"

    if max_size_mb is not None:
        size_bytes = file_obj.get("size")
        if size_bytes is not None:
            try:
                size_mb = float(size_bytes) / (1024.0 * 1024.0)
                if size_mb > max_size_mb:
                    return True, f"Too Large ({size_mb:.2f} MB)"
            except (TypeError, ValueError):
                pass

    return False, ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Harvest Dryad metadata and optionally download dataset files."
    )
    parser.add_argument("--download-files", action="store_true")
    parser.add_argument("--files-outdir", type=str, default="training_data/files")
    parser.add_argument("--max-file-mb", type=float, default=None)
    parser.add_argument(
        "--allowed-ext",
        type=str,
        default=None,
        help="Comma-separated extensions, e.g. .vcf,.vcf.gz,.phylip",
    )
    parser.add_argument("--overwrite-files", action="store_true")
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument(
        "--resume-from",
        type=int,
        default=1,
        help="Resume at this results index (e.g., 2500 means start at results2500).",
    )
    parser.add_argument(
        "--api-token",
        type=str,
        default=None,
    )
    # Retained for compatibility but unused due to exponential backoff
    parser.add_argument("--fixed-wait", type=float, default=5.0)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    api_token = args.api_token or os.environ.get("DRYAD_API_TOKEN")
    if not api_token:
        LOGGER.warning(
            "No API Token provided. Rate limits will be strict and downloads may fail."
        )

    searches = [
        "vcf",
        "phylip",
        "phy",
        "genepop",
        "rad-seq",
        "ddrad",
        "double-digest-rad",
        "genotyping-by-sequencing",
        "stacks",
        "ipyrad",
        "pyrad",
        "radseq",
        "gbs",
    ]
    terms = [normalize_term(t) for t in searches]

    client = DryadClient(
        max_requests_per_minute=args.rpm,
        api_token=api_token,
    )

    allowed_ext = None
    if args.allowed_ext:
        allowed_ext = [
            e.strip().lower() for e in args.allowed_ext.split(",") if e.strip()
        ]

    # Estimate total
    print("Estimating totals per term...")
    total_raw = sum(client.term_total(term) for term in terms)
    print(f"Parsing approximately {total_raw} datasets (before deduplication)...")

    rows: List[Dict[str, Any]] = []
    seen: set = set()

    # Resume logic
    counter = 1
    resume_target = args.resume_from

    files_base = Path(args.files_outdir)
    if args.download_files:
        files_base.mkdir(parents=True, exist_ok=True)

    with tqdm(total=total_raw, unit="ds") as pbar:
        for term in terms:
            for ds_summary in client.iter_datasets(term):
                pbar.update(1)

                unique_key = (
                    ds_summary.get("id")
                    or ds_summary.get("identifier")
                    or ds_summary.get("_links", {}).get("self", {}).get("href")
                )
                if unique_key in seen:
                    continue
                seen.add(unique_key)

                # Resume Check
                if counter < resume_target:
                    counter += 1
                    continue

                record_num = f"results{counter}"

                # 1. Fetch Full Metadata (Cost: 1 API Call)
                try:
                    ds_full = client.get_full_dataset(ds_summary)
                    rows.append(extract_dataset_row(ds_full, record_num))
                except Exception as e:
                    LOGGER.error(f"Skipping metadata for {record_num}: {e}")
                    counter += 1
                    continue

                # 2. Download Files (Cost: N API Calls)
                if args.download_files:
                    dataset_dir = files_base / record_num

                    # We pass ds_full to avoid re-fetching
                    files_iter = client.iter_files_for_dataset(ds_full)
                    files = list(files_iter)

                    if files:
                        LOGGER.info(f"Dataset {record_num}: Found {len(files)} files.")
                        for fobj in files:
                            skip, reason = should_skip_file(
                                fobj, args.max_file_mb, allowed_ext
                            )
                            if skip:
                                LOGGER.info(
                                    f"  Skipped {fobj.get('path', 'unknown')}: {reason}"
                                )
                                continue

                            try:
                                out = client.download_file(
                                    fobj, dataset_dir, args.overwrite_files
                                )
                                if out:
                                    LOGGER.info(f"  Downloaded: {out.name}")
                            except Exception as e:
                                LOGGER.warning(f"  Failed {record_num}: {e}")
                    else:
                        LOGGER.info(f"Dataset {record_num}: No accessible files found.")

                counter += 1

    df = pd.DataFrame(rows)
    out_dir = Path("training_data/metadata")
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = out_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False, encoding="utf-8")

    genomic_df = filter_genomic_datasets(df)
    genomic_df.to_csv(
        out_dir / "metadata_genomic_data.csv", index=False, encoding="utf-8"
    )

    print(f"Done. Saved {len(df)} total records, {len(genomic_df)} genomic records.")
    if args.download_files:
        print(f"Files saved to: {files_base.resolve()}")


if __name__ == "__main__":
    main()
