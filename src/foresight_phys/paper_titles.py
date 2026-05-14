from __future__ import annotations

import html
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ARXIV_ID_PATTERN = re.compile(r'^(?P<id>\d{4}\.\d{4,5})(?:v\d+)?$', re.IGNORECASE)
ARXIV_CITATION_TITLE_PATTERN = re.compile(
    r'<meta[^>]+name=["\']citation_title["\'][^>]+content=["\'](?P<title>.*?)["\']',
    re.IGNORECASE | re.DOTALL,
)
ARXIV_PAGE_TITLE_PATTERN = re.compile(
    r'<title>(?P<title>.*?)</title>',
    re.IGNORECASE | re.DOTALL,
)
DEFAULT_ARXIV_TITLE_CACHE_PATH = Path('.cache/arxiv_titles.json')
ARXIV_ABS_URL_TEMPLATE = 'https://arxiv.org/abs/{arxiv_id}'
USER_AGENT = 'Foresight-Phys/0.1 (+https://github.com/kazeevn/Foresight-Phys)'


def normalize_arxiv_id(value: str) -> str | None:
    match = ARXIV_ID_PATTERN.fullmatch(value.strip())
    if match is None:
        return None
    return match.group('id')


def extract_arxiv_id_from_file_name(file_name: str) -> str | None:
    return normalize_arxiv_id(Path(file_name).stem)


def clean_title_text(value: str) -> str:
    return ' '.join(html.unescape(value).split())


def parse_arxiv_title_html(html_text: str) -> str | None:
    meta_match = ARXIV_CITATION_TITLE_PATTERN.search(html_text)
    if meta_match is not None:
        title = clean_title_text(meta_match.group('title'))
        if title:
            return title

    title_match = ARXIV_PAGE_TITLE_PATTERN.search(html_text)
    if title_match is None:
        return None

    title = clean_title_text(title_match.group('title'))
    if not title:
        return None

    if title.lower().startswith('arxiv:') and ']' in title:
        title = title.split(']', 1)[1].strip()
    return title or None


def load_arxiv_title_cache(cache_path: Path = DEFAULT_ARXIV_TITLE_CACHE_PATH) -> dict[str, str]:
    if not cache_path.exists():
        return {}

    try:
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}

    if not isinstance(payload, dict):
        return {}

    cache: dict[str, str] = {}
    for key, value in payload.items():
        normalized_id = normalize_arxiv_id(str(key))
        if normalized_id is None or not isinstance(value, str):
            continue

        cleaned_title = value.strip()
        if cleaned_title:
            cache[normalized_id] = cleaned_title

    return cache


def save_arxiv_title_cache(
    cache: dict[str, str],
    cache_path: Path = DEFAULT_ARXIV_TITLE_CACHE_PATH,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(dict(sorted(cache.items())), ensure_ascii=False, indent=2),
        encoding='utf-8',
    )


def fetch_arxiv_title(arxiv_id: str) -> str | None:
    normalized_id = normalize_arxiv_id(arxiv_id)
    if normalized_id is None:
        return None

    request = Request(
        ARXIV_ABS_URL_TEMPLATE.format(arxiv_id=normalized_id),
        headers={'User-Agent': USER_AGENT},
    )

    try:
        with urlopen(request, timeout=20) as response:
            html_text = response.read().decode('utf-8', errors='replace')
    except (HTTPError, URLError, TimeoutError, OSError):
        return None

    return parse_arxiv_title_html(html_text)


def resolve_arxiv_titles(
    arxiv_ids: list[str],
    *,
    cache_path: Path = DEFAULT_ARXIV_TITLE_CACHE_PATH,
) -> dict[str, str]:
    normalized_ids: list[str] = []
    seen_ids: set[str] = set()
    for arxiv_id in arxiv_ids:
        normalized_id = normalize_arxiv_id(arxiv_id)
        if normalized_id is None or normalized_id in seen_ids:
            continue
        seen_ids.add(normalized_id)
        normalized_ids.append(normalized_id)

    if not normalized_ids:
        return {}

    cache = load_arxiv_title_cache(cache_path)
    cache_updated = False
    for arxiv_id in normalized_ids:
        if arxiv_id in cache:
            continue

        title = fetch_arxiv_title(arxiv_id)
        if title is None:
            continue

        cache[arxiv_id] = title
        cache_updated = True

    if cache_updated:
        save_arxiv_title_cache(cache, cache_path)

    return {
        arxiv_id: cache[arxiv_id]
        for arxiv_id in normalized_ids
        if arxiv_id in cache
    }