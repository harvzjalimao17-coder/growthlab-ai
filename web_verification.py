"""Official-source extraction helpers for GrowthLab AI."""

import re
from datetime import datetime


def extract_python_release(sources):
    candidates = []

    for source in sources:
        url = str(source.get("url", ""))
        title = str(source.get("title", ""))
        snippet = str(source.get("snippet", ""))

        if url.rstrip("/") != "https://www.python.org/downloads/latest":
            continue

        version_match = re.search(
            r"Python Release Python (\d+)\.(\d+)\.(\d+)(?![a-zA-Z0-9])",
            title,
            re.I
        )

        date_match = re.search(
            r"Release date:\s*([A-Za-z]+\.?\s+\d{1,2},\s+\d{4})",
            snippet,
            re.I
        )

        if not version_match or not date_match:
            continue

        version_tuple = tuple(
            int(value) for value in version_match.groups()
        )

        date_text = date_match.group(1).replace(".", "")

        release_date = None

        for fmt in ("%b %d, %Y", "%B %d, %Y"):
            try:
                release_date = datetime.strptime(
                    date_text, fmt
                ).date()
                break
            except ValueError:
                continue

        if release_date is None:
            continue

        candidates.append({
            "version": ".".join(map(str, version_tuple)),
            "version_tuple": version_tuple,
            "date": release_date,
            "url": url
        })

    if not candidates:
        return None

    selected = max(
        candidates,
        key=lambda item: item["version_tuple"]
    )

    return {
        "version": selected["version"],
        "date": selected["date"],
        "url": selected["url"]
    }
