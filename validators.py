"""
Validators to enforce platform constraints and safety checks on generated text.
"""

from __future__ import annotations

import re
from typing import Dict, List

from config.platform_rules import get_rules


HASHTAG_RE = re.compile(r"(^|\s)#\w+")


def count_hashtags(text: str) -> int:
    return len(HASHTAG_RE.findall(text))


def validate_platform(text: str, platform: str | None) -> Dict[str, object]:
    rules = get_rules(platform)
    issues: List[str] = []
    fixed = text

    # Character limit
    if len(fixed) > rules["max_chars"]:
        fixed = fixed[: rules["max_chars"] - 1]
        issues.append("trimmed_to_max_chars")

    # Hashtag limit
    hcount = count_hashtags(fixed)
    if hcount > rules["hashtags_max"]:
        # remove extra hashtags from the end heuristically
        parts = fixed.split()
        kept = []
        keep_hashtags = rules["hashtags_max"]
        for p in parts:
            if p.startswith("#"):
                if keep_hashtags > 0:
                    kept.append(p)
                    keep_hashtags -= 1
                else:
                    issues.append("removed_extra_hashtags")
                    continue
            else:
                kept.append(p)
        fixed = " ".join(kept)

    # Linebreak policy
    if not rules["linebreaks_ok"]:
        if "\n" in fixed:
            fixed = fixed.replace("\n", " ")
            issues.append("removed_linebreaks")

    return {"text": fixed, "issues": issues, "rules": rules}


