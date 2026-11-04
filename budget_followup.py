import re
from budget_request_detector import detect_budget_request

FOLLOWUP_PATTERN = re.compile(
    r"\b(?:correct|revise|adjust|fix|rebalance|"
    r"recalculate|update)\b",
    re.I
)

BUDGET_REFERENCE = re.compile(
    r"\b(?:budget|allocation|allocations|"
    r"calculation|calculations|total|totals|"
    r"expenses|costs|spending)\b",
    re.I
)

def detect_budget_followup(question, history):
    text = str(question or "").strip()

    if not FOLLOWUP_PATTERN.search(text):
        return None

    if not BUDGET_REFERENCE.search(text):
        return None

    previous_users = [
        message
        for message in (history or [])
        if isinstance(message, dict)
        and message.get("role") == "user"
    ]

    if not previous_users:
        return None

    for message in reversed(previous_users):
        content = str(message.get("content", "")).strip()

        previous = detect_budget_request(content)

        if previous:
            return {
                "budget_limit": previous["budget_limit"],
                "question": text,
                "previous_request": previous["question"],
                "is_followup": True,
            }

        if not (
            FOLLOWUP_PATTERN.search(content)
            and BUDGET_REFERENCE.search(content)
        ):
            return None

    return None
