import re
from decimal import Decimal

CURRENCY = r"(?:\u20B1|PHP\s*)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"

def detect_budget_request(question):
    text = str(question or "").strip()

    planning = re.search(
        r"\b(?:create|make|build|prepare|draft|develop|plan|"
        r"allocate|breakdown|design|revise|correct)\b",
        text,
        re.I
    )

    topic = re.search(
        r"\b(?:budget|business plan|spending plan|"
        r"startup plan|financial plan)\b",
        text,
        re.I
    )

    if not (planning and topic):
        return None

    patterns = [
        r"\b(?:total|overall|maximum|max)\s+"
        r"(?:budget|limit|amount)\s*(?:of|is|:)?\s*" + CURRENCY,

        r"\bbudget\s+(?:of|is|:)\s*" + CURRENCY,

        r"\b(?:limit|capital)\s+(?:of|is|:)\s*" + CURRENCY,

        CURRENCY + r"\s+(?:business|startup|spending|financial)?\s*budget\b",
    ]

    selected = None

    for pattern in patterns:
        match = re.search(pattern, text, re.I)
        if match:
            selected = match.group(1)
            break

    if selected is None:
        amounts = re.findall(CURRENCY, text, re.I)

        if len(amounts) != 1:
            return None

        selected = amounts[0]

    limit = Decimal(selected.replace(",", ""))

    if not limit.is_finite() or limit <= 0:
        return None

    return {
        "budget_limit": limit,
        "question": text
    }
