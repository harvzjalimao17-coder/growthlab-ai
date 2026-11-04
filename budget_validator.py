from decimal import Decimal, InvalidOperation

def amount(value):
    try:
        result = Decimal(str(value).replace(",", "").replace("₱", "").strip())
        if not result.is_finite() or result < 0:
            raise ValueError("Amounts must be finite and non-negative.")
        return result
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid amount: {value}") from exc


def validate_budget(limit, categories):
    limit = amount(limit)
    errors = []
    total = Decimal("0")

    for category in categories:
        name = category["name"]
        allocation = amount(category["allocation"])
        total += allocation

        if "items" in category:
            item_total = sum(
                (amount(item["amount"]) for item in category["items"]),
                Decimal("0")
            )

            if item_total != allocation:
                errors.append(
                    f"{name}: item total {item_total} "
                    f"does not match allocation {allocation}"
                )

    if total > limit:
        errors.append(
            f"Total {total} exceeds budget {limit} by {total - limit}"
        )

    return {
        "valid": len(errors) == 0,
        "limit": limit,
        "total": total,
        "remaining": limit - total,
        "errors": errors,
    }
