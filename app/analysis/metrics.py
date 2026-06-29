from __future__ import annotations

from decimal import Decimal


def decimal_or_zero(value: object) -> Decimal:
    if value is None:
        return Decimal("0")
    return Decimal(str(value))


def calculate_rate(numerator: int, denominator: int) -> Decimal | None:
    if denominator <= 0:
        return None
    return Decimal(numerator) / Decimal(denominator)
