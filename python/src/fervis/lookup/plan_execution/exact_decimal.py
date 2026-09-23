"""Stable decimal arithmetic for factual values from relational sources."""

from decimal import Context, Decimal, localcontext
from fervis.lookup.plan_execution.errors import RelationEngineError


_MIN_DIVISION_PRECISION = 50
_MAX_EXACT_PRECISION = 10_000


def _value_span(value: Decimal) -> tuple[int, int]:
    if not value.is_finite():
        raise RelationEngineError("non-finite decimal cannot enter factual arithmetic")
    integer = max(1, value.adjusted() + 1) if value else 1
    fraction = max(0, -int(value.as_tuple().exponent))
    return integer, fraction


def _precision(values: tuple[Decimal, ...], *, product: bool = False) -> int:
    spans = tuple(_value_span(value) for value in values)
    if product:
        needed = sum(integer + fraction for integer, fraction in spans) + 2
    else:
        needed = (
            max(integer for integer, _ in spans)
            + max(fraction for _, fraction in spans)
            + len(str(len(values))) + 2
        )
    if needed > _MAX_EXACT_PRECISION:
        raise RelationEngineError("exact numeric precision exceeds the supported bound")
    return max(needed, 28)


def exact_sum(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        return Decimal(0)
    with localcontext(Context(prec=_precision(values))):
        return sum(values, Decimal(0))


def exact_add(left: Decimal, right: Decimal) -> Decimal:
    return exact_sum((left, right))


def exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    return exact_sum((left, right.copy_negate()))


def exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    with localcontext(Context(prec=_precision((left, right), product=True))):
        return left * right


def stable_divide(left: Decimal, right: Decimal) -> Decimal:
    precision = _precision((left, right))
    if precision + 20 > _MAX_EXACT_PRECISION:
        raise RelationEngineError("decimal division precision exceeds the supported bound")
    with localcontext(Context(prec=max(_MIN_DIVISION_PRECISION, precision + 20))):
        return left / right
