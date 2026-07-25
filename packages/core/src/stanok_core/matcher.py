"""Fail-fast, budgeted validation of untrusted arguments against a tool's input_schema.

Draft 2020-12 semantics stay jsonschema's: input_schema is the normative truth, so the runtime must
not reinterpret it. What is replaced here is *how* the check is executed.

The stock `anyOf` validates every branch to completion and keeps the errors of all of them, and each
error formats the offending value into its message. The price of one subschema check is therefore
O(instance size), not O(1) - measured on one 936-byte schema, the same number of checks costs 39 ms
against a scalar and 3.7 s against a 1000-element array. That is why counting subschema checks can
never bound the time, however the count is calibrated (K2b).

Here every branch short-circuits at its first error and no error object ever carries a value, so a
step is genuinely constant work and a step counter is a real bound. Two counters are kept, because
the work is not uniform: `descents` for the expensive kind (entering a subschema) and `steps` for
everything, including cheap bulk scans like `required` or `enum`. Both fail closed.

The counters are the normative runtime limit. The registration-time bounds in schema.py are their
mirror image: a schema is only accepted if its own worst case fits inside them, so a legitimate call
can never be rejected for running out of budget.
"""

import contextvars
from collections.abc import Mapping
from itertools import islice
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker, validators
from jsonschema.exceptions import ValidationError

from stanok_core.schema import ALLOWED_FORMATS, MAX_DESCENTS, MAX_VALIDATION_STEPS

_FORMAT_CHECKER = FormatChecker()

# Fail fast: an allowed format without a real checker would silently pass (AD-8).
# The nongpl checkers (uri, hostname, date-time, ...) require jsonschema[format-nongpl].
_missing_checkers = frozenset(ALLOWED_FORMATS) - set(_FORMAT_CHECKER.checkers)
if _missing_checkers:
    raise RuntimeError(
        f"missing jsonschema format checkers {sorted(_missing_checkers)}; "
        "install jsonschema[format-nongpl]"
    )

_MAX_ERROR_PATHS = 3


class ArgumentTooComplex(Exception):
    """Validation ran out of measured budget (or stack). The caller must fail closed."""


# [steps left, descents left]; per-context so concurrent validations can't share a budget.
_budget: contextvars.ContextVar[list[int] | None] = contextvars.ContextVar(
    "stanok_validation_budget", default=None
)


def _spend(steps: int, *, descent: bool = False) -> None:
    cell = _budget.get()
    if cell is None:  # a bounded keyword ran outside failing_paths: refuse rather than run free
        raise ArgumentTooComplex("validation budget not established")
    cell[0] -= steps
    if descent:
        cell[1] -= 1
    if cell[0] < 0 or cell[1] < 0:
        raise ArgumentTooComplex("argument validation budget exhausted")


_STOCK = Draft202012Validator.VALIDATORS


def _any_of(validator: Any, any_of: Any, instance: Any, schema: Any) -> Any:
    """Accept on the first matching branch and never collect per-branch errors.

    Same verdict as the stock keyword (valid iff some branch validates); only the error detail
    differs, and it is deliberately value-free. descend() is lazy, so taking just its first error
    abandons the rest of that branch instead of materializing it."""
    for index, subschema in enumerate(any_of):
        _spend(1, descent=True)
        errors = validator.descend(instance, subschema, schema_path=index)
        try:
            failed = next(errors, None) is not None
        finally:
            errors.close()  # don't leave the branch's generator to be finalized whenever
        if not failed:
            return
    yield ValidationError("value does not match any allowed variant")


def _ref(validator: Any, ref: Any, instance: Any, schema: Any) -> Any:
    _spend(1, descent=True)
    yield from _STOCK["$ref"](validator, ref, instance, schema)


def _properties(validator: Any, properties: Any, instance: Any, schema: Any) -> Any:
    if validator.is_type(instance, "object"):
        _spend(len(properties), descent=True)
    yield from _STOCK["properties"](validator, properties, instance, schema)


def _items(validator: Any, items: Any, instance: Any, schema: Any) -> Any:
    # Charged for the whole array up front: jsonschema descends into every element regardless of
    # maxItems, so an over-long array is paid for before it is rejected.
    if validator.is_type(instance, "array"):
        _spend(len(instance), descent=True)
    yield from _STOCK["items"](validator, items, instance, schema)


def _required(validator: Any, required: Any, instance: Any, schema: Any) -> Any:
    # Every required key is checked even against {} - the cheap scan that made the old cost model
    # underestimate a wide object.
    _spend(len(required))
    yield from _STOCK["required"](validator, required, instance, schema)


def _additional_properties(validator: Any, a_p: Any, instance: Any, schema: Any) -> Any:
    if validator.is_type(instance, "object"):
        _spend(len(instance))
    yield from _STOCK["additionalProperties"](validator, a_p, instance, schema)


def _enum(validator: Any, enums: Any, instance: Any, schema: Any) -> Any:
    _spend(len(enums))
    yield from _STOCK["enum"](validator, enums, instance, schema)


# Keywords whose work is charged to the budget; every descending keyword must be in here.
_METERED = frozenset(
    {"anyOf", "$ref", "properties", "items", "required", "additionalProperties", "enum"}
)

_BoundedValidator = validators.extend(  # type: ignore[no-untyped-call]  # jsonschema is untyped here
    Draft202012Validator,
    validators={
        "anyOf": _any_of,
        "$ref": _ref,
        "properties": _properties,
        "items": _items,
        "required": _required,
        "additionalProperties": _additional_properties,
        "enum": _enum,
    },
)


def failing_paths(schema: Mapping[str, Any], instance: Any) -> list[str]:
    """Validate `instance` against `schema` under a measured budget.

    Returns [] when the instance is valid, otherwise up to three JSON paths locating what failed -
    paths only, never values (AD-7, AD-8). Raises ArgumentTooComplex when the budget runs out, which
    the caller must turn into a rejection.
    """
    token = _budget.set([MAX_VALIDATION_STEPS, MAX_DESCENTS])
    try:
        validator = _BoundedValidator(dict(schema), format_checker=_FORMAT_CHECKER)
        errors = list(islice(validator.iter_errors(instance), _MAX_ERROR_PATHS))
    except RecursionError as exc:
        # Depth is bounded at registration; this is the second line of defence, not the first.
        raise ArgumentTooComplex("argument validation exceeded stack depth") from exc
    finally:
        _budget.reset(token)
    return ["/".join(str(p) for p in error.absolute_path) or "(root)" for error in errors]
