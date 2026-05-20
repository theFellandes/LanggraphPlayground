"""Strip Pydantic-emitted JSON-schema keys that strict-mode endpoints reject.

Background: `langchain_core.utils.function_calling._rm_titles` strips `title`
but leaves `format`, `$ref` (to siblings), `additionalProperties: {}`, and
numeric constraints on nullable types — all of which Mistral and OpenAI
strict-mode endpoints reject. The upstream issue
(https://github.com/langchain-ai/langchain/issues/29604) is closed as
"not planned", so we sanitize ourselves.

See docs/research/langchain-date-field-bug.md for the full investigation.

Usage:
    from shared.llm import with_structured_output_safe

    class Person(BaseModel):
        birth_date: date          # ← native date type works again

    llm = with_structured_output_safe(Person)
    person = llm.invoke("Born April 5, 1987.")
    type(person.birth_date)       # → datetime.date
"""

from __future__ import annotations

import copy
from typing import Any, Type

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from shared.llm.base import get_llm

# Keys we strip outright. These are JSON-Schema keywords that:
#  - some strict-mode providers reject, AND
#  - aren't carrying information the LLM actually needs.
# `format` is the most common offender (date, date-time, email, uri, uuid, etc.).
_STRIP_KEYS: set[str] = {
    "format",         # date / date-time / email / uri / uuid / ipv4 / ...
    "default",         # OpenAI strict mode rejects `default` on required fields
    "examples",        # not in the strict-mode subset
    "$schema",         # meta key that some providers reject
}


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively strip provider-incompatible keys from a JSON schema.

    Pure function: doesn't mutate the input. Use this to clean a
    Pydantic-generated schema before handing it to a structured-output endpoint.

    What it does:
      1. Removes the keys in `_STRIP_KEYS` (most importantly `format`).
      2. Replaces `additionalProperties: {}` with `additionalProperties: False`
         (the strict-mode-friendly equivalent).
      3. Leaves everything else untouched. Conservative on purpose.

    For more aggressive sanitization (nullable + numeric constraints,
    union flattening, $ref resolution) you'd extend this — but those
    transformations also change semantics, so we keep the default minimal.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _STRIP_KEYS:
            continue
        if key == "additionalProperties" and value == {}:
            cleaned[key] = False
            continue
        if isinstance(value, dict):
            cleaned[key] = sanitize_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [
                sanitize_schema(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def sanitize_pydantic_model(model_cls: Type[BaseModel]) -> dict[str, Any]:
    """Convenience: call `sanitize_schema(model_cls.model_json_schema())`."""
    return sanitize_schema(copy.deepcopy(model_cls.model_json_schema()))


def with_structured_output_safe(
    schema: Type[BaseModel] | dict[str, Any],
    *,
    llm: BaseChatModel | Runnable | None = None,
    method: str = "json_schema",
    **kwargs,
) -> Runnable:
    """Drop-in replacement for `llm.with_structured_output(schema)`.

    Sanitizes the schema first (strips `format`, etc.) so providers like
    Mistral and OpenAI strict mode don't 400 on Pydantic-emitted date /
    email / uri fields.

    Args:
        schema: A Pydantic BaseModel class OR a raw JSON-schema dict.
        llm: An optional pre-built LLM. If None, calls `get_llm()`.
        method: Forwarded to `.with_structured_output(method=...)`. Default
            "json_schema". Pass "function_calling" if you prefer tool-calling
            semantics on supported providers.
        **kwargs: Forwarded to `.with_structured_output(...)`.

    Returns:
        A Runnable. `.invoke()` returns an instance of `schema` (if schema is
        a Pydantic class) or a dict (if schema is a raw dict).
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        sanitized: dict[str, Any] | Type[BaseModel] = sanitize_pydantic_model(schema)
        # Keep the BaseModel class for parsing — pass sanitized as a raw schema dict.
        # But .with_structured_output() with a dict returns dicts, not BaseModel
        # instances. So we use a two-step trick: sanitize the schema, but parse the
        # output back into the Pydantic model ourselves.
        base_llm = llm if llm is not None else get_llm(with_fallback=False)
        runnable = base_llm.with_structured_output(sanitized, method=method, **kwargs)

        # Wrap so the dict gets re-validated into the original BaseModel class.
        original_cls = schema

        def _parse(result: dict[str, Any]) -> BaseModel:
            return original_cls.model_validate(result)

        from langchain_core.runnables import RunnableLambda
        return runnable | RunnableLambda(_parse)

    # Raw-dict path: just sanitize and pass through.
    base_llm = llm if llm is not None else get_llm(with_fallback=False)
    return base_llm.with_structured_output(
        sanitize_schema(copy.deepcopy(schema)), method=method, **kwargs
    )
