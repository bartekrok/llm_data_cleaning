"""Prompt builders for the experiment harness.

Every builder returns a (messages, response_format) tuple:

    messages: list[dict[str, Any]]   - ready for OpenRouter
    response_format: dict | None     - json_schema dict, json_object, or None

Builders never know which model they will be sent to. The
OpenRouterCleaner is responsible for falling back from json_schema to
json_object to text when a model doesn't support structured outputs.

Variants:
    P0_BASELINE        - current prompt from script.py
    P1_FEW_SHOT        - same + 3 worked examples
    P2_COT             - same + "think step by step"
    P3_STRICT_SCHEMA   - structured outputs via response_format
    P4_MINIMAL         - terse, schema-only
    P_WITH_CONFIDENCE  - P0 + extra confidence field (for exp08)

Each builder also has a *_batch counterpart that takes a list of
values and asks for a JSON array of results.
"""

from __future__ import annotations

from typing import Any, Sequence

P0_BASELINE = "P0_baseline"
P1_FEW_SHOT = "P1_few_shot"
P2_COT = "P2_cot"
P3_STRICT_SCHEMA = "P3_strict_schema"
P4_MINIMAL = "P4_minimal"
P_WITH_CONFIDENCE = "P_with_confidence"

ALL_PROMPT_VARIANTS = (
    P0_BASELINE,
    P1_FEW_SHOT,
    P2_COT,
    P3_STRICT_SCHEMA,
    P4_MINIMAL,
)


SINGLE_SCHEMA = {
    "name": "data_cleaning_decision",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["acceptance", "decline", "suggest"]},
            "message": {"type": "string"},
            "value": {"type": "string"},
        },
        "required": ["state", "message", "value"],
        "additionalProperties": False,
    },
}

SINGLE_SCHEMA_WITH_CONFIDENCE = {
    "name": "data_cleaning_decision_with_confidence",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "state": {"type": "string", "enum": ["acceptance", "decline", "suggest"]},
            "message": {"type": "string"},
            "value": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": ["state", "message", "value", "confidence"],
        "additionalProperties": False,
    },
}

BATCH_SCHEMA = {
    "name": "data_cleaning_batch",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": SINGLE_SCHEMA["schema"],
            }
        },
        "required": ["results"],
        "additionalProperties": False,
    },
}


BASELINE_SYSTEM = """You are an automated data ingestion assistant.
Your allowed scope of values is: [{scope}].

Evaluate the user's input and respond strictly in JSON format with exactly three keys: "state", "message", and "value".
Do not wrap your response in markdown blocks (e.g., ```json).

Rules:
1. "state": "acceptance"
   - Use when: The input matches something in the scope (allowing for typos, case differences, or clear synonyms).
   - "message": MUST say "This value is good and should be named [Standardized Name] because [Your reason]".
   - "value": MUST be the exact Standardized Name from the scope.

2. "state": "decline"
   - Use when: The input is garbage, a completely different category, or invalid data.
   - "message": MUST explain why it shouldn't be ingested.
   - "value": MUST be an empty string "".

3. "state": "suggest"
   - Use when: The input is a valid item of the same category (e.g., a fruit) but is NOT in the scope.
   - "message": MUST say "This value should be added to our scope because [Your reason]".
   - "value": MUST be the cleaned name of the suggested new item.
"""

MINIMAL_SYSTEM = """Allowed scope: [{scope}].
Return JSON: {{"state":"acceptance"|"decline"|"suggest","message":"...","value":"..."}}.
acceptance -> value is the canonical scope name.
decline -> value is "".
suggest -> value is the cleaned new name.
"""

COT_SUFFIX = "\n\nThink step by step before responding, but output ONLY the final JSON object as your answer (no preamble, no markdown)."

CONFIDENCE_SUFFIX = (
    '\n\nAdditionally include a "confidence" key in the JSON with a number between 0 and 1 '
    "indicating how certain you are about your decision."
)

FEW_SHOT_EXAMPLES = [
    {
        "scope": ["Information Technology", "HR", "Finance"],
        "input": "IT",
        "output": {
            "state": "acceptance",
            "message": "This value is good and should be named Information Technology because IT is the standard abbreviation.",
            "value": "Information Technology",
        },
    },
    {
        "scope": ["Red", "Blue", "Green"],
        "input": "Elephant",
        "output": {
            "state": "decline",
            "message": "Elephant is an animal, not a color, so it does not belong in the color scope.",
            "value": "",
        },
    },
    {
        "scope": ["Laptop", "Smartphone", "Tablet"],
        "input": "Smartwatch",
        "output": {
            "state": "suggest",
            "message": "This value should be added to our scope because a smartwatch is a consumer electronic device similar to the existing scope items.",
            "value": "Smartwatch",
        },
    },
]


def _scope_str(scope: Sequence[str]) -> str:
    return ", ".join(scope)


def _user_single(value: str) -> dict[str, Any]:
    return {"role": "user", "content": f"Value to evaluate: {value}"}


def _user_batch(values: Sequence[str]) -> dict[str, Any]:
    enumerated = "\n".join(f"{i+1}. {v}" for i, v in enumerate(values))
    return {
        "role": "user",
        "content": (
            "Evaluate each of the following values independently. "
            'Respond with a single JSON object {"results": [...]} where the array has exactly '
            f"{len(values)} elements in the same order as the inputs, each element following the per-value schema.\n\n"
            f"Inputs:\n{enumerated}"
        ),
    }


def _few_shot_messages() -> list[dict[str, Any]]:
    import json

    out: list[dict[str, Any]] = []
    for ex in FEW_SHOT_EXAMPLES:
        out.append(
            {
                "role": "user",
                "content": f"Scope: [{', '.join(ex['scope'])}]\nValue to evaluate: {ex['input']}",
            }
        )
        out.append({"role": "assistant", "content": json.dumps(ex["output"])})
    return out


def build_single(variant: str, value: str, scope: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (messages, response_format) for a single-value evaluation."""
    scope_str = _scope_str(scope)

    if variant == P0_BASELINE:
        return (
            [
                {"role": "system", "content": BASELINE_SYSTEM.format(scope=scope_str)},
                _user_single(value),
            ],
            None,
        )

    if variant == P1_FEW_SHOT:
        return (
            [
                {"role": "system", "content": BASELINE_SYSTEM.format(scope=scope_str)},
                *_few_shot_messages(),
                _user_single(value),
            ],
            None,
        )

    if variant == P2_COT:
        sys = BASELINE_SYSTEM.format(scope=scope_str) + COT_SUFFIX
        return [{"role": "system", "content": sys}, _user_single(value)], None

    if variant == P3_STRICT_SCHEMA:
        return (
            [
                {"role": "system", "content": BASELINE_SYSTEM.format(scope=scope_str)},
                _user_single(value),
            ],
            {"type": "json_schema", "json_schema": SINGLE_SCHEMA},
        )

    if variant == P4_MINIMAL:
        return (
            [
                {"role": "system", "content": MINIMAL_SYSTEM.format(scope=scope_str)},
                _user_single(value),
            ],
            None,
        )

    if variant == P_WITH_CONFIDENCE:
        sys = BASELINE_SYSTEM.format(scope=scope_str) + CONFIDENCE_SUFFIX
        return (
            [{"role": "system", "content": sys}, _user_single(value)],
            {"type": "json_schema", "json_schema": SINGLE_SCHEMA_WITH_CONFIDENCE},
        )

    raise ValueError(f"Unknown prompt variant: {variant!r}")


def build_batch(variant: str, values: Sequence[str], scope: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Return (messages, response_format) for a batch evaluation of `values`."""
    scope_str = _scope_str(scope)

    if variant == P0_BASELINE:
        return (
            [
                {"role": "system", "content": BASELINE_SYSTEM.format(scope=scope_str)},
                _user_batch(values),
            ],
            None,
        )

    if variant == P1_FEW_SHOT:
        return (
            [
                {"role": "system", "content": BASELINE_SYSTEM.format(scope=scope_str)},
                *_few_shot_messages(),
                _user_batch(values),
            ],
            None,
        )

    if variant == P2_COT:
        sys = BASELINE_SYSTEM.format(scope=scope_str) + COT_SUFFIX
        return [{"role": "system", "content": sys}, _user_batch(values)], None

    if variant == P3_STRICT_SCHEMA:
        return (
            [
                {"role": "system", "content": BASELINE_SYSTEM.format(scope=scope_str)},
                _user_batch(values),
            ],
            {"type": "json_schema", "json_schema": BATCH_SCHEMA},
        )

    if variant == P4_MINIMAL:
        return (
            [
                {"role": "system", "content": MINIMAL_SYSTEM.format(scope=scope_str)},
                _user_batch(values),
            ],
            None,
        )

    if variant == P_WITH_CONFIDENCE:
        sys = BASELINE_SYSTEM.format(scope=scope_str) + CONFIDENCE_SUFFIX
        return [{"role": "system", "content": sys}, _user_batch(values)], None

    raise ValueError(f"Unknown prompt variant: {variant!r}")
