# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Robust extraction and grading for mathematical answers.

The public entry points intentionally match the original DR-GRPO grader.  In
particular, ``r1_zero_reward_fn`` and ``question_only_reward_fn`` keep the
response dictionaries consumed by the training pipeline.

Comparison proceeds from cheap and conservative checks to a bounded
``math_verify`` fallback.  Generated answers are untrusted input, so symbolic
parsing is protected by size/complexity limits and, on the main thread, the
timeouts supplied by ``math_verify``.
"""

from __future__ import annotations

import math
import re
import signal
import threading
from contextlib import AbstractContextManager
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from functools import lru_cache
from numbers import Integral, Real
from typing import Any, Optional

import sympy
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from pylatexenc import latex2text
from sympy import Basic, MatrixBase, N


# Compatibility constants retained for callers that imported them from the
# legacy grader. Units are deliberately not removed globally: doing so turns
# variables such as ``m`` into units and creates false-positive rewards.
unit_texts = [
    "degree",
    "mph",
    "kmph",
    "foot",
    "feet",
    "inch",
    "mile",
    "yard",
    "meter",
    "metre",
    "centimeter",
    "millimeter",
    "kilometer",
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",
    "gram",
    "kilogram",
    "liter",
    "litre",
    "percent",
    "dollar",
]

SUBSTITUTIONS = [
    (r"\tfrac", r"\frac"),
    (r"\dfrac", r"\frac"),
    (r"\left", ""),
    (r"\right", ""),
]
REMOVED_EXPRESSIONS = [r"\,", r"\!", r"\;", r"\quad", r"\qquad"]

BAD_SUBSTRINGS = ("__", "lambda", "import", "exec", "eval", ";")
BAD_REGEXES = (r"\*{3,}", r"\^{3,}")
TUPLE_CHARS = "()[]{}"

MAX_ANSWER_CHARS = 4096
MAX_NESTING_DEPTH = 128
MAX_DIGIT_RUN = 1000
MAX_OPERATOR_COUNT = 1024
FAST_TIMEOUT_SECONDS = 1
THOROUGH_TIMEOUT_SECONDS = 3
NUMERIC_ABS_TOL = 5e-7
NUMERIC_REL_TOL = 0.0

_MATH_EXTRACTION_CONFIG = (
    LatexExtractionConfig(boxed_match_priority=0),
    ExprExtractionConfig(),
)
_BOX_COMMAND_RE = re.compile(r"\\(?:boxed|fbox)\s*\{")
_R1_TAIL_RE = re.compile(
    r"\A\s*<answer>(?P<answer>.*?)</answer>\s*\Z",
    flags=re.DOTALL,
)
_PLAIN_NUMBER_RE = re.compile(
    r"^[+-]?(?:"
    r"(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?"
    r"|\.\d+"
    r")(?:[eE][+-]?\d+)?$"
)
_PLAIN_FRACTION_RE = re.compile(
    r"^(?P<num>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+))\s*/\s*"
    r"(?P<den>[+-]?(?:\d{1,3}(?:,\d{3})+|\d+))$"
)
_LATEX_FRACTION_RE = re.compile(
    r"^\\(?:d|t)?frac\s*\{(?P<num>[+-]?[\d,]+)\}\s*"
    r"\{(?P<den>[+-]?[\d,]+)\}$"
)
_MIXED_NUMBER_RE = re.compile(
    r"^(?P<whole>[+-]?\d+)\s+(?P<num>\d+)\s*/\s*(?P<den>[1-9]\d*)$"
)
_OUTER_TEXT_RE = re.compile(
    r"^\\(?:text|mathrm|textrm)\s*\{(?P<text>.*)\}$",
    re.DOTALL,
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_OPERATOR_RE = re.compile(r"[+\-*/^=<>]|\\(?:frac|sqrt|sum|prod|int|lim)\b")
_NUMERIC_EXPONENT_RE = re.compile(r"(?:\^|\*\*)\s*\{?\s*([+-]?\d+)")
_POWER_TOWER_RE = re.compile(
    r"(?:\^|\*\*)\s*[({]?\s*\d+\s*(?:\^|\*\*)"
)
_FACTORIAL_RE = re.compile(
    r"(?:factorial\s*\(\s*(?P<function>\d+)\s*\)|(?P<postfix>\d+)\s*!)"
)

_UNICODE_REPLACEMENTS = str.maketrans(
    {
        "−": "-",
        "–": "-",
        "—": "-",
        "×": r"\times ",
        "÷": r"\div ",
        "∞": r"\infty ",
        "π": r"\pi ",
    }
)


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def _matching_brace(text: str, open_index: int) -> int | None:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if _is_escaped(text, index):
            continue
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
            if depth < 0:
                return None
    return None


def last_boxed_only_string(string: str | None) -> str | None:
    """Return the final well-formed ``\boxed``/``\fbox`` expression."""

    if not isinstance(string, str):
        return None
    matches = list(_BOX_COMMAND_RE.finditer(string))
    if not matches:
        return None
    match = matches[-1]
    open_index = string.find("{", match.start(), match.end())
    close_index = _matching_brace(string, open_index)
    if close_index is None:
        return None
    return string[match.start() : close_index + 1]


def remove_boxed(s: str | None) -> str | None:
    """Remove one complete outer ``\boxed`` or ``\fbox`` wrapper."""

    if not isinstance(s, str):
        return None
    stripped = s.strip()
    match = _BOX_COMMAND_RE.match(stripped)
    if match is None:
        return None
    open_index = stripped.find("{", match.start(), match.end())
    close_index = _matching_brace(stripped, open_index)
    if close_index is None or stripped[close_index + 1 :].strip():
        return None
    return stripped[open_index + 1 : close_index]


def extract_boxed_answer(solution: str | None) -> str | None:
    """Extract the content of the final complete LaTeX answer box."""

    return remove_boxed(last_boxed_only_string(solution))


def extract_answer(passage: str | None) -> str | None:
    """Extract a boxed answer from a model response."""

    return extract_boxed_answer(passage)


def _strip_outer_math_delimiters(text: str) -> str:
    text = text.strip()
    changed = True
    while changed:
        changed = False
        pairs = (("$$", "$$"), (r"\[", r"\]"), (r"\(", r"\)"), ("$", "$"))
        for left, right in pairs:
            if (
                len(text) > len(left) + len(right)
                and text.startswith(left)
                and text.endswith(right)
            ):
                text = text[len(left) : -len(right)].strip()
                changed = True
                break
    return text


def _unwrap_outer_text(text: str) -> str:
    match = _OUTER_TEXT_RE.fullmatch(text.strip())
    return match.group("text").strip() if match else text


def _canonical_answer(answer: str) -> str:
    """Conservatively normalize notation without changing its meaning."""

    text = answer.strip().translate(_UNICODE_REPLACEMENTS)
    text = _strip_outer_math_delimiters(text)

    whole_box = remove_boxed(text)
    if whole_box is not None:
        text = _strip_outer_math_delimiters(whole_box.strip())

    text = _unwrap_outer_text(text)
    for before, after in SUBSTITUTIONS:
        text = text.replace(before, after)
    for expression in REMOVED_EXPRESSIONS:
        text = text.replace(expression, "")

    text = text.replace(r"\%", "%").replace(r"\$", "$")
    if re.search(r"\d|[\\{}()[\]+*/^=<>%$]", text):
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _plain_text_key(answer: str) -> str | None:
    text = _strip_outer_math_delimiters(answer.strip())
    text = _unwrap_outer_text(text)
    if not text or any(char.isdigit() for char in text):
        return None
    if re.search(r"[\\{}()[\]+*/^=<>]", text):
        return None
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n.,;:!")
    return text.casefold() or None


def _coerce_answer(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(value)
    if isinstance(value, Real):
        numeric = float(value)
        return str(value) if math.isfinite(numeric) else None
    if isinstance(value, (Decimal, Fraction, sympy.Number)):
        return str(value)
    return None


def _prepare_answer(value: Any) -> str | None:
    text = _coerce_answer(value)
    if text is None or _CONTROL_CHAR_RE.search(text):
        return None
    text = text.strip()
    if not text:
        return None
    if _BOX_COMMAND_RE.search(text):
        text = extract_boxed_answer(text)
        if text is None:
            return None
        text = text.strip()
    return text or None


def _clean_grouped_integer(value: str) -> str | None:
    if "," not in value:
        return value
    unsigned = value.lstrip("+-")
    if not re.fullmatch(r"\d{1,3}(?:,\d{3})+", unsigned):
        return None
    sign = value[:1] if value[:1] in "+-" else ""
    return sign + unsigned.replace(",", "")


def _fraction_from_integer_parts(numerator: str, denominator: str) -> Fraction | None:
    numerator = _clean_grouped_integer(numerator)
    denominator = _clean_grouped_integer(denominator)
    if numerator is None or denominator is None:
        return None
    try:
        denominator_int = int(denominator)
        if denominator_int == 0:
            return None
        return Fraction(int(numerator), denominator_int)
    except (TypeError, ValueError, OverflowError):
        return None


def _parse_simple_number(answer: str) -> tuple[Fraction, bool] | None:
    """Parse a plain number without evaluating arbitrary expressions."""

    if len(answer) > MAX_ANSWER_CHARS or re.search(
        rf"\d{{{MAX_DIGIT_RUN + 1},}}",
        answer,
    ):
        return None

    # Preserve the one meaningful whitespace form (``7 3/4``) until mixed
    # numbers have been recognized. Other numeric forms use the whitespace-
    # free canonical representation below.
    raw_text = _strip_outer_math_delimiters(answer.strip().translate(_UNICODE_REPLACEMENTS))
    whole_box = remove_boxed(raw_text)
    if whole_box is not None:
        raw_text = whole_box.strip()
    raw_text = raw_text.replace(r"\%", "%").replace(r"\$", "$")

    # Currency symbols are answer units, not mathematical operators.
    for prefix in ("$", "€", "£", "¥"):
        if raw_text.startswith(prefix):
            raw_text = raw_text[len(prefix) :]
            break

    is_percent = raw_text.rstrip().endswith("%")
    if is_percent:
        raw_text = raw_text.rstrip()[:-1]

    mixed_match = _MIXED_NUMBER_RE.fullmatch(raw_text.strip())
    if mixed_match:
        denominator = int(mixed_match.group("den"))
        whole = int(mixed_match.group("whole"))
        fraction = Fraction(int(mixed_match.group("num")), denominator)
        value = whole - fraction if whole < 0 else whole + fraction
        return (value / 100 if is_percent else value, is_percent)

    text = _canonical_answer(raw_text)

    latex_match = _LATEX_FRACTION_RE.fullmatch(text)
    if latex_match:
        value = _fraction_from_integer_parts(
            latex_match.group("num"),
            latex_match.group("den"),
        )
        if value is None:
            return None
        return (value / 100 if is_percent else value, is_percent)

    fraction_match = _PLAIN_FRACTION_RE.fullmatch(text)
    if fraction_match:
        value = _fraction_from_integer_parts(
            fraction_match.group("num"),
            fraction_match.group("den"),
        )
        if value is None:
            return None
        return (value / 100 if is_percent else value, is_percent)

    if not _PLAIN_NUMBER_RE.fullmatch(text):
        return None

    if "," in text:
        mantissa = re.split(r"[eE][+-]?\d+$", text, maxsplit=1)[0]
        integer_part = mantissa.split(".", maxsplit=1)[0]
        if _clean_grouped_integer(integer_part) is None:
            return None
        text = text.replace(",", "")

    try:
        value = Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError, OverflowError):
        return None
    return (value / 100 if is_percent else value, is_percent)


def _fractions_close(left: Fraction, right: Fraction) -> bool:
    if left == right:
        return True
    try:
        return math.isclose(
            float(left),
            float(right),
            rel_tol=NUMERIC_REL_TOL,
            abs_tol=NUMERIC_ABS_TOL,
        )
    except (OverflowError, ValueError):
        return False


def numeric_equal(prediction: float, reference: float) -> bool:
    try:
        return math.isclose(
            float(reference),
            float(prediction),
            rel_tol=NUMERIC_REL_TOL,
            abs_tol=NUMERIC_ABS_TOL,
        )
    except (TypeError, ValueError, OverflowError):
        return False


def _nesting_is_bounded(text: str) -> bool:
    depth = 0
    for index, char in enumerate(text):
        if _is_escaped(text, index):
            continue
        if char in "([{":
            depth += 1
            if depth > MAX_NESTING_DEPTH:
                return False
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return False
    # Mixed interval delimiters such as (0, 1] are valid mathematics, so the
    # budget tracks depth rather than requiring bracket types to match.
    return depth == 0


def _within_symbolic_budget(text: str) -> bool:
    if not text or len(text) > MAX_ANSWER_CHARS:
        return False
    lowered = text.casefold()
    if any(bad in lowered for bad in BAD_SUBSTRINGS):
        return False
    if any(re.search(pattern, text) for pattern in BAD_REGEXES):
        return False
    if re.search(rf"\d{{{MAX_DIGIT_RUN + 1},}}", text):
        return False
    if _POWER_TOWER_RE.search(text):
        return False
    for match in _NUMERIC_EXPONENT_RE.finditer(text):
        if abs(int(match.group(1))) > 10_000:
            return False
    for match in _FACTORIAL_RE.finditer(text):
        value = match.group("function") or match.group("postfix")
        if int(value) > 1_000:
            return False
    if len(_OPERATOR_RE.findall(text)) > MAX_OPERATOR_COUNT:
        return False
    return _nesting_is_bounded(text)


def _math_source(text: str) -> str:
    stripped = text.strip()
    if "$" in stripped or stripped.startswith((r"\(", r"\[")):
        return stripped
    return f"${stripped}$"


def _set_builder_condition(text: str) -> str:
    r"""Convert a simple set-builder answer to its defining relation.

    ``math_verify`` can compare relations with intervals, but some versions do
    not parse forms such as ``\{x\mid x>0\}`` directly.
    """

    stripped = _strip_outer_math_delimiters(text.strip())
    stripped = stripped.replace(r"\left", "").replace(r"\right", "")
    if stripped.startswith(r"\{") and stripped.endswith(r"\}"):
        inner = stripped[2:-2]
    elif stripped.startswith("{") and stripped.endswith("}"):
        inner = stripped[1:-1]
    else:
        return text

    for separator in (r"\mid", "|"):
        if separator in inner:
            _, condition = inner.split(separator, maxsplit=1)
            return condition.strip() or text
    return text


def _timeout_for_current_thread(fast: bool) -> int | None:
    if threading.current_thread() is not threading.main_thread():
        return None
    return FAST_TIMEOUT_SECONDS if fast else THOROUGH_TIMEOUT_SECONDS


def _parse_math(text: str, parsing_timeout: int | None) -> list[Any]:
    text = _set_builder_condition(text)
    return parse(
        _math_source(text),
        extraction_config=_MATH_EXTRACTION_CONFIG,
        fallback_mode="first_match",
        extraction_mode="first_match",
        parsing_timeout=parsing_timeout,
        raise_on_error=False,
    )


@lru_cache(maxsize=4096)
def _parse_reference(text: str, parsing_timeout: int | None) -> tuple[Any, ...]:
    return tuple(_parse_math(text, parsing_timeout))


def _math_verify_equal(given_answer: str, ground_truth: str, fast: bool) -> bool:
    if not _within_symbolic_budget(given_answer) or not _within_symbolic_budget(ground_truth):
        return False

    timeout_seconds = _timeout_for_current_thread(fast)
    try:
        gold = list(_parse_reference(ground_truth, timeout_seconds))
        target = _parse_math(given_answer, timeout_seconds)
        if not gold or not target:
            return False
        if verify(
            gold,
            target,
            float_rounding=6,
            numeric_precision=15,
            strict=True,
            allow_set_relation_comp=True,
            timeout_seconds=timeout_seconds,
            raise_on_error=False,
        ):
            return True
        return _equations_equivalent(gold, target, timeout_seconds)
    except (ArithmeticError, TypeError, ValueError, RuntimeError, RecursionError):
        return False


def _equations_equivalent(
    gold: list[Any],
    target: list[Any],
    timeout_seconds: int | None,
) -> bool:
    """Compare equations up to multiplication by a nonzero constant."""

    try:
        with timeout(timeout_seconds or 0):
            for gold_expr in gold:
                if not isinstance(gold_expr, sympy.Equality):
                    continue
                gold_zero = gold_expr.lhs - gold_expr.rhs
                for target_expr in target:
                    if not isinstance(target_expr, sympy.Equality):
                        continue
                    target_zero = target_expr.lhs - target_expr.rhs
                    ratio = sympy.simplify(gold_zero / target_zero)
                    if (
                        ratio.is_zero is False
                        and not ratio.free_symbols
                        and ratio not in (sympy.nan, sympy.zoo, sympy.oo, -sympy.oo)
                    ):
                        return True
    except (ArithmeticError, TypeError, ValueError, RuntimeError, RecursionError, TimeoutError):
        return False
    return False


def _answers_equal(given_answer: str, ground_truth: str, fast: bool = True) -> bool:
    if given_answer == ground_truth:
        return bool(given_answer)

    given_canonical = _canonical_answer(given_answer)
    truth_canonical = _canonical_answer(ground_truth)
    if given_canonical and given_canonical == truth_canonical:
        return True

    given_text = _plain_text_key(given_answer)
    truth_text = _plain_text_key(ground_truth)
    if given_text is not None and truth_text is not None and given_text == truth_text:
        return True
    if given_text is not None and truth_text is not None:
        # Do not let a symbolic parser erase meaningful word boundaries in
        # ordinary text answers (for example, "New York" vs "Newy ork").
        given_compact = given_text.replace(" ", "")
        truth_compact = truth_text.replace(" ", "")
        if (
            given_compact == truth_compact
            and given_text != truth_text
            and len(given_compact) > 2
        ):
            return False

    given_numeric = _parse_simple_number(given_answer)
    truth_numeric = _parse_simple_number(ground_truth)
    if given_numeric is not None and truth_numeric is not None:
        given_value, given_percent = given_numeric
        truth_value, truth_percent = truth_numeric
        if _fractions_close(given_value, truth_value):
            return True
        # A percent sign is both a factor and an answer unit in common math
        # datasets. Let math_verify resolve that ambiguity.
        if not given_percent and not truth_percent:
            return False

    return _math_verify_equal(given_answer, ground_truth, fast)


def mathd_normalize_answer(answer: Optional[str]) -> Optional[str]:
    """Compatibility wrapper for the legacy MathD normalizer."""

    if answer is None:
        return None
    if not isinstance(answer, str):
        answer = str(answer)
    return _canonical_answer(answer)


def _strip_string(string: str) -> str:
    return _canonical_answer(string)


def normalize_final_answer(final_answer: str) -> str:
    return _canonical_answer(final_answer)


def _normalize(expr: str | None) -> str | None:
    if expr is None:
        return None
    return _canonical_answer(expr)


def grade_answer_mathd(given_answer: str, ground_truth: str) -> bool:
    given = _prepare_answer(given_answer)
    truth = _prepare_answer(ground_truth)
    if given is None or truth is None:
        return False
    given_normalized = _canonical_answer(given)
    return bool(given_normalized and given_normalized == _canonical_answer(truth))


def grade_answer_sympy(given_answer: str, ground_truth: str) -> bool:
    given = _prepare_answer(given_answer)
    truth = _prepare_answer(ground_truth)
    if given is None or truth is None:
        return False
    return _answers_equal(given, truth, fast=True)


def is_latex_equal(given_answer: str, ground_truth: str) -> bool:
    given = _prepare_answer(given_answer)
    truth = _prepare_answer(ground_truth)
    if given is None or truth is None:
        return False
    return _answers_equal(given, truth, fast=False)


def _is_latex_equal(str1: str, str2: str) -> bool:
    return is_latex_equal(str1, str2)


def is_value_equal(given_answer: str, ground_truth: str) -> bool:
    given = _prepare_answer(given_answer)
    truth = _prepare_answer(ground_truth)
    if given is None or truth is None:
        return False
    given_numeric = _parse_simple_number(given)
    truth_numeric = _parse_simple_number(truth)
    if given_numeric is not None and truth_numeric is not None:
        return _fractions_close(given_numeric[0], truth_numeric[0])
    return _canonical_answer(given) == _canonical_answer(truth)


def symbolic_equal(a: str, b: str) -> bool:
    given = _prepare_answer(a)
    truth = _prepare_answer(b)
    if given is None or truth is None:
        return False
    return _answers_equal(given, truth, fast=False)


def _sympy_parse(expr: str) -> Basic | MatrixBase:
    if not _within_symbolic_budget(expr):
        raise ValueError("Expression exceeds the symbolic parsing budget")
    parsed = _parse_math(expr, _timeout_for_current_thread(fast=False))
    for value in parsed:
        if isinstance(value, (Basic, MatrixBase)):
            return value
    raise ValueError(f"Unable to parse expression: {expr!r}")


def latex_eval(latex: str) -> tuple[Basic | MatrixBase, Any]:
    expression = _sympy_parse(latex)
    return expression, N(expression)


def _parse_latex(expr: str) -> str:
    """Convert common LaTeX tokens to a plain-text expression."""

    text = expr.replace(r"\tfrac", r"\frac").replace(r"\dfrac", r"\frac")
    text = latex2text.LatexNodes2Text().latex_to_text(text)
    return (
        text.replace("√", "sqrt")
        .replace("π", "pi")
        .replace("∞", "inf")
        .replace("∪", "U")
        .replace("·", "*")
        .replace("×", "*")
        .strip()
    )


def _is_float(num: str) -> bool:
    try:
        float(num)
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _is_int(x: float) -> bool:
    try:
        return math.isclose(float(x), round(float(x)), rel_tol=0.0, abs_tol=1e-7)
    except (TypeError, ValueError, OverflowError):
        return False


def _is_frac(expr: str) -> bool:
    return _PLAIN_FRACTION_RE.fullmatch(expr.strip()) is not None


def _strip_properly_formatted_commas(expr: str) -> str:
    pattern = re.compile(r"(?<!\d)([+-]?\d{1,3}(?:,\d{3})+)(?!\d)")
    return pattern.sub(lambda match: match.group(1).replace(",", ""), expr)


def _str_is_int(x: str) -> bool:
    parsed = _parse_simple_number(x)
    return parsed is not None and parsed[0].denominator == 1 and not parsed[1]


def _str_to_int(x: str) -> int:
    parsed = _parse_simple_number(x)
    if parsed is None or parsed[0].denominator != 1:
        raise ValueError(f"Not an integer: {x!r}")
    return int(parsed[0])


def _inject_implicit_mixed_number(step: str) -> str:
    return re.sub(r"(?<=\d)\s+(?=\d+\s*/\s*\d+)", "+", step)


def count_unknown_letters_in_expr(expr: str) -> int:
    without_commands = re.sub(r"\\[A-Za-z]+", "", expr)
    return len({char.casefold() for char in without_commands if char.isalpha()})


def should_allow_eval(expr: str) -> bool:
    return _within_symbolic_budget(expr)


def are_equal_under_sympy(ground_truth_normalized: str, given_normalized: str) -> bool:
    return _math_verify_equal(given_normalized, ground_truth_normalized, fast=True)


def split_tuple(expr: str) -> list[str]:
    """Split top-level tuple/set/interval elements without splitting 1,000."""

    text = expr.strip()
    if not text:
        return []
    if len(text) < 2 or text[0] not in "([{" or text[-1] not in ")]}":
        return [expr]

    inner = text[1:-1]
    parts: list[str] = []
    start = 0
    stack: list[str] = []
    closing = {")": "(", "]": "[", "}": "{"}
    for index, char in enumerate(inner):
        if _is_escaped(inner, index):
            continue
        if char in "([{":
            stack.append(char)
        elif char in closing:
            if stack and stack[-1] == closing[char]:
                stack.pop()
        elif char == "," and not stack:
            suffix = inner[index + 1 :]
            prefix = inner[:index]
            if re.search(r"\d$", prefix) and re.match(r"\d{3}(?:\D|$)", suffix):
                continue
            parts.append(inner[start:index].strip())
            start = index + 1
    if not parts:
        return [expr]
    parts.append(inner[start:].strip())
    return parts


def repeatness(s: str) -> bool:
    """Detect conspicuous repeated output in bounded linear time."""

    if len(s) < 64:
        return False
    for width in (1, 2, 4, 8, 16, 32):
        if width * 4 > len(s):
            break
        for start in range(min(width, len(s) - width * 4 + 1)):
            block = s[start : start + width]
            if block and block * 4 in s:
                return True
    return False


class timeout(AbstractContextManager):
    """Backward-compatible signal timeout that restores the prior handler."""

    def __init__(self, seconds: int = 1, error_message: str = "Timeout"):
        self.seconds = seconds
        self.error_message = error_message
        self._enabled = False
        self._old_handler: Any = None
        self._old_alarm = 0

    def handle_timeout(self, signum: int, frame: Any) -> None:
        raise TimeoutError(self.error_message)

    def __enter__(self) -> "timeout":
        if threading.current_thread() is not threading.main_thread() or self.seconds <= 0:
            return self
        self._old_handler = signal.getsignal(signal.SIGALRM)
        self._old_alarm = signal.alarm(0)
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        self._enabled = True
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._enabled:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)
            if self._old_alarm:
                signal.alarm(self._old_alarm)
        return False


def grade(model_answer: Any, gt_answer: Any, fast: bool = True) -> bool:
    """Return whether a model answer is mathematically equivalent to a reference."""

    given = _prepare_answer(model_answer)
    truth = _prepare_answer(gt_answer)
    if given is None or truth is None:
        return False
    return _answers_equal(given, truth, fast=bool(fast))


def _references(ground_truth: Any) -> list[Any]:
    return ground_truth if isinstance(ground_truth, list) else [ground_truth]


def _is_correct_against_any(model_answer: str, ground_truth: Any, fast: bool) -> bool:
    return any(
        grade(model_answer, reference, fast=fast)
        for reference in _references(ground_truth)
    )


def _reward(format_ok: bool, answer_ok: bool) -> dict[str, float]:
    return {
        "format_reward": float(format_ok),
        "answer_reward": float(answer_ok),
        "reward": float(format_ok and answer_ok),
    }


def _extract_r1_answer(response: Any) -> tuple[bool, str | None]:
    """Parse the completion produced after a prompt ending in ``<think>``."""

    if not isinstance(response, str) or response.count("</think>") != 1:
        return False, None
    _, tail = response.split("</think>", maxsplit=1)
    # The existing training prompt and smoke contract require exactly one
    # literal space between </think> and <answer>.
    if not tail.startswith(" <answer>"):
        return False, None
    if tail.count("<answer>") != 1 or tail.count("</answer>") != 1:
        return False, None
    match = _R1_TAIL_RE.fullmatch(tail)
    if match is None:
        return False, None
    answer = match.group("answer").strip()
    if not answer or "<answer>" in answer or "</answer>" in answer:
        return False, None
    return True, answer


def r1_zero_reward_fn(
    response: Any,
    ground_truth: Any,
    fast: bool = True,
) -> dict[str, float]:
    """Grade an R1-zero completion without changing the reward schema."""

    format_ok, model_answer = _extract_r1_answer(response)
    if not format_ok or model_answer is None:
        return _reward(False, False)

    if _BOX_COMMAND_RE.search(model_answer):
        model_answer = extract_boxed_answer(model_answer)
        if model_answer is None:
            return _reward(True, False)

    try:
        is_correct = _is_correct_against_any(model_answer, ground_truth, fast=bool(fast))
    except Exception:
        # A malformed generated answer must never terminate the training loop.
        is_correct = False
    return _reward(True, is_correct)


def question_only_reward_fn(
    response: Any,
    ground_truth: Any,
    fast: bool = True,
) -> dict[str, float]:
    """Grade a response whose required output format is a boxed answer."""

    model_answer = extract_answer(response)
    if model_answer is None:
        return _reward(False, False)
    try:
        is_correct = _is_correct_against_any(model_answer, ground_truth, fast=bool(fast))
    except Exception:
        is_correct = False
    return _reward(True, is_correct)
