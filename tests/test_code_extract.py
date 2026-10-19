"""Agent-authored synthetic corpus; awaiting human audit. Never execute snippets."""

import pytest

from grader_power.code_extract import extract_code
from grader_power.schemas import ExtractionStatus


@pytest.mark.parametrize("tag", ["python", "py", ""])
def test_fenced_source_preserved(tag: str) -> None:
    """Only fence delimiters/prose are removed; source bytes represented as text stay exact."""
    code = "# comment\r\ndef transform(items):\r\n    return [x for x in items]  \r\n"
    raw = f"Here is the file.\r\n```{tag}\r\n{code}```\r\nExplanation.\r\n"
    result = extract_code(raw, "transform")
    assert result.raw_response == raw
    assert result.code == code
    assert result.status == "ok"
    assert result.output_format == "fenced_python"


def test_raw_source_is_not_rewritten() -> None:
    """Keep helpers, imports, comments, trailing spaces, and even hostile statements as data."""
    code = "import nonexistent_package\nraise RuntimeError('never execute')\n\ndef helper():\n    pass\n\ndef transform(items):\n    return [x for x in items]  \n"
    result = extract_code(code, "transform")
    assert result.code == result.raw_response == code
    assert result.output_format == "raw_python"
    assert result.status == "ok"


@pytest.mark.parametrize(
    "raw,status,reason",
    [
        ("", "format_error", "Empty response is an explicit format failure"),
        ("    return []\n", "parse_error", "No function-body wrapping or dedenting"),
        ("def transform(:\n", "parse_error", "No syntax repair"),
        (
            "def wrong(items):\n    return []\n",
            "target_missing",
            "No function renaming",
        ),
        (
            "class Wrapper:\n    def transform(self):\n        return []\n",
            "target_missing",
            "Methods do not match module targets",
        ),
        (
            "def helper():\n    def transform(items):\n        return []\n",
            "target_missing",
            "Nested functions are not module targets",
        ),
        (
            "def transform(x): return []\ndef transform(x): return [x]\n",
            "target_ambiguous",
            "Never choose between duplicate implementations",
        ),
        (
            "if flag:\n    def transform(x): return []\n",
            "target_ambiguous",
            "Conditional definitions are not reliable entry points",
        ),
        (
            "def transform(x): return []\ntransform = other\n",
            "target_ambiguous",
            "Static rebinding changes the callable",
        ),
        (
            "def transform(x): return []\nfrom package import transform\n",
            "target_ambiguous",
            "Imports can shadow the target",
        ),
        (
            "def transform(x): return []\nfrom package import *\n",
            "target_ambiguous",
            "Star imports can shadow the target",
        ),
        ("```python\ndef transform(x): return []\n", "format_error", "Unclosed fence"),
        (
            "```javascript\ndef transform(x): return []\n```\n",
            "format_error",
            "No guessing language tags",
        ),
        (
            "  ```python\ndef transform(x): return []\n  ```\n",
            "format_error",
            "Indented fences are not accepted",
        ),
        (
            "```python\ndef transform(x): return []\n```\n```python\ndef transform(x): return []\n```",
            "ambiguous_output",
            "Even identical implementations are ambiguous",
        ),
        (
            "```python\ndef transform(x): return []\n```\n```text\nNotes\n```",
            "ambiguous_output",
            "No heuristic selection among multiple blocks",
        ),
        (
            "def transform(x): return []\n```python\ndef transform(x): return [x]\n```",
            "ambiguous_output",
            "Raw plus fenced implementation is ambiguous",
        ),
        (
            '{"solution": "def transform(x): return []"}',
            "target_missing",
            "JSON wrappers are not decoded",
        ),
        (
            "Here is the code:\ndef transform(x): return []\n",
            "parse_error",
            "Unfenced prose is not stripped",
        ),
    ],
)
def test_explicit_failures(raw: str, status: ExtractionStatus, reason: str) -> None:
    """Every rejection retains raw evidence; the reason column documents expected policy."""
    result = extract_code(raw, "transform")
    assert result.status == status, reason
    assert result.raw_response == raw
    assert result.reason
    if status in ("parse_error", "target_missing", "target_ambiguous"):
        assert result.code == raw


def test_fences_inside_python_string_are_data() -> None:
    """A valid raw file takes precedence over Markdown-looking docstring contents."""
    raw = (
        'def transform(x):\n    """\n```python\nexample\n```\n    """\n    return []\n'
    )
    result = extract_code(raw, "transform")
    assert result.code == raw
    assert result.status == "ok"
    assert result.output_format == "raw_python"


@pytest.mark.parametrize(
    "suffix",
    [
        "try:\n    pass\nexcept Exception as transform:\n    pass\n",
        "match value:\n    case transform:\n        pass\n",
        "del transform\n",
    ],
)
def test_indirect_ast_bindings_are_ambiguous(suffix: str) -> None:
    """Exception/match targets store names outside ast.Name fields."""
    result = extract_code("def transform(items): return []\n" + suffix, "transform")
    assert result.status == "target_ambiguous"


@pytest.mark.parametrize(
    "suffix",
    [
        "other = [transform for transform in items]\n",
        "other = {transform for transform in items}\n",
        "other = {transform: value for transform, value in items}\n",
        "other = (transform for transform in items)\n",
        "other = [[transform for transform in row] for row in items]\n",
        "def helper(transform=None):\n    transform = None\n",
        "other = lambda transform=None: (transform := None)\n",
        "class Helper:\n    transform = None\n",
    ],
)
def test_nested_local_bindings_do_not_replace_target(suffix: str) -> None:
    """Comprehension targets and nested local names do not rebind the module entry."""
    code = "def transform(items): return []\n" + suffix
    result = extract_code(code, "transform")
    assert result.status == "ok"
    assert result.code == code


@pytest.mark.parametrize(
    "suffix",
    [
        "def helper(value=(transform := None)):\n    pass\n",
        "def helper(*, value=(transform := None)):\n    pass\n",
        "async def helper(value=(transform := None)):\n    pass\n",
        "other = lambda value=(transform := None): value\n",
        "@(transform := decorator)\ndef helper():\n    pass\n",
        "def helper(value: (transform := int)):\n    pass\n",
        "def helper() -> (transform := int):\n    pass\n",
        "class Helper((transform := object)):\n    pass\n",
        "class Helper(metaclass=(transform := type)):\n    pass\n",
        "@(transform := decorator)\nclass Helper:\n    pass\n",
        "other = [(transform := value) for value in items]\n",
        "other = {value for value in items if (transform := value)}\n",
        "other = {value: (transform := value) for value in items}\n",
        "other = ((transform := value) for value in items)\n",
    ],
)
def test_outer_scope_expression_bindings_are_ambiguous(suffix: str) -> None:
    """Definition headers and comprehension walruses can overwrite the entry point."""
    code = "def transform(items): return []\n" + suffix
    result = extract_code(code, "transform")
    assert result.status == "target_ambiguous"
    assert result.code == code


def test_invalid_fenced_source_is_retained() -> None:
    """Parse failure preserves the exact selected source as well as the full raw response."""
    code = "def transform(:\n    return []\n"
    raw = f"```python\n{code}```\n"
    result = extract_code(raw, "transform")
    assert result.status == "parse_error"
    assert result.code == code
    assert result.raw_response == raw


@pytest.mark.parametrize("entry_point", ["", "class", "a.b", "two names"])
def test_invalid_entry_point(entry_point: str) -> None:
    """Bad API contracts fail before inspecting response text."""
    with pytest.raises(ValueError, match="identifier"):
        extract_code("", entry_point)
