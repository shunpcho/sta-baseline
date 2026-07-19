---
applyTo: "**/*.py"
description: "Python coding standards for maintainable, typed, testable, and readable code."
---

# Python Coding Guidelines

These guidelines are for AI coding agents working on Python files. They complement, rather than duplicate, checks enforced by Ruff, Pyright, Ty, pytest, and project configuration files.

## Priority Order

When rules appear to conflict, follow this order:

1. Correctness and safety
2. Existing project conventions
3. Public API compatibility
4. Type safety
5. Readability and maintainability
6. Performance
7. Brevity

Do not introduce a large abstraction merely to satisfy a stylistic preference.

## Version and Tooling Assumptions

- Follow the Python version declared in `pyproject.toml` (`project.requires-python` and tool target versions).
- Prefer modern Python syntax supported by the declared target version.
- Keep code compatible with Ruff formatting and linting.
- Keep code type-checkable by Pyright and Ty.
- Do not bypass tools with broad ignores. Use narrow, justified suppressions only when necessary.

Recommended local checks:

```bash
ruff format .
ruff check --fix .
pyright
ty check
uv run pytest
```

## Package and Import Structure

### Package Layout

- Use a standard `src/` layout for importable packages when the project is packaged.
- Include `__init__.py` in normal package directories.
- Keep `__init__.py` lightweight: expose stable public APIs only; avoid heavy imports and side effects.
- Avoid namespace packages unless the project explicitly needs them.

```text
src/
└── mypackage/
    ├── __init__.py
    ├── core.py
    └── adapters/
        ├── __init__.py
        └── filesystem.py
```

### Imports

- Use absolute imports for package modules.
- Keep imports at module top level unless lazy loading is needed to avoid heavy optional dependencies or circular imports.
- Do not manipulate `sys.path` in application code, tests, or notebooks.
- Do not shadow standard-library modules, builtins, imported modules, or important domain names.

```python
# Good
from pathlib import Path

from mypackage.core import Processor


# Bad
import sys

sys.path.insert(0, "src")
```

## Typing Guidelines

### General Typing

- Annotate all public functions, methods, class attributes, and module-level constants.
- Prefer precise types over `Any`.
- Use `typing.Any` only at validated boundaries or when integrating with truly dynamic APIs.
- Prefer `object` over `Any` when the value is intentionally opaque.
- Use `TypeAlias` for complex reusable type expressions.
- Use `Protocol` for structural interfaces and dependency inversion.
- Use `Literal` for finite string modes and states.
- Use `TypedDict` or Pydantic models for structured dictionaries crossing boundaries.
- Always use `pathlib.Path` for paths.
- For NumPy arrays, prefer `npt.NDArray[...]` to document dtype expectations.
  - Example: distinguish raw images (`np.uint8`) vs normalized tensors/arrays (`np.float32`).

#### Suppressing type errors (preferred approach)

Avoid suppressions when possible. If you must suppress:

- Prefer **narrow, documented suppressions** over blanket ignores.
- Prefer Pyright-specific ignores with a diagnostic code:
  - ✅ `# pyright: ignore[reportUnknownVariableType]  # <reason>`
  - ✅ `# pyright: ignore[reportGeneralTypeIssues]  # <reason>`
- Avoid blanket ignores:
  - ❌ `# type: ignore`
  - ❌ `# pyright: ignore`

```python
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal, Protocol, TypeAlias

PathLike: TypeAlias = str | bytes
Mode = Literal["train", "eval", "predict"]


class SupportsPredict(Protocol):
    def predict(self, x: Sequence[float]) -> float: ...
```

### Modern Syntax

- Use PEP 604 unions: `str | None`, not `Optional[str]`.
- Use built-in generics: `list[str]`, `dict[str, int]`, not `List[str]`.
- Import abstract collection types from `collections.abc` when values are consumed generically.
- Prefer `Self` for fluent APIs when supported by the project Python version.

```python
from collections.abc import Iterable
from typing import Self


class Builder:
    def add(self, values: Iterable[str]) -> Self:
        return self
```

## Function Design

- Keep functions small, cohesive, and testable.
- Prefer dependency injection over hidden global access.
- Avoid mutable default arguments.
- Avoid boolean flags that create multiple modes; use separate functions or `Literal` modes when clearer.
- Return explicit result types. Avoid returning unrelated shapes from the same function.
- Raise specific exceptions with actionable messages.

```python
def add_item(item: str, items: list[str] | None = None) -> list[str]:
    values = [] if items is None else list(items)
    values.append(item)
    return values
```

## Class and Data Model Design

### Dataclass vs Pydantic

- Use Pydantic models for external input/output boundaries: config files, API payloads, CLI input, serialized data.
- Use `@dataclass(slots=True)` for internal immutable or lightweight domain data.
- Avoid using Pydantic as a general-purpose internal data container unless validation/serialization is needed.
- Prefer immutable data (`frozen=True`) when mutation is not required.

```python
from dataclasses import dataclass

from pydantic import BaseModel, Field


class TrainConfigInput(BaseModel):
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    batch_size: int
    learning_rate: float
```

### Interfaces

- Prefer `Protocol` over inheritance when only behavior matters.
- Use abstract base classes only when shared implementation or nominal hierarchy is important.
- Keep constructors lightweight; avoid I/O or GPU allocation in `__init__` unless explicitly documented.

## Error Handling and Logging

### Exception Policy

- Catch specific exceptions before broad exceptions.
- Use exception chaining (`raise NewError(...) from e`) when adding context.
- Do not use `assert` for runtime validation in production code.
- Let inner functions raise; handle recovery or user-facing reporting at boundaries.

```python
from pathlib import Path
import json


def load_json(path: Path) -> dict[str, object]:
    try:
        with path.open(encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON file: {path}"
        raise ValueError(msg) from e

    if not isinstance(data, dict):
        msg = f"Expected JSON object: {path}"
        raise TypeError(msg)
    return data
```

### Logging Policy

- Use a module-specific logger: `logging.getLogger(__name__)`.
- Log an exception once, at the outermost boundary that can add useful operational context.
- Inner layers should raise or chain exceptions, not call `logger.exception()`.
- Do not log secrets, credentials, tokens, raw personal data, or sensitive file contents.

```python
import logging

logger = logging.getLogger(__name__)


def run_workflow(config_path: Path) -> None:
    try:
        config = load_json(config_path)
        execute(config)
    except Exception:
        logger.exception("Workflow failed: config_path=%s", config_path)
        raise
```

## Filesystem, Paths, and Serialization

- Use `pathlib.Path` for filesystem paths.
- Accept `Path | str` at public boundaries only when useful; normalize to `Path` immediately.
- Use explicit encodings for text I/O.
- Use atomic writes for important output files when partial writes are harmful.
- Do not use `pickle` for untrusted data.
- Use safe YAML loading (`yaml.safe_load`) for YAML.

```python
from pathlib import Path


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")
```

## Security and Subprocesses

Never use these for untrusted input:

```python
eval(user_input)
exec(user_code)
yaml.load(data)
pickle.load(file)
subprocess.run(command, shell=True)
```

Prefer safe alternatives:

```python
import ast
import subprocess
import yaml

value = ast.literal_eval(text)
data = yaml.safe_load(text)
subprocess.run(["git", "status", "--short"], check=True)
```

## Async and Concurrency

- Do not call blocking I/O inside `async def`.
- Use `asyncio.TaskGroup` for structured concurrency when supported by the target Python version.
- Keep cancellation behavior in mind; avoid swallowing `CancelledError`.
- For CPU-bound work, use a process pool or a dedicated worker strategy rather than blocking the event loop.

```python
import asyncio


async def fetch_all(urls: list[str]) -> dict[str, str]:
    results: dict[str, str] = {}
    async with asyncio.TaskGroup() as group:
        tasks = {url: group.create_task(fetch_text(url)) for url in urls}
    for url, task in tasks.items():
        results[url] = task.result()
    return results
```

## Performance Guidelines

- Write clear code first; optimize after measuring.
- Avoid unnecessary copies of large arrays, tensors, images, and data frames.
- Prefer vectorized NumPy/PyTorch operations over Python loops for numeric workloads.
- Be explicit about device placement and dtype in PyTorch code.
- Avoid hidden synchronization points in GPU code when performance matters.
- Do not introduce caching unless invalidation and memory growth are understood.

## Scientific / ML Code Guidelines

- Separate pure model/loss/dataset logic from experiment orchestration.
- Keep random seeds, device selection, precision policy, and deterministic settings explicit.
- Do not silently change tensor shape, dtype, device, or value range.
- Name tensor dimensions in comments or variable names when ambiguity is likely.
- Validate external image/model/config inputs at boundaries; avoid repeated checks in inner tensor kernels.
- Preserve reproducibility metadata when changing training or evaluation code.

```python
# Good: explicit shape convention
# image: (batch, channels, height, width), float32, range [0, 1]
def normalize_image(image: torch.Tensor) -> torch.Tensor:
    return image.clamp(0, 1)
```

## Testing Guidelines

- Use pytest.
- Keep tests deterministic and independent.
- Prefer fixtures for setup shared by multiple tests.
- Use specific exception assertions with `match=`.
- Avoid broad `pytest.raises(Exception)`.
- Use simple assertions so failures are easy to diagnose.
- Add regression tests for bug fixes.
- Mark resource-heavy tests with the appropriate marker: `slow`, `native`, or `gpu`.

```python
import pytest


def test_negative_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-negative"):
        calculate(-1)
```

## Documentation and Comments

- Use docstrings for public modules, classes, functions, and non-obvious behavior.
- Prefer Google-style docstrings if the project has no stronger convention.
- Explain why, not what, in comments.
- Keep examples small and executable where practical.
- Update documentation when commands, public APIs, configuration, or behavior changes.

## AI Agent Anti-Patterns

Avoid these common mistakes:

- Large unrelated refactors while solving a small task.
- Adding compatibility layers that are not required by the target Python version.
- Introducing `Any`, `# type: ignore`, or `# noqa` to silence errors without justification.
- Creating generic utility modules with vague names like `helpers.py` for unrelated functions.
- Adding repeated defensive checks inside trusted internal functions.
- Logging the same exception at multiple layers.
- Hiding I/O, network access, GPU allocation, or global state inside seemingly pure functions.
- Adding dependencies for trivial functionality available in the standard library.

## Quick Checklist Before Returning Code

- Is the diff minimal and related to the task?
- Are public functions typed?
- Are imports clean and at top level?
- Are errors specific and chained where appropriate?
- Are filesystem paths represented with `Path`?
- Are tests added or updated for behavior changes?
- Can Ruff, Pyright/Ty, and pytest reasonably pass?
