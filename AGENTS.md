# Agent Instructions (Copilot / AI assistants)

## Overview

This is a template repository providing guidelines, best practices, and conventions for Python projects working with AI assistants. It defines agent instructions, coding standards, project structure, and dependency management practices.

## 0) Where to find coding rules

- Coding rules (style, typing, testing, tooling) are defined in:
  - `.github/copilot-instructions.md`
- Consult it before making stylistic decisions.

## 1) Priority / scope

- Follow this file first.
- Then follow `.github/copilot-instructions.md`.
- Then follow other repository docs (`README.md`, `CONTRIBUTING.md`, `docs/*`).
- If instructions conflict, ask for clarification rather than guessing.

## 2) Repository structure (must follow)

Expected at project root:

- `pyproject.toml` (required; single source of truth)
- `src/<package_name>/` (package code; keep `src/` layout consistent)
- `tests/`
- `docs/`

## 3) Dependency management (uv required)

- Dependency management MUST use **uv**.
- Prefer `uv run ...` to execute tools in the managed environment.

## 4) What to include in proposals/patches

- Exact file paths to create/edit.
- Final code (not only pseudocode).
- Brief rationale.
- How to run formatting/lint/type-check/tests (uv-based commands).

## 5) Key resources

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) - Referenced for style conventions
- [Ruff](https://docs.astral.sh/ruff/) - Fast Python linter and formatter
- [Pyright](https://github.com/microsoft/pyright) - Static type checker
- [UV](https://github.com/astral-sh/uv) - Python package manager
