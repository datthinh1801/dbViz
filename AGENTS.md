# AGENTS.md

## Development Guidelines
- Use `uv sync --dev` to install the environment in dev mode.
- Use `uv sync` to install the production environment.
- Use `uv run ruff check --fix` at the project root to format and lint all Python files before commit.
- Fix any linting or formatting errors until all ruff's checks are passed.
- Each class, method, and function must be focused and concise.

## Contributions
- PR/branch names follow these rules:
    - Fix bugs: fix/<bug-name>
    - Implement features: feat/<feature-name>
    - Create documentation: docs/<doc-name>
    - Refactor: refactor/<refactored-part>
- Commits:
    - Fix bugs: fix: message
    - Implement features: implement: message
    - Create docs: docs: message
    - Refactor: refactor: message