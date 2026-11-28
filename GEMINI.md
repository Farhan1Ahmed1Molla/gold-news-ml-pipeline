# Project Rules: gold-news-ml-pipeline (V2)

## Environment & Package Management (`uv` ONLY)
- Use `uv` exclusively for all Python execution and dependency management:
  - Run pipeline: `uv run python run-pipeline.py`
  - Run tests: `uv run pytest`
  - Add packages: `uv add <package>` (or `uv add --dev <package>`)
- **FORBIDDEN:** Never use `pip install`, `python -m venv`, or `Activate.ps1`.
- Database: Built-in `sqlite3` (`gold_database.db` in project root) using parameterized `?` queries only (never f-strings for SQL values).
- Keep hardware footprint light: run tasks sequentially without heavy background processes.

## Architecture & Workflow (Spec-Driven Development)
- Follow "Deep Modules" principles: isolate SQLite persistence (`storage.py`) and long-term accumulation signal rules (`signals.py`) behind clean, typed interfaces.
- **Planning First:** Output a clear Implementation Plan and Task List first. Do NOT modify or create code files until approved.
- **Scope Guardrails:** Keep `run-pipeline.py` as the single active entry point and keep updating `forecast_log.csv` alongside `gold_database.db`. Leave `Gold_News_Pipeline.ipynb` and `.github/workflows` untouched in this phase.
