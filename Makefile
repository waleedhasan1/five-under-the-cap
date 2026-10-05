.PHONY: install data features models app-data test lint app

install:
	uv sync

data:
	uv run python -m fivecap.ingest

features:
	@echo "Milestone 3/4: not implemented yet" && exit 1

models:
	@echo "Milestone 5: not implemented yet" && exit 1

app-data:
	@echo "Milestone 6: not implemented yet" && exit 1

test:
	uv run pytest -q

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

app:
	uv run streamlit run app/Home.py
