DEFAULT := install

install:
	uv sync

run:
	uv run python -m src $(ARGS)
debug:
	uv run python3 -m pdb -m src
lint:
	@flake8 . --exclude "llm_sdk .venv" || true
	@mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
clean:
	rm -rf .venv
	rm -rf .mypy_cache
	rm -rf llm_sdk/__pycache__
	rm -rf llm_sdk/llm_sdk/__pycache__
	rm -rf src/__pycache__
	rm -rf data/output