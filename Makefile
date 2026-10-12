.PHONY: install lint format test demo clean

install:
	pip install -r requirements.lock.txt
	pip install pytest ruff

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q -W ignore::UserWarning --cov=aiaforge --cov-report=term

demo:
	python examples/run_demo.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache benchmark.json
