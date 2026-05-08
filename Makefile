.PHONY: setup etl train validate-journal validate-street train-street-downscaler paper-eval test run-api dev docker-up

setup:
	pip install poetry
	poetry install
	cd frontend && npm install

etl:
	poetry run python src/data/etl.py
	poetry run python src/data/feature_engineering.py
	poetry run python src/data/grid_builder.py

train:
	poetry run python src/models/train.py

validate-journal:
	poetry run python scripts/run_journal_validation.py

validate-street:
	poetry run python scripts/validate_spatial_interpolation.py

train-street-downscaler:
	poetry run python scripts/train_street_downscaler.py

paper-eval:
	poetry run python scripts/run_paper_experiments.py

test:
	poetry run pytest tests/
	cd frontend && npm run lint
	poetry run mypy src/ --strict
	poetry run ruff check src/
	poetry run black --check src/

run-api:
	poetry run uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

dev:
	cd frontend && npm run dev

docker-up:
	docker-compose -f docker/docker-compose.yml up --build -d
