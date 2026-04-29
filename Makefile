.PHONY: setup etl train test run-api dev docker-up

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
