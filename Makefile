.PHONY: install install-all data train evaluate api benchmark export test lint format docker-cpu docker-cuda

install:
	python -m pip install -e ".[dev,tracking,onnx]"

install-all:
	python -m pip install -e ".[all]"

data:
	python scripts/download_data.py --config configs/quickstart.yaml

train:
	python scripts/train.py --config configs/quickstart.yaml

evaluate:
	python scripts/evaluate.py --config configs/quickstart.yaml

api:
	uvicorn foodvision.api.main:app --host 0.0.0.0 --port 8000 --reload

benchmark:
	python scripts/benchmark.py --config configs/quickstart.yaml

export:
	python scripts/export_onnx.py --config configs/quickstart.yaml

test:
	pytest --cov=foodvision --cov-report=term-missing

lint:
	ruff check .
	mypy src/foodvision

format:
	ruff format .
	ruff check --fix .

docker-cpu:
	docker compose --profile cpu up --build

docker-cuda:
	docker compose --profile cuda up --build
