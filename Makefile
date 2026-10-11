# Fraud detection pipeline. `make help` lists the targets.

-include .env

COMPOSE := docker compose
EXECUTOR_MEMORY ?= 2g

# Git Bash on Windows rewrites container paths such as /opt/spark/... into
# Windows paths before Docker sees them. This switches that off; it has no
# effect on Linux or macOS.
export MSYS_NO_PATHCONV := 1

SUBMIT := $(COMPOSE) exec -T spark-master /opt/spark/bin/spark-submit \
	--master spark://spark-master:7077 \
	--conf spark.eventLog.enabled=true \
	--conf spark.eventLog.dir=file:/opt/spark-events \
	--conf spark.sql.adaptive.enabled=true \
	--executor-cores 2 \
	--executor-memory $(EXECUTOR_MEMORY) \
	--driver-memory 2g

RUNNER := $(COMPOSE) run --rm --no-deps runner

.PHONY: help build up down data validate features train report run eda test lint clean

help:
	@echo "make up        build the image and start Spark + MLflow"
	@echo "make data      check the PaySim CSV is in data/raw/"
	@echo "make run       validate -> features -> train -> report (on the cluster)"
	@echo "make report    rebuild evidence/RESULTS.md from evidence/*.json"
	@echo "make eda       execute notebooks/01_eda.ipynb with real outputs"
	@echo "make test      run the test suite in the Spark image"
	@echo "make lint      ruff lint + format check (needs ruff installed)"
	@echo "make down      stop the stack (volumes are kept)"
	@echo "make clean     stop the stack and delete volumes and processed data"

build:
	$(COMPOSE) build spark-master

up: build
	$(COMPOSE) up -d --wait spark-master spark-worker-1 spark-worker-2 spark-history mlflow
	@echo "Spark :8080  history :18080  MLflow :5000"

down:
	$(COMPOSE) down

data:
	@sh scripts/fetch_data.sh

validate: data
	$(SUBMIT) /app/jobs/validate_data.py

features:
	$(SUBMIT) /app/jobs/build_features.py

train:
	$(SUBMIT) /app/jobs/train.py

report:
	$(RUNNER) python3 scripts/report.py

run: validate features train report

eda: data
	$(RUNNER) jupyter nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/01_eda.ipynb

test:
	$(RUNNER) python3 -m pytest -p no:cacheprovider

lint:
	ruff check .
	ruff format --check .

clean:
	$(COMPOSE) down -v
	rm -rf data/processed/*.parquet spark-events output
