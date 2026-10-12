# Fraud detection pipeline. `make help` lists the targets.

# Secrets for InfluxDB and Grafana live in .env, generated on first use from
# .env.example (see scripts/make_env.sh). GNU make builds .env automatically
# because it is included here and has a rule below.
-include .env

COMPOSE := docker compose
STREAM := $(COMPOSE) --profile streaming
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
STREAM_TOOLS := $(STREAM) run --rm --no-deps stream-tools

.PHONY: help env build up down data validate features train report run eda test lint clean \
	stream-data stream-up stream-reset stream-run stream-verify stream-down

help:
	@echo "Batch"
	@echo "  make up             build the image and start Spark + MLflow"
	@echo "  make data           check the PaySim CSV is in data/raw/"
	@echo "  make run            validate -> features -> train -> report (on the cluster)"
	@echo "                      (or one stage: make validate | features | train | report)"
	@echo "  make report         rebuild evidence/RESULTS.md from evidence/*.json"
	@echo "  make eda            execute notebooks/01_eda.ipynb with real outputs"
	@echo "Streaming (after make run)"
	@echo "  make stream-data    Spark: state snapshot + replay window for the stream"
	@echo "  make stream-up      stop Spark, start Kafka + Flink + InfluxDB + Grafana"
	@echo "  make stream-run     empty the topics, replay into Kafka, run Flink to the end"
	@echo "  make stream-reset   empty the Kafka topics and the monitoring bucket"
	@echo "  make stream-verify  check every streamed feature and score against batch"
	@echo "  make stream-down    stop the streaming services"
	@echo "Quality"
	@echo "  make env            create .env with random secrets (automatic)"
	@echo "  make test           run the test suite in the Spark image"
	@echo "  make lint           ruff lint + format check (needs ruff installed)"
	@echo "  make down           stop everything (volumes are kept)"
	@echo "  make clean          stop everything, delete volumes and processed data"

.env:
	@sh scripts/make_env.sh

env: .env

# --- Batch -------------------------------------------------------------------

build:
	$(COMPOSE) build spark-master

up: build
	$(COMPOSE) up -d --wait spark-master spark-worker-1 spark-worker-2 spark-history mlflow
	@echo "Spark :8080  history :18080  MLflow :5000"

down:
	$(STREAM) --profile tools down

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

# --- Streaming -----------------------------------------------------------------

stream-data:
	$(SUBMIT) /app/jobs/export_stream_state.py

stream-up:
	$(COMPOSE) stop spark-master spark-worker-1 spark-worker-2 spark-history
	$(STREAM) build flink-jobmanager
	$(STREAM) up -d --wait mlflow kafka influxdb grafana flink-jobmanager flink-taskmanager
	$(STREAM) run --rm kafka-init
	@echo "Flink :8081  Grafana :3000  InfluxDB :8086  MLflow :5000"

# Every run starts from empty topics and an empty monitoring bucket, so a
# second replay is never appended to the first (topics keep data forever).
# Topic deletion is asynchronous: wait until the topics are gone before
# kafka-init recreates them.
stream-reset:
	$(STREAM) exec -T kafka sh -c '\
		T=/opt/kafka/bin/kafka-topics.sh; B="--bootstrap-server localhost:9092"; \
		for t in transactions scores transactions-dlq; do $$T $$B --delete --if-exists --topic $$t; done; \
		for i in $$(seq 60); do \
			$$T $$B --list | grep -qxE "transactions|scores|transactions-dlq" || exit 0; sleep 1; \
		done; echo "topics still being deleted" >&2; exit 1'
	$(STREAM) run --rm kafka-init
	$(STREAM) exec -T influxdb influx delete --org "$(INFLUX_ORG)" --bucket "$(INFLUX_BUCKET)" \
		--start 1970-01-01T00:00:00Z --stop 2100-01-01T00:00:00Z

stream-run: stream-reset
	$(STREAM_TOOLS) python3 /app/streaming/producer.py
	$(STREAM) exec -T -e BOUNDED=1 flink-jobmanager flink run --python /app/streaming/job.py

stream-verify:
	$(STREAM_TOOLS) python3 /app/streaming/verify.py

stream-down:
	$(STREAM) stop kafka flink-jobmanager flink-taskmanager influxdb grafana

# --- Quality -------------------------------------------------------------------

test:
	$(RUNNER) python3 -m pytest -p no:cacheprovider

lint:
	ruff check .
	ruff format --check .

clean:
	$(STREAM) --profile tools down -v
	rm -rf data/processed/*.parquet data/processed/stream output
