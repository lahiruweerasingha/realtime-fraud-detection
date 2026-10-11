# One image for the Flink JobManager and TaskManager, the Kafka replay
# producer and the end-to-end verification, so every streaming component runs
# the same library versions.
#
# The model libraries match training (Python 3.10, XGBoost 2.1.4, MLflow
# 3.17.0). PyFlink pins older NumPy and PyArrow than the Spark image uses;
# `make stream-verify` measures that streamed scores still equal batch scores
# rather than assuming it.
FROM flink:1.20.5-scala_2.12-java17

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# PyFlink first, so its own dependency pins win; the scoring libraries are then
# resolved against them.
RUN pip3 install --no-cache-dir apache-flink==1.20.5
COPY docker/flink/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Kafka connector for the DataStream API. The "sql" artifact bundles the Kafka
# client; the thin connector would need kafka-clients on the classpath too.
RUN curl -fsSL -o /opt/flink/lib/flink-sql-connector-kafka-3.3.0-1.20.jar \
    https://repo1.maven.org/maven2/org/apache/flink/flink-sql-connector-kafka/3.3.0-1.20/flink-sql-connector-kafka-3.3.0-1.20.jar

# Checkpoint directory must exist and belong to the flink user before a named
# volume is mounted over it, or the volume is created root-owned.
RUN mkdir -p /opt/flink/checkpoints && chown -R flink:flink /opt/flink/checkpoints

ENV PYTHONPATH=/app/src \
    GIT_PYTHON_REFRESH=quiet
USER flink
