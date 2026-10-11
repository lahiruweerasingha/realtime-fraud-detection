# One image for every Spark role (master, workers, history server) and for the
# jobs, tests and EDA notebook, so the code always runs against the same
# library versions it was tested with.
FROM apache/spark:3.5.9-java17-python3

USER root

COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt pytest==9.1.1 \
    && rm /tmp/requirements.txt

# Make the bundled PySpark importable from plain `python3` and pytest, not
# only from spark-submit.
ENV SPARK_HOME=/opt/spark \
    PYTHONPATH=/opt/spark/python:/opt/spark/python/lib/py4j-0.10.9.7-src.zip:/app/src \
    PYSPARK_PYTHON=python3 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /app /opt/spark-events /tmp/matplotlib \
    && chown -R spark:spark /app /opt/spark-events /tmp/matplotlib

WORKDIR /app
USER spark
