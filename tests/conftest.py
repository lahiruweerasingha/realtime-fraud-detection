from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def spark():
    pyspark = pytest.importorskip("pyspark.sql")
    session = (
        pyspark.SparkSession.builder.master("local[2]")
        .appName("fraud-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture(scope="session")
def fixture_csv() -> str:
    return str(FIXTURES / "transactions.csv")
