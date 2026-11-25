#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Live-server tests of the RDD API of Spark Connect through the drop-in, against a server of
the commit of the protos, which has the RDD API (.github/workflows/spark-branch-rdd.yml): the
members of the session and the DataFrame of it, the DataFrame of an RDD, the local data of
parallelize, and the closures that cloudpickle-rust pickles. The suites of the API itself, of
that commit (pyspark/tests/connect), run through scripts/run_official_tests.py --dropin."""

import os
import warnings

import pytest
from pyspark.sql import Row, SparkSession
from pyspark.sql.types import LongType, StringType, StructField, StructType

pytestmark = pytest.mark.skipif(
    not os.environ.get("SPARK_REMOTE"), reason="needs a live server (SPARK_REMOTE)"
)


@pytest.fixture(scope="module")
def spark():
    session = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate()
    try:
        session.client.execute_command  # noqa: B018
        if not hasattr(session.sparkContext, "parallelize"):
            pytest.skip("the server has no RDD API")
        yield session
    finally:
        session.stop()


def test_spark_context(spark):
    from pyspark.core.connect.context import SparkContext

    sc = spark.sparkContext
    assert isinstance(sc, SparkContext)
    assert spark.sparkContext is sc
    # RDDCommand.GetContextInfo and RDDCommand.Describe.
    assert sc.defaultParallelism >= 1
    assert sc.parallelize(range(10), 3).getNumPartitions() == 3


def test_dataframe_rdd(spark):
    df = spark.range(5)
    assert df.rdd is df.rdd
    assert df.rdd.map(lambda r: r.id * 2).collect() == [0, 2, 4, 6, 8]
    rows = spark.range(3).selectExpr("id", "cast(id as string) s").rdd.collect()
    assert rows == [Row(id=0, s="0"), Row(id=1, s="1"), Row(id=2, s="2")]


def test_create_dataframe_of_rdd(spark):
    sc = spark.sparkContext
    rows = sc.parallelize([Row(a=1, b="x"), Row(a=2, b="y")])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # Of an RDD, samplingRatio and verifySchema are not ignored, so there is no warning.
        df = spark.createDataFrame(rows, samplingRatio=1.0, verifySchema=True)
    assert df.collect() == [Row(a=1, b="x"), Row(a=2, b="y")]
    assert spark.createDataFrame(sc.parallelize([(1, "x")]), ["n", "s"]).columns == ["n", "s"]
    schema = StructType([StructField("n", LongType()), StructField("s", StringType())])
    assert spark.createDataFrame(sc.parallelize([(1, "x")]), schema).schema == schema
    assert spark.createDataFrame(sc.parallelize([(1, "x")]), "n long, s string").schema == schema
    assert spark.createDataFrame(sc.parallelize([1, 2]), LongType()).collect() == [
        Row(value=1),
        Row(value=2),
    ]
    with pytest.raises(Exception, match="LongType|long"):
        spark.createDataFrame(sc.parallelize([("x",)]), "n long").collect()
    # Of other data, they still warn, as PySpark Connect.
    with pytest.warns(UserWarning, match="'samplingRatio' is ignored"):
        spark.createDataFrame([(1,)], samplingRatio=1.0)


def test_parallelize_of_cached_local_data(spark):
    # Pickled batches over spark.sql.session.localRelationCacheThreshold (1 MiB) are cached as
    # artifacts, of a ChunkedCachedLocalRelation.
    sc = spark.sparkContext
    n = 300000
    assert sc.parallelize(range(n), 4).map(lambda x: x % 7).sum() == sum(x % 7 for x in range(n))
    assert sc.parallelize(list(range(n)), 4).count() == n


def test_closures_of_cloudpickle_rust(spark):
    from pyspark.serializers import _cloudpickle

    if _cloudpickle().__name__ != "cloudpickle_rust":
        pytest.skip("cloudpickle-rust is not installed, or PYSPARK_CLOUDPICKLE_RUST is not 1")
    sc = spark.sparkContext
    offset = 10

    class Point:
        def __init__(self, x):
            self.x = x

    # A local class and a closure, pickled by value by cloudpickle-rust, run on the server's
    # Python workers.
    assert sc.parallelize(range(3)).map(lambda x: Point(x + offset).x).collect() == [10, 11, 12]
