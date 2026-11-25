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
"""Offline tests of the RDD API of Spark Connect in the drop-in (pyspark.core.connect, vendored
from the commit of the protos): its exports, the members of the session and the DataFrame of it,
and the helpers of the session's module that PySpark Connect's of that commit has (the schema of
an RDD, the DataFrame of an RDD, the local data cached as artifacts), the readers of an RDD of
strings, the plans of its local data, and the protos."""

import datetime
import functools
import hashlib
import os
import subprocess
import sys

import pyarrow as pa
import pyspark
import pyspark.sql.connect.proto as pb2
import pyspark.sql.connect.session as connect_session
import pytest
from _mock_connect_server import mock_connect_session


def test_exports():
    from pyspark.core import rdd as facade
    from pyspark.core.connect.context import SparkContext as ConnectSparkContext

    # As PySpark's package when is_remote_only(), of the commit of the protos: the facade of the
    # RDDs of both clients, of its import path of pyspark.rdd too, and the Connect SparkContext.
    assert pyspark.RDD is facade.RDD
    assert pyspark.RDDBarrier is facade.RDDBarrier
    assert pyspark.rdd is facade and sys.modules["pyspark.rdd"] is facade
    from pyspark import SparkContext

    assert SparkContext is ConnectSparkContext
    assert "RDD" not in pyspark.__all__ and "SparkContext" not in pyspark.__all__


def test_session_and_dataframe_members():
    from pyspark.core.connect.rdd import RDD
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.session import SparkSession, _is_connect_rdd

    assert isinstance(SparkSession.__dict__["sparkContext"], property)
    assert isinstance(DataFrame.__dict__["rdd"], property)
    for name in (
        "_cache_local_relation",
        "_inferSchema",
        "_createFromRDD",
        "_create_dataframe_from_connect_rdd",
    ):
        assert getattr(SparkSession, name).__qualname__ == f"SparkSession.{name}"
    assert not _is_connect_rdd([1, 2])
    assert _is_connect_rdd(RDD.__new__(RDD))


def test_protos():
    rel = pb2.Relation()
    rel.rdd_relation.with_resources.profile_id = 3
    assert rel.WhichOneof("rel_type") == "rdd_relation"
    assert rel.rdd_relation.WhichOneof("relation_type") == "with_resources"
    command = pb2.Command()
    command.rdd_command.describe.rdd.CopyFrom(rel)
    assert pb2.Command.FromString(command.SerializeToString()) == command
    result = pb2.ExecutePlanResponse()
    result.rdd_command_result.info.num_partitions = 4
    assert result.WhichOneof("response_type") == "rdd_command_result"


def test_local_relation():
    from pyspark.sql.connect.plan import ChunkedCachedLocalRelation, LocalRelation

    table = pa.Table.from_arrays([pa.array([b"a", b"bc"], pa.large_binary())], names=["value"])
    relation = LocalRelation(table, schema="value binary").plan(None)
    assert relation.local_relation.schema == "value binary"
    read = pa.ipc.open_stream(relation.local_relation.data).read_all()
    assert read.equals(table)
    # Chunks of at most two rows (of batches of at most two rows), each an IPC stream.
    chunks = list(LocalRelation(table)._serialize_table_chunks(1, 1))
    assert [pa.ipc.open_stream(c).read_all().num_rows for c in chunks] == [1, 1]
    cached = ChunkedCachedLocalRelation(["h1", "h2"], "s").plan(None)
    assert list(cached.chunked_cached_local_relation.dataHashes) == ["h1", "h2"]
    assert cached.chunked_cached_local_relation.schemaHash == "s"


def test_portable_hash(monkeypatch):
    from pyspark import util

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    assert util.portable_hash(None) == 0
    assert util.portable_hash((None, 1)) & 0xFFFFFFFF == 219750521


@pytest.mark.parametrize(
    "setting, expected", [(None, "pyspark.cloudpickle"), ("1", "cloudpickle_rust"), ("0", None)]
)
def test_closures_pickle_with_cloudpickle_rust_when_enabled(setting, expected):
    # The drop-in is a Spark Connect client: the serializers pickle closures with cloudpickle-rust
    # (its `cloudpickle-rust` extra) only with PYSPARK_CLOUDPICKLE_RUST=1, as the Python workers of
    # the server load them. Of a new interpreter, as the choice is the process's.
    pytest.importorskip("cloudpickle_rust")
    env = {k: v for k, v in os.environ.items() if k != "PYSPARK_CLOUDPICKLE_RUST"}
    if setting is not None:
        env["PYSPARK_CLOUDPICKLE_RUST"] = setting
    code = (
        "from pyspark.serializers import CloudPickleSerializer, _cloudpickle\n"
        "payload = CloudPickleSerializer().dumps(lambda: 1)\n"
        "print(_cloudpickle().__name__, b'cloudpickle_rust' in payload)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.split()
    expected = expected or "pyspark.cloudpickle"
    assert out == [expected, str(expected == "cloudpickle_rust")]


class _LocalRDD:
    """An eager RDD of a list, of the methods that the session's helpers of the RDD API call."""

    def __init__(self, items):
        self.items = list(items)
        self.sampled = None

    def first(self):
        return self.items[0]

    def take(self, num):
        return self.items[:num]

    def sample(self, withReplacement, fraction):
        self.sampled = (withReplacement, fraction)
        return _LocalRDD(self.items)

    def map(self, f):
        return _LocalRDD(f(item) for item in self.items)

    def reduce(self, f):
        return functools.reduce(f, self.items)

    def _to_dataframe(self, struct):
        return struct, self.items


class _Session:
    """A session of the helpers of the RDD API of the session's module, of a client of the given
    settings, which records the batches of artifacts that it caches."""

    _inferSchema = connect_session._inferSchema
    _createFromRDD = connect_session._createFromRDD
    _create_dataframe_from_connect_rdd = connect_session._create_dataframe_from_connect_rdd
    _cache_local_relation = connect_session._cache_local_relation

    def __init__(self, **settings):
        self.settings = {
            "spark.sql.timestampType": "TIMESTAMP_LTZ",
            "spark.sql.pyspark.inferNestedDictAsStruct.enabled": "false",
            "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled": "false",
            "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled": "false",
            **settings,
        }
        self.batches = []
        self._client = self

    def get_config_dict(self, *keys):
        return {key: self.settings[key] for key in keys}

    def cache_artifacts(self, batch):
        self.batches.append(list(batch))
        return [hashlib.sha256(chunk).hexdigest() for chunk in batch]


def test_the_schema_of_an_rdd_is_inferred_as_spark_classics():
    from pyspark.errors import PySparkValueError
    from pyspark.sql import Row
    from pyspark.sql.types import (
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
        TimestampNTZType,
        TimestampType,
    )

    session = _Session()
    # Of the first row, of Rows, of tuples and of their names.
    assert session._inferSchema(_LocalRDD([Row(a=1, b="x")])) == StructType(
        [StructField("a", LongType()), StructField("b", StringType())]
    )
    assert session._inferSchema(_LocalRDD([(1, "x")]), names=["n", "s"]).names == ["n", "s"]
    # A null type is merged with those of the next of the first 100 rows, until it has none.
    schema = session._inferSchema(_LocalRDD([(None, "x"), (1, None), (2, "z")]))
    assert [field.dataType for field in schema.fields] == [LongType(), StringType()]
    for rows, condition in (
        ([(None,), (None,)], "CANNOT_DETERMINE_TYPE"),
        ([()], "CANNOT_INFER_EMPTY_SCHEMA"),
    ):
        with pytest.raises(PySparkValueError) as raised:
            session._inferSchema(_LocalRDD(rows))
        assert raised.value.getCondition() == condition
    # Of a sampling ratio, of the rows of a sample, which a ratio of 0.99 or more does not take.
    rows = _LocalRDD([(None, "x"), (1, None)])
    schema = session._inferSchema(rows, samplingRatio=0.5)
    assert [field.dataType for field in schema.fields] == [LongType(), StringType()]
    assert rows.sampled == (False, 0.5)
    rows = _LocalRDD([(1,)])
    assert session._inferSchema(rows, samplingRatio=1.0).names == ["_1"]
    assert rows.sampled is None
    # Of the settings of the session.
    stamp = _LocalRDD([(datetime.datetime(2020, 1, 1),)])
    assert session._inferSchema(stamp).fields[0].dataType == TimestampType()
    ntz = _Session(**{"spark.sql.timestampType": "TIMESTAMP_NTZ"})
    assert ntz._inferSchema(stamp).fields[0].dataType == TimestampNTZType()
    nested = _LocalRDD([({"k": 1},)])
    assert isinstance(session._inferSchema(nested).fields[0].dataType, MapType)
    as_struct = _Session(**{"spark.sql.pyspark.inferNestedDictAsStruct.enabled": "true"})
    assert isinstance(as_struct._inferSchema(nested).fields[0].dataType, StructType)


def test_the_rdd_of_a_dataframe_of_an_rdd_is_of_the_internal_values_of_its_rows():
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import Row
    from pyspark.sql.types import LongType, StringType, StructField, StructType

    session = _Session()
    # Of names, of the inferred types.
    internal, struct = session._createFromRDD(_LocalRDD([(1, "x"), (2, "y")]), ["n", "s"], None)
    assert struct.names == ["n", "s"] and [field.name for field in struct.fields] == ["n", "s"]
    assert internal.items == [(1, "x"), (2, "y")]
    # Of no schema, inferred of the Rows.
    internal, struct = session._createFromRDD(_LocalRDD([Row(a=1)]), None, None)
    assert struct == StructType([StructField("a", LongType())]) and internal.items == [(1,)]
    # Of a struct, as it is.
    schema = StructType([StructField("s", StringType())])
    internal, struct = session._createFromRDD(_LocalRDD([("x",)]), schema, None)
    assert struct is schema and internal.items == [("x",)]
    with pytest.raises(PySparkTypeError) as raised:
        session._createFromRDD(_LocalRDD([(1,)]), 5, None)
    assert raised.value.getCondition() == "NOT_EXPECTED_TYPE"

    # The DataFrame of the RDD: of a struct, whose rows are verified unless verifySchema is false.
    schema = StructType([StructField("n", LongType())])
    create = session._create_dataframe_from_connect_rdd
    assert create(_LocalRDD([(1,)]), schema, None, True) == (schema, [(1,)])
    with pytest.raises(PySparkTypeError):
        create(_LocalRDD([("x",)]), schema, None, True)
    assert create(_LocalRDD([("x",)]), schema, None, False) == (schema, [("x",)])
    # Of a data type that is not a struct: of a column `value` of it.
    struct, items = create(_LocalRDD([1, 2]), LongType(), None, True)
    assert struct.names == ["value"] and items == [(1,), (2,)]
    with pytest.raises(PySparkTypeError):
        create(_LocalRDD(["x"]), LongType(), None, True)
    assert create(_LocalRDD(["x"]), LongType(), None, False)[1] == [("x",)]
    # Of names, inferred.
    assert create(_LocalRDD([(1, "x")]), ["a", "b"], None, True)[0].names == ["a", "b"]


def test_local_data_is_cached_as_artifacts_of_batches_of_chunks():
    from pyspark.errors import AnalysisException
    from pyspark.sql.connect.plan import ChunkedCachedLocalRelation, LocalRelation

    table = pa.Table.from_arrays([pa.array(range(3000), pa.int64())], names=["v"])
    session = _Session()
    # The schema first, then the chunks of the table, of at most 1024 rows a batch, sent in batches
    # of at most 20000 bytes.
    cached = session._cache_local_relation(
        LocalRelation(table, schema="v bigint"), 1 << 30, 1024, 1 << 20, 20000
    )
    assert isinstance(cached, ChunkedCachedLocalRelation)
    chunks = [chunk for batch in session.batches for chunk in batch]
    hashes = [hashlib.sha256(chunk).hexdigest() for chunk in chunks]
    assert chunks[0] == b"v bigint"
    assert cached._schema_hash == hashes[0] and cached._data_hashes == hashes[1:]
    assert len(session.batches) > 1
    assert all(sum(map(len, batch)) <= 20000 for batch in session.batches)
    assert sum(pa.ipc.open_stream(chunk).read_all().num_rows for chunk in chunks[1:]) == 3000
    # Of no schema.
    session = _Session()
    cached = session._cache_local_relation(LocalRelation(table), 1 << 30, 1024, 1 << 20, 1 << 20)
    assert cached._schema_hash is None and len(session.batches) == 1
    assert cached._data_hashes == [hashlib.sha256(c).hexdigest() for c in session.batches[0]]
    # Local data larger than the limit of the server raises, as PySpark's.
    with pytest.raises(AnalysisException) as raised:
        _Session()._cache_local_relation(LocalRelation(table), 100, 1024, 1 << 20, 1 << 20)
    assert raised.value.getCondition() == "LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"


def test_the_readers_read_the_strings_of_an_rdd(monkeypatch):
    from pyspark.sql.connect.readwriter import DataFrameReader
    from pyspark.sql.types import StringType

    made = []

    class Client:
        def createDataFrame(self, data, schema):
            made.append((data.items, schema))
            return "the DataFrame of the strings"

    rdd = _LocalRDD(['{"a": 1}', 2])
    monkeypatch.setattr(connect_session, "_is_connect_rdd", lambda obj: obj is rdd)
    reader = DataFrameReader(Client())
    # An element that is not a string is read as its str.
    assert reader._strings_of_rdd(rdd) == "the DataFrame of the strings"
    assert made == [(['{"a": 1}', "2"], StringType())]
    # Any other path is itself.
    assert reader._strings_of_rdd(["p"]) == ["p"]


def test_the_session_and_the_dataframe_keep_their_spark_context_and_rdd():
    from pyspark.core.connect.context import SparkContext
    from pyspark.core.connect.rdd import RDD

    with mock_connect_session() as (spark, _):
        sc = spark.sparkContext
        assert isinstance(sc, SparkContext) and spark.sparkContext is sc
        df = spark.range(3)
        assert isinstance(df.rdd, RDD) and df.rdd is df.rdd
