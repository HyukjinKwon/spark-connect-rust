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
the plans of its local data, and the protos."""

import sys

import pyarrow as pa
import pyspark
import pyspark.sql.connect.proto as pb2
import pytest
from pyspark.serializers import CloudPickleSerializer


def test_exports():
    from pyspark.core import rdd as facade
    from pyspark.core.connect.context import SparkContext as ConnectSparkContext

    # As PySpark's package when not is_remote_only(): the facade of the RDDs of both clients.
    assert pyspark.RDD is facade.RDD
    assert pyspark.RDDBarrier is facade.RDDBarrier
    assert sys.modules["pyspark.rdd"] is facade
    # The Connect SparkContext, imported at its first use.
    from pyspark import SparkContext

    assert SparkContext is ConnectSparkContext
    assert "RDD" not in pyspark.__all__ and "SparkContext" not in pyspark.__all__
    with pytest.raises(AttributeError, match="no attribute 'NoSuchName'"):
        pyspark.NoSuchName


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


def test_closures_pickle_with_cloudpickle_rust():
    # The drop-in depends on cloudpickle-rust: the serializers pickle closures with it.
    pytest.importorskip("cloudpickle_rust")
    from pyspark.serializers import _cloudpickle

    assert _cloudpickle().__name__ == "cloudpickle_rust"
    assert b"cloudpickle_rust" in CloudPickleSerializer().dumps(lambda x: x + 1)
