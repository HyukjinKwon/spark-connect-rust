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

import os
import subprocess
import sys

import pyarrow as pa
import pyspark
import pyspark.sql.connect.proto as pb2
import pytest


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
