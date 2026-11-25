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
# The SparkSession of Spark Connect, the native class, under PySpark Connect's import path.
#
# As in PySpark, ``pyspark.sql.SparkSession`` is the classic class (``pyspark.sql.session``, of
# Apache Spark), whose builder makes a session of this class, which it is not a subclass of.
import functools
import sys
from collections.abc import Sized
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

from pyspark._pyspark import SparkSession, _install_builder
from pyspark.errors import AnalysisException as _AnalysisException
from pyspark.errors import PySparkTypeError as _PySparkTypeError
from pyspark.errors import PySparkValueError as _PySparkValueError

# As PySpark Connect's module of the session, which imports the modules of the classes of the
# objects of a session when it is imported: those of its DataFrames (and so those of their groups,
# merges, writers and columns, see `pyspark.sql.connect.dataframe`), of its runtime config, of its
# readers and of its streaming queries, by PySpark's imports here, and those of its observations
# and table arguments, which PySpark's client and module of UDTFs import. The native classes name
# these modules in their `__module__`, and cloudpickle pickles a class by reference only when its
# module is in `sys.modules`, else by value, as a dynamic class, which a Python worker of the
# server cannot unpickle: so an object of a session that a UDF or a `foreachBatch` function holds
# pickles as PySpark's, in a new interpreter too. The properties of the session import the modules
# of `catalog`, `tvf` and `dataSource` when they make the objects, as PySpark's do.
from pyspark.sql.connect import observation as _observation  # noqa: F401
from pyspark.sql.connect import table_arg as _table_arg  # noqa: F401
from pyspark.sql.connect.conf import RuntimeConf  # noqa: F401
from pyspark.sql.connect.dataframe import DataFrame  # noqa: F401
from pyspark.sql.connect.readwriter import DataFrameReader  # noqa: F401
from pyspark.sql.connect.streaming.query import StreamingQueryManager  # noqa: F401
from pyspark.sql.connect.streaming.readwriter import DataStreamReader  # noqa: F401

# As PySpark Connect's module of the session, which binds the conversions of `createDataFrame` and
# the types that they make when it is imported, with the first session: its `createDataFrame`
# calls these by their names here, so a name rebound in `pyspark.sql.types` or
# `pyspark.sql.pandas.types` after this import is not seen here, and one rebound before it, or
# here, is. The conversions of a pandas DataFrame read them here at each call
# (`pyspark.sql._pandas_input`), as PySpark's do; the converters that the conversion of a column
# imports when it is called (`create_arrow_array_from_pandas`) are those of the module of
# `sys.modules` then, in both.
from pyspark.sql.pandas.conversion import create_arrow_table_from_pandas  # noqa: F401
from pyspark.sql.pandas.types import (  # noqa: F401
    _deduplicate_field_names,
    from_arrow_type,
    to_arrow_schema,
)
from pyspark.sql.session import SparkSession as _PySparkSession
from pyspark.sql.session import classproperty as _classproperty
from pyspark.sql.types import (  # noqa: F401
    AtomicType,
    DataType,
    DayTimeIntervalType,
    MapType,
    StringType,
    StructType,
    TimestampType,
    _create_converter,
    _has_nulltype,
    _infer_schema,
    _make_type_verifier,
    _merge_type,
)

if TYPE_CHECKING:
    from pyspark.core.connect.context import SparkContext
    from pyspark.sql.connect.catalog import Catalog
    from pyspark.sql.connect.plan import ChunkedCachedLocalRelation, LocalRelation
    from pyspark.sql.dataframe import DataFrame as ParentDataFrame

__all__ = ["SparkSession"]

# As PySpark Connect's `SparkSession.builder`, of its class body: PySpark's classproperty, whose
# each access, of the class or of a session, is a new builder, with the docstring of the classic
# builder. The native class takes it from here (`_install_builder`), as the class of
# pyspark.sql.session cannot be imported while the extension loads.
_builder = _classproperty(lambda cls: cls.Builder())
_builder.__doc__ = _PySparkSession.builder.__doc__
_install_builder(_builder)
del _builder


# As PySpark Connect's `catalog` and `version`, `functools.cached_property`s of its class body,
# with the docstrings of the classic session's: the first access keeps the value in the
# `__dict__` of the session, which the next ones read, of PySpark's functions.
def catalog(self: "SparkSession") -> "Catalog":
    from pyspark.sql.connect.catalog import Catalog

    return Catalog(self)


def version(self: "SparkSession") -> str:
    result = self._client._analyze(method="spark_version").spark_version
    assert result is not None
    return result


for _function in (catalog, version):
    _function.__qualname__ = f"SparkSession.{_function.__name__}"
    _cached = functools.cached_property(_function)
    _cached.__doc__ = getattr(_PySparkSession, _function.__name__).__doc__
    _cached.__set_name__(SparkSession, _function.__name__)
    setattr(SparkSession, _function.__name__, _cached)
del catalog, version, _function, _cached


# The members of PySpark Connect's session of the RDD API of Spark Connect (pyspark.core.connect),
# of the connect-rdd branch of HyukjinKwon/spark, of its class body: the session-scoped
# SparkContext of `sparkContext`, the DataFrame of an RDD of `createDataFrame` (whose native
# method calls `_create_dataframe_from_connect_rdd` for an RDD), and the cache of the local data
# of `SparkContext.parallelize` (`_cache_local_relation`).
def _is_connect_rdd(obj: Any) -> bool:
    # A Spark Connect RDD only exists once its module was imported, so do not import it here.
    rdd_module = sys.modules.get("pyspark.core.connect.rdd")
    return rdd_module is not None and isinstance(obj, rdd_module.RDD)


def sparkContext(self: "SparkSession") -> "SparkContext":
    """
    Session-scoped Spark context entry for Connect RDD helpers.

    This returns a :class:`~pyspark.core.connect.context.SparkContext` tied to this
    Spark session. It does not expose a JVM-wide :class:`~pyspark.core.context.SparkContext`.

    .. versionadded:: 5.0.0
    """
    from pyspark.core.connect.context import SparkContext

    if getattr(self, "_connect_spark_context", None) is None:
        self._connect_spark_context = SparkContext(self)
    return self._connect_spark_context


def _inferSchema(
    self: "SparkSession",
    rdd: Any,
    samplingRatio: Optional[float] = None,
    names: Optional[list[str]] = None,
) -> StructType:
    """
    Infer the schema of an RDD of Row, dict, or tuple, as ``_inferSchema`` of Spark Classic.
    """
    first = rdd.first()
    if isinstance(first, Sized) and len(first) == 0:
        raise _PySparkValueError(
            errorClass="CANNOT_INFER_EMPTY_SCHEMA",
            messageParameters={},
        )

    configs = self._client.get_config_dict(
        "spark.sql.timestampType",
        "spark.sql.pyspark.inferNestedDictAsStruct.enabled",
        "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
        "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
    )
    prefer_timestamp_ntz = configs["spark.sql.timestampType"] == "TIMESTAMP_NTZ"
    infer_dict_as_struct = configs["spark.sql.pyspark.inferNestedDictAsStruct.enabled"] == "true"
    infer_array_from_first_element = (
        configs["spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled"] == "true"
    )
    infer_map_from_first_pair = (
        configs["spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled"] == "true"
    )

    if samplingRatio is None:
        schema = _infer_schema(
            first,
            names=names,
            infer_dict_as_struct=infer_dict_as_struct,
            prefer_timestamp_ntz=prefer_timestamp_ntz,
        )
        if _has_nulltype(schema):
            for row in rdd.take(100)[1:]:
                schema = _merge_type(
                    schema,
                    _infer_schema(
                        row,
                        names=names,
                        infer_dict_as_struct=infer_dict_as_struct,
                        infer_array_from_first_element=infer_array_from_first_element,
                        infer_map_from_first_pair=infer_map_from_first_pair,
                        prefer_timestamp_ntz=prefer_timestamp_ntz,
                    ),
                )
                if not _has_nulltype(schema):
                    break
            else:
                raise _PySparkValueError(
                    errorClass="CANNOT_DETERMINE_TYPE",
                    messageParameters={},
                )
    else:
        if samplingRatio < 0.99:
            rdd = rdd.sample(False, float(samplingRatio))
        schema = rdd.map(
            lambda row: _infer_schema(
                row,
                names,
                infer_dict_as_struct=infer_dict_as_struct,
                infer_array_from_first_element=infer_array_from_first_element,
                infer_map_from_first_pair=infer_map_from_first_pair,
                prefer_timestamp_ntz=prefer_timestamp_ntz,
            )
        ).reduce(_merge_type)
    return schema


def _createFromRDD(
    self: "SparkSession",
    rdd: Any,
    schema: Optional[Union[DataType, list[str]]],
    samplingRatio: Optional[float],
) -> tuple[Any, StructType]:
    """
    Create an RDD of the internal values of the rows of a DataFrame from an RDD, and the
    schema of the DataFrame, as ``_createFromRDD`` of Spark Classic.
    """
    if schema is None or isinstance(schema, (list, tuple)):
        struct = self._inferSchema(rdd, samplingRatio, names=schema)
        converter = _create_converter(struct)
        tupled_rdd = rdd.map(converter)
        if isinstance(schema, (list, tuple)):
            for i, name in enumerate(schema):
                struct.fields[i].name = name
                struct.names[i] = name
    elif isinstance(schema, StructType):
        struct = schema
        tupled_rdd = rdd
    else:
        raise _PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "schema",
                "expected_type": "list, None or StructType",
                "arg_type": type(schema).__name__,
            },
        )

    # convert python objects to sql data
    internal_rdd = tupled_rdd.map(struct.toInternal)
    return internal_rdd, struct


def _create_dataframe_from_connect_rdd(
    self: "SparkSession",
    data: Any,
    schema: Optional[Union[DataType, list[str]]],
    samplingRatio: Optional[float],
    verifySchema: bool,
) -> "ParentDataFrame":
    """
    Create a DataFrame from a Spark Connect RDD as ``_create_dataframe`` of Spark Classic does
    from an RDD: the server applies the schema to the internal values that the RDD pickles.
    """
    if isinstance(schema, StructType):
        verify_func = _make_type_verifier(schema) if verifySchema else lambda _: True

        def prepare(obj: Any) -> Any:
            verify_func(obj)
            return obj

    elif isinstance(schema, DataType):
        dataType = schema
        schema = StructType().add("value", schema)
        verify_func = (
            _make_type_verifier(dataType, name="field value") if verifySchema else lambda _: True
        )

        def prepare(obj: Any) -> Any:
            verify_func(obj)
            return (obj,)

    else:

        def prepare(obj: Any) -> Any:
            return obj

    rdd, struct = self._createFromRDD(data.map(prepare), schema, samplingRatio)
    return rdd._to_dataframe(struct)


def _cache_local_relation(
    self: "SparkSession",
    local_relation: "LocalRelation",
    local_relation_size_limit: int,
    max_chunk_size_rows: int,
    max_chunk_size_bytes: int,
    max_batch_of_chunks_size_bytes: int,
) -> "ChunkedCachedLocalRelation":
    """
    Cache the local relation at the server side if it has not been cached yet.

    This method serializes the input local relation into multiple data chunks and
    a schema chunk (if the schema is available) and uploads these chunks as artifacts
    to the server.

    The method collects a batch of chunks of size up to max_batch_of_chunks_size_bytes and
    uploads them together to the server.
    Uploading each chunk separately would require an additional RPC call for each chunk.
    Uploading all chunks together would require materializing all chunks in memory which
    may cause high memory usage on the client.
    Uploading batches of chunks is the middle-ground solution.

    Should only be called on a LocalRelation with a non-empty _table.
    """
    from pyspark.sql.connect.plan import ChunkedCachedLocalRelation

    assert local_relation._table is not None, "table cannot be None"
    has_schema = local_relation._schema is not None

    hashes = []
    current_batch = []
    current_batch_size = 0
    total_size = 0
    if has_schema:
        schema_chunk = local_relation._serialize_schema()
        current_batch.append(schema_chunk)
        current_batch_size += len(schema_chunk)
        total_size += len(schema_chunk)

    data_chunks = local_relation._serialize_table_chunks(
        max_chunk_size_rows, min(max_chunk_size_bytes, max_batch_of_chunks_size_bytes)
    )

    for chunk in data_chunks:
        chunk_size = len(chunk)
        total_size += chunk_size

        # Check if total size exceeds the limit
        if total_size > local_relation_size_limit:
            raise _AnalysisException(
                errorClass="LOCAL_RELATION_SIZE_LIMIT_EXCEEDED",
                messageParameters={
                    "actualSize": str(total_size),
                    "sizeLimit": str(local_relation_size_limit),
                },
            )

        # Check if adding this chunk would exceed batch size
        if (
            len(current_batch) > 0
            and current_batch_size + chunk_size > max_batch_of_chunks_size_bytes
        ):
            hashes += self._client.cache_artifacts(current_batch)
            # start a new batch
            current_batch = []
            current_batch_size = 0

        current_batch.append(chunk)
        current_batch_size += chunk_size
    hashes += self._client.cache_artifacts(current_batch)

    if has_schema:
        schema_hash = hashes[0]
        data_hashes = hashes[1:]
    else:
        schema_hash = None
        data_hashes = hashes
    return ChunkedCachedLocalRelation(data_hashes, schema_hash)


for _function in (
    _inferSchema,
    _createFromRDD,
    _create_dataframe_from_connect_rdd,
    _cache_local_relation,
):
    _function.__qualname__ = f"SparkSession.{_function.__name__}"
    setattr(SparkSession, _function.__name__, _function)
sparkContext.__qualname__ = "SparkSession.sparkContext"
SparkSession.sparkContext = property(sparkContext)  # type: ignore[attr-defined]
del _function, sparkContext


def _check_schema(schema: object) -> None:
    """PySpark Connect's check of the schema of ``createDataFrame``, of a DDL string as the server
    parsed it, against the ``AtomicType`` and the ``StructType`` of this module, which it reads
    when it is called, as PySpark's reads the names of its module: a struct reads its ``fields``,
    and an object that neither takes, nor a list, a tuple or None, raises ``NOT_EXPECTED_TYPE``.
    The session calls it before it reads any setting, when the names are not the native classes,
    of which its own checks are the same, also of an ``isinstance`` that raises."""
    if isinstance(schema, (AtomicType, StructType)):
        if isinstance(schema, StructType):
            len(schema.fields)  # type: ignore[attr-defined]
    elif not isinstance(schema, (list, tuple)) and schema is not None:
        raise _PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "schema",
                "expected_type": "list, None or StructType",
                "arg_type": type(schema).__name__,
            },
        )


def _reduce_session(session_id: str) -> tuple[Callable[[str], "SparkSession"], tuple[str]]:
    """The ``__reduce__`` of the session of ``session_id``, as PySpark Connect's: a function,
    which cloudpickle pickles by value, that gives back the active session of that ID. It reads
    the class of ``pyspark.sql.connect.session`` when it is called, so that on a Python worker
    of the server, whose PySpark's has no active session, it raises ``NO_ACTIVE_SESSION``."""

    def creator(old_session_id: str) -> "SparkSession":
        from pyspark.sql.connect.session import SparkSession

        return SparkSession._getActiveSessionIfMatches(old_session_id)

    # Named as PySpark's, so that the error of a plain pickle of a session names the same function.
    creator.__qualname__ = "SparkSession.__reduce__.<locals>.creator"
    return creator, (session_id,)
