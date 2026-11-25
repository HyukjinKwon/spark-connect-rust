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
# Logical plans under the connect import path.
#
# The Rust core builds the plans of the DataFrame API. ``LogicalPlan`` is PySpark Connect's, from
# Apache Spark v4.3.0, the base of the plans that the modules of PySpark build themselves, such as
# those of ``pyspark.ml.connect.proto``: a plan of a child, of a ``plan(session)`` that is its
# ``spark.connect.Relation``, which ``DataFrame(plan, session)`` sends as it is, and of a
# ``to_proto(session)`` that is the ``spark.connect.Plan`` of it. It has no
# ``_collect_references``, ``print`` or ``_repr_html_``, which are of the plans and the Columns of
# PySpark Connect's DataFrame, which the core builds. ``_DataFramePlan`` is the plan of a
# DataFrame of the core, its ``DataFrame._plan``. ``Read``, ``Range``, ``SQL``, ``SubqueryAlias``,
# ``CachedRemoteRelation`` and ``RemoveRemoteCachedRelation`` are PySpark Connect's, which
# ``pyspark.testing.connectutils`` and ``pyspark.sql.connect.conversion`` build DataFrames of.
# ``LocalRelation`` and ``ChunkedCachedLocalRelation`` are PySpark Connect's too, of which the RDD
# API (``pyspark.core.connect``) builds the local data of ``SparkContext.parallelize``, cached as
# artifacts with ``SparkSession._cache_local_relation`` when it is large. The messages are those
# of ``pyspark.sql.connect.proto``.

from collections.abc import Iterator, Sequence
from threading import Lock
from typing import TYPE_CHECKING, Any, Optional, cast

import pyspark.sql.connect.proto as proto
from pyspark._pyspark import _NativePlan
from pyspark.errors import PySparkValueError
from pyspark.sql.column import Column
from pyspark.sql.connect.logging import logger

if TYPE_CHECKING:
    import pyarrow as pa


class LogicalPlan:
    _lock: Lock = Lock()
    # The next plan ID, as PySpark's, of the plans of the module where it runs without the Rust
    # core, as the tests run it on a Python worker of the server; the core numbers the others.
    _nextPlanId: int = 0

    def __init__(
        self, child: Optional["LogicalPlan"], references: Optional[Sequence["LogicalPlan"]] = None
    ) -> None:
        """

        Parameters
        ----------
        child : :class:`LogicalPlan`, optional.
            The child logical plan.
        references : list of :class:`LogicalPlan`, optional.
            The list of logical plans that are referenced as subqueries in this logical plan.
        """
        self._child = child
        self._root_plan_id = LogicalPlan._fresh_plan_id()

        self._references: Sequence[LogicalPlan] = references or []
        self._plan_id_with_rel: Optional[int] = None
        if len(self._references) > 0:
            assert all(isinstance(r, LogicalPlan) for r in self._references)
            self._plan_id_with_rel = LogicalPlan._fresh_plan_id()

    @property
    def _plan_id(self) -> int:
        return self._plan_id_with_rel or self._root_plan_id

    @staticmethod
    def _fresh_plan_id() -> int:
        # Of the one counter of the process from which the Rust core numbers the relations of its
        # plans too, as PySpark's numbers all of its plans: so the ID of a plan, which a
        # DataFrame(plan, session) keeps, is of no other relation.
        try:
            from pyspark import _pyspark
        except ImportError:
            with LogicalPlan._lock:
                plan_id = LogicalPlan._nextPlanId
                LogicalPlan._nextPlanId += 1
            return plan_id
        return _pyspark._fresh_plan_id()

    def _create_proto_relation(self) -> "proto.Relation":
        plan = proto.Relation()
        plan.common.plan_id = self._root_plan_id
        return plan

    def plan(self, session: Any) -> "proto.Relation":  # type: ignore[empty-body]
        ...

    def command(self, session: Any) -> "proto.Command":  # type: ignore[empty-body]
        ...

    def to_proto(self, session: Any, debug: bool = False) -> "proto.Plan":
        """
        Generates connect proto plan based on this LogicalPlan.

        Parameters
        ----------
        session : :class:`SparkConnectClient`, optional.
            a session that connects remote spark cluster.
        debug: bool
            if enabled, the proto plan will be printed.
        """
        # As PySpark's, of its relation set in the plan by the client, which compresses it as its
        # plan compression asks, with what it logs of it, of the relations of its placeholders.
        plan = proto.Plan()
        relation = self.plan(session)
        session._set_relation_in_plan(plan, relation)

        if debug:
            print(plan)

        return plan

    def _verify(self, session: Any) -> bool:
        """This method is used to verify that the current logical plan
        can be serialized to Proto and back and afterwards is identical."""
        plan = proto.Plan()
        plan.root.CopyFrom(self.plan(session))

        serialized_plan = plan.SerializeToString()
        test_plan = proto.Plan()
        test_plan.ParseFromString(serialized_plan)

        return test_plan == plan

    @property
    def observations(self) -> dict[str, Any]:
        if self._child is None:
            return {}
        else:
            return self._child.observations

    def _with_relations(self, root: "proto.Relation", session: Any) -> "proto.Relation":
        if len(self._references) == 0:
            return root
        else:
            # When there are references to other DataFrame, e.g., subqueries, build new plan like:
            # with_relations [id 10]
            #     root: plan  [id 9]
            #     reference:
            #          refs#1: [id 8]
            #          refs#2: [id 5]
            plan = proto.Relation()
            assert isinstance(self._plan_id_with_rel, int)
            plan.common.plan_id = self._plan_id_with_rel
            plan.with_relations.root.CopyFrom(root)
            plan.with_relations.references.extend([ref.plan(session) for ref in self._references])
            return plan


class _DataFramePlan(_NativePlan, LogicalPlan):
    """The plan of a DataFrame of the Rust core, as its ``DataFrame._plan``: its relation, as the
    core sends it, of the plan IDs of its session, which are its ``_plan_id`` and
    ``_root_plan_id``, as PySpark's plan of a DataFrame is the plan whose IDs the columns of the
    DataFrame refer to.

    The plan of the core is of its native base, ``_NativePlan``, which holds it and the parts of it
    that the garbage collector sees, and no DataFrame, as PySpark's plan object holds none; its
    ``__dict__`` holds the fields of PySpark's ``LogicalPlan``. Its ``plan`` is a placeholder that
    holds that relation, which the core builds at the call, as PySpark's ``plan`` builds the
    relation, and replaces the placeholder with when the plan built over it reaches the core, as
    ``DataFrame(plan, session)`` or a command does, also after the DataFrame is gone. It is one
    message, an extension whose value holds the relation encoded, so a plan of ``pyspark.ml`` over a
    deep DataFrame is never decoded in Python, as protobuf decoders limit the depth of what they
    decode. ``DataFrame(df._plan, session)`` is a DataFrame of the plan of ``df`` itself, as
    PySpark's is of the plan object of ``df``. Its copies are of its class and its fields, of the
    same plan for ``copy.copy`` and of a deep copy of it for ``copy.deepcopy``, as PySpark's."""

    def __init__(self, df: Any) -> None:
        # LogicalPlan's fields, in the order of its __init__, of the plan IDs of the DataFrame
        # rather than of new ones.
        plan_id, root_plan_id = self._plan_ids()
        self._child = None
        self._root_plan_id = plan_id if root_plan_id is None else root_plan_id
        self._references: Sequence[LogicalPlan] = []
        self._plan_id_with_rel: Optional[int] = None if root_plan_id is None else plan_id

    def plan(self, session: Any) -> "proto.Relation":
        # Built of the session of the client ``session``, as PySpark's plan builds the plan objects
        # that it holds of the client that it is given.
        return proto.Relation.FromString(self._relation_placeholder(session))

    @property
    def observations(self) -> dict[str, Any]:
        # As PySpark's plan of a DataFrame, whose ``observations`` are the Observations of its
        # plan by their names, which the DataFrame gives the executions of the client, and so do
        # the plans over it, such as a ``SubqueryAlias``: those of the core's plan.
        return self._observations()

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        # As PySpark's plan object of a DataFrame, which pickles its fields and no session: the
        # plan of the relation of the DataFrame as it was sent (``_column_reduce.sent_plan``), of
        # the functions of its UDFs, of the error of a plan that does not build and of the
        # Observations of the plan, a LogicalPlan on the client and on a Python worker of the
        # server, which has no class of the drop-in. So a plan of the Python client over this
        # one, such as a ``SubqueryAlias``, pickles plainly and unpickles on the worker, as
        # PySpark's does. One of the plan of a model fails as PySpark's, as the DataFrame's.
        return self._plan_reduce(self.observations)


class Read(LogicalPlan):
    def __init__(
        self,
        table_name: str,
        options: Optional[dict[str, str]] = None,
        is_streaming: Optional[bool] = None,
    ) -> None:
        super().__init__(None)
        self.table_name = table_name
        self.options = options or {}
        self._is_streaming = is_streaming

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.read.named_table.unparsed_identifier = self.table_name
        if self._is_streaming is not None:
            plan.read.is_streaming = self._is_streaming
        for k, v in self.options.items():
            plan.read.named_table.options[k] = v
        return plan

    # The docstring of PySpark Connect's `LogicalPlan.print`, which `inspect.getdoc` and `help()`
    # find of its `Read.print`, of none of its own: the `LogicalPlan` here has no `print`.
    def print(self, indent: int = 0) -> str:
        """
        Print the simple string representation of the current :class:`LogicalPlan`.

        Parameters
        ----------
        indent : int
            The number of leading spaces for the output string.

        Returns
        -------
        str
            Simple string representation of this :class:`LogicalPlan`.
        """
        return f"{' ' * indent}<Read table_name={self.table_name}>\n"


class CachedRemoteRelation(LogicalPlan):
    """Logical plan object for a DataFrame reference which represents a DataFrame that's been
    cached on the server with a given id."""

    def __init__(self, relation_id: str, spark_session: Any):
        super().__init__(None)
        self._relation_id = relation_id
        # Needs to hold the session to make a request itself.
        self._spark_session = spark_session

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.cached_remote_relation.relation_id = self._relation_id
        return plan

    def __del__(self) -> None:
        session = self._spark_session
        # If session is already closed, all cached DataFrame should be released.
        if session is not None and not session.client.is_closed and self._relation_id is not None:
            try:
                # The release of the core, as of the relations of its checkpoints: the command of
                # PySpark Connect, built on this thread, of the context of its requests, and sent
                # without waiting. The garbage collector runs a finalizer on any thread whenever
                # it collects, also during a call that a request of its own would wait on, so a
                # finalizer never waits on a request (PySpark Connect's releases from a thread).
                session._release_cached_remote_relation(self._relation_id)
            except Exception as e:
                logger.warning(f"RemoveRemoteCachedRelation failed with exception: {e}.")


class SubqueryAlias(LogicalPlan):
    """Alias for a relation."""

    def __init__(self, child: Optional["LogicalPlan"], alias: str) -> None:
        super().__init__(child)
        self._alias = alias

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        if self._child is not None:
            plan.subquery_alias.input.CopyFrom(self._child.plan(session))
        plan.subquery_alias.alias = self._alias
        return plan


class SQL(LogicalPlan):
    def __init__(
        self,
        query: str,
        args: Optional[list[Column]] = None,
        named_args: Optional[dict[str, Column]] = None,
        views: Optional[Sequence[SubqueryAlias]] = None,
    ) -> None:
        if args is not None:
            assert isinstance(args, list)
            assert all(isinstance(arg, Column) for arg in args)

        if named_args is not None:
            assert isinstance(named_args, dict)
            for k, arg in named_args.items():
                assert isinstance(k, str)
                assert isinstance(arg, Column)

        if views is not None:
            assert isinstance(views, list)
            assert all(isinstance(v, SubqueryAlias) for v in views)

        super().__init__(None, views)
        self._query = query
        self._args = args
        self._named_args = named_args
        self._views = views

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.sql.query = self._query

        if self._args is not None and len(self._args) > 0:
            plan.sql.pos_arguments.extend([arg.to_plan(session) for arg in self._args])
        if self._named_args is not None and len(self._named_args) > 0:
            for k, arg in self._named_args.items():
                plan.sql.named_arguments[k].CopyFrom(arg.to_plan(session))

        return self._with_relations(plan, session)

    def command(self, session: Any) -> "proto.Command":
        cmd = proto.Command()
        cmd.sql_command.input.CopyFrom(self.plan(session))
        return cmd


class Range(LogicalPlan):
    def __init__(
        self,
        start: int,
        end: int,
        step: int,
        num_partitions: Optional[int] = None,
    ) -> None:
        super().__init__(None)
        self._start = start
        self._end = end
        self._step = step
        self._num_partitions = num_partitions

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        plan.range.start = self._start
        plan.range.end = self._end
        plan.range.step = self._step
        if self._num_partitions is not None:
            plan.range.num_partitions = self._num_partitions
        return plan


class LocalRelation(LogicalPlan):
    """Creates a LocalRelation plan object based on a PyArrow Table."""

    def __init__(
        self,
        table: Optional["pa.Table"],
        schema: Optional[str] = None,
    ) -> None:
        import pyarrow as pa

        super().__init__(None)

        if table is None:
            assert schema is not None
        else:
            assert isinstance(table, pa.Table)

        assert schema is None or isinstance(schema, str)

        self._table = table

        self._schema = schema

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        if self._table is not None:
            plan.local_relation.data = self._serialize_table()

        if self._schema is not None:
            plan.local_relation.schema = self._schema
        return plan

    def _serialize_table(self) -> bytes:
        import pyarrow as pa

        assert self._table is not None, "table cannot be None"
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, self._table.schema) as writer:
            batches = self._table.to_batches()
            for b in batches:
                writer.write_batch(b)
        return sink.getvalue().to_pybytes()

    def _serialize_table_chunks(
        self,
        max_chunk_size_rows: int,
        max_chunk_size_bytes: int,
    ) -> Iterator[bytes]:
        """
        Serialize the table into multiple chunks, each up to max_chunk_size_bytes bytes
        and max_chunk_size_rows rows.
        Each chunk is a valid Arrow IPC stream.

        This method processes the table in fixed-size batches (1024 rows) for
        efficiency, matching the Scala implementation's batchSizeCheckInterval.

        Yields chunks one at a time to avoid materializing all chunks in memory.
        """
        import pyarrow as pa

        assert self._table is not None, "table cannot be None"
        assert self._table.num_rows > 0, "table must have at least one row"
        schema = self._table.schema

        # Calculate schema serialization size once (empty table = just schema)
        schema_size = len(self._serialize_batches_to_ipc([], schema))

        current_batches: list[pa.RecordBatch] = []
        current_size = schema_size

        for batch in self._table.to_batches(max_chunksize=min(1024, max_chunk_size_rows)):
            # Approximate batch size using raw column data (fast, ignores IPC overhead).
            # Calculating the real batch size of the IPC stream would require serializing each
            # batch separately, which adds overhead.
            batch_size = sum(arr.nbytes for arr in batch.columns)

            # If this batch would exceed limit and we have data, flush current chunk
            if len(current_batches) > 0 and current_size + batch_size > max_chunk_size_bytes:
                yield self._serialize_batches_to_ipc(current_batches, schema)
                current_batches = []
                current_size = schema_size

            current_batches.append(batch)
            current_size += batch_size

        # Flush remaining batches (guaranteed to have at least one due to assertion)
        yield self._serialize_batches_to_ipc(current_batches, schema)

    def _serialize_batches_to_ipc(
        self,
        batches: "list[pa.RecordBatch]",
        schema: "pa.Schema",
    ) -> bytes:
        """Helper method to serialize Arrow batches to IPC stream format."""
        import pyarrow as pa

        combined = pa.Table.from_batches(batches, schema=schema)
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, schema) as writer:
            writer.write_table(combined)
        return sink.getvalue().to_pybytes()

    def _serialize_schema(self) -> bytes:
        # the server uses UTF-8 for decoding the schema
        assert self._schema is not None, "schema cannot be None"
        return self._schema.encode("utf-8")

    def serialize(self, session: Any) -> bytes:
        p = self.plan(session)
        return bytes(p.local_relation.SerializeToString())

    def print(self, indent: int = 0) -> str:
        # The docstring of PySpark Connect's `LogicalPlan.print`, which `inspect.getdoc` and
        # `help()` find of its `LocalRelation.print`, of none of its own: the `LogicalPlan`
        # here has no `print`.
        """
        Print the simple string representation of the current :class:`LogicalPlan`.

        Parameters
        ----------
        indent : int
            The number of leading spaces for the output string.

        Returns
        -------
        str
            Simple string representation of this :class:`LogicalPlan`.
        """
        return f"{' ' * indent}<LocalRelation>\n"

    def _repr_html_(self) -> str:
        return """
        <ul>
            <li><b>LocalRelation</b></li>
        </ul>
        """


class ChunkedCachedLocalRelation(LogicalPlan):
    """Creates a CachedLocalRelation plan object based on a hash of a LocalRelation."""

    def __init__(self, data_hashes: list[str], schema_hash: Optional[str]) -> None:
        super().__init__(None)

        self._data_hashes = data_hashes
        self._schema_hash = schema_hash

    def plan(self, session: Any) -> "proto.Relation":
        plan = self._create_proto_relation()
        clr = plan.chunked_cached_local_relation

        # Add hex string hashes directly to protobuf
        for data_hash in self._data_hashes:
            clr.dataHashes.append(data_hash)
        if self._schema_hash is not None:
            clr.schemaHash = self._schema_hash

        return plan

    def print(self, indent: int = 0) -> str:
        # The docstring of PySpark Connect's `LogicalPlan.print`, which `inspect.getdoc` and
        # `help()` find of its `ChunkedCachedLocalRelation.print`, of none of its own: the
        # `LogicalPlan` here has no `print`.
        """
        Print the simple string representation of the current :class:`LogicalPlan`.

        Parameters
        ----------
        indent : int
            The number of leading spaces for the output string.

        Returns
        -------
        str
            Simple string representation of this :class:`LogicalPlan`.
        """
        return f"{' ' * indent}<ChunkedCachedLocalRelation>\n"

    def _repr_html_(self) -> str:
        return """
        <ul>
            <li><b>ChunkedCachedLocalRelation</b></li>
        </ul>
        """


class RemoveRemoteCachedRelation(LogicalPlan):
    def __init__(self, relation: CachedRemoteRelation) -> None:
        super().__init__(None)
        self._relation = relation

    def command(self, session: Any) -> "proto.Command":
        plan = self._create_proto_relation()
        plan.cached_remote_relation.relation_id = self._relation._relation_id
        cmd = proto.Command()
        cmd.remove_cached_remote_relation_command.relation.CopyFrom(plan.cached_remote_relation)
        return cmd


__all__ = ["LogicalPlan", "Read", "Range", "SQL", "SubqueryAlias", "CachedRemoteRelation"]


# The write operations of PySpark Connect's writers and stream writer, of Apache Spark v4.3.0
# (python/pyspark/sql/connect/plan.py), which the drop-in's writers, PySpark's classes, keep as their
# `_write` (and `_write_stream`) and send the command of, as PySpark's: a command over the relation
# of the plan of the DataFrame, the core's placeholder (see `_DataFramePlan`), which the client
# executes.


class WriteOperation(LogicalPlan):
    def __init__(self, child: "LogicalPlan") -> None:
        super().__init__(child)
        self.source: Optional[str] = None
        self.path: Optional[str] = None
        self.table_name: Optional[str] = None
        self.table_save_method: Optional[str] = None
        self.mode: Optional[str] = None
        self.sort_cols: list[str] = []
        self.partitioning_cols: list[str] = []
        self.clustering_cols: list[str] = []
        self.options: dict[str, Optional[str]] = {}
        self.num_buckets: int = -1
        self.bucket_cols: list[str] = []
        self.with_schema_evolution: bool = False

    def command(self, session: Any) -> proto.Command:
        assert self._child is not None
        plan = proto.Command()

        plan.write_operation.input.CopyFrom(self._child.plan(session))
        if self.source is not None:
            plan.write_operation.source = self.source
        plan.write_operation.sort_column_names.extend(self.sort_cols)
        plan.write_operation.partitioning_columns.extend(self.partitioning_cols)
        plan.write_operation.clustering_columns.extend(self.clustering_cols)
        plan.write_operation.with_schema_evolution = self.with_schema_evolution

        if self.num_buckets > 0:
            plan.write_operation.bucket_by.bucket_column_names.extend(self.bucket_cols)
            plan.write_operation.bucket_by.num_buckets = self.num_buckets

        for k in self.options:
            if self.options[k] is None:
                plan.write_operation.options.pop(k, None)
            else:
                plan.write_operation.options[k] = cast(str, self.options[k])

        if self.table_name is not None:
            plan.write_operation.table.table_name = self.table_name
            if self.table_save_method is not None:
                tsm = self.table_save_method.lower()
                if tsm == "save_as_table":
                    plan.write_operation.table.save_method = proto.WriteOperation.SaveTable.TableSaveMethod.TABLE_SAVE_METHOD_SAVE_AS_TABLE
                elif tsm == "insert_into":
                    plan.write_operation.table.save_method = (
                        proto.WriteOperation.SaveTable.TableSaveMethod.TABLE_SAVE_METHOD_INSERT_INTO
                    )
                else:
                    raise PySparkValueError(
                        errorClass="UNSUPPORTED_OPERATION",
                        messageParameters={"operation": tsm},
                    )
        elif self.path is not None:
            plan.write_operation.path = self.path

        if self.mode is not None:
            wm = self.mode.lower()
            if wm == "append":
                plan.write_operation.mode = proto.WriteOperation.SaveMode.SAVE_MODE_APPEND
            elif wm == "overwrite":
                plan.write_operation.mode = proto.WriteOperation.SaveMode.SAVE_MODE_OVERWRITE
            elif wm == "error":
                plan.write_operation.mode = proto.WriteOperation.SaveMode.SAVE_MODE_ERROR_IF_EXISTS
            elif wm == "ignore":
                plan.write_operation.mode = proto.WriteOperation.SaveMode.SAVE_MODE_IGNORE
            else:
                raise PySparkValueError(
                    errorClass="UNSUPPORTED_OPERATION",
                    messageParameters={"operation": self.mode},
                )
        return plan

    def print(self, indent: int = 0) -> str:
        # The docstring of PySpark Connect's `LogicalPlan.print`, which `inspect.getdoc` and `help()`
        # find of its `WriteOperation.print`, of none of its own: the `LogicalPlan` here has no
        # `print`.
        """
        Print the simple string representation of the current :class:`LogicalPlan`.

        Parameters
        ----------
        indent : int
            The number of leading spaces for the output string.

        Returns
        -------
        str
            Simple string representation of this :class:`LogicalPlan`.
        """
        i = " " * indent
        return (
            f"{i}"
            f"<WriteOperation source='{self.source}' "
            f"path='{self.path} "
            f"table_name='{self.table_name}' "
            f"table_save_method='{self.table_save_method}' "
            f"mode='{self.mode}' "
            f"sort_cols='{self.sort_cols}' "
            f"partitioning_cols='{self.partitioning_cols}' "
            f"clustering_cols='{self.clustering_cols}' "
            f"num_buckets='{self.num_buckets}' "
            f"bucket_cols='{self.bucket_cols}' "
            f"options='{self.options}'>"
        )

    def _repr_html_(self) -> str:
        return (
            f"<uL><li>WriteOperation <br />source='{self.source}'<br />"
            f"path: '{self.path}<br />"
            f"table_name: '{self.table_name}' <br />"
            f"table_save_method: '{self.table_save_method}' <br />"
            f"mode: '{self.mode}' <br />"
            f"sort_cols: '{self.sort_cols}' <br />"
            f"partitioning_cols: '{self.partitioning_cols}' <br />"
            f"clustering_cols: '{self.clustering_cols}' <br />"
            f"num_buckets: '{self.num_buckets}' <br />"
            f"bucket_cols: '{self.bucket_cols}' <br />"
            f"options: '{self.options}'<br />"
            f"</li></ul>"
        )


class WriteOperationV2(LogicalPlan):
    def __init__(self, child: "LogicalPlan", table_name: str) -> None:
        super().__init__(child)
        self.table_name: Optional[str] = table_name
        self.provider: Optional[str] = None
        self.partitioning_columns: list[Column] = []
        self.clustering_columns: list[str] = []
        self.options: dict[str, Optional[str]] = {}
        self.table_properties: dict[str, Optional[str]] = {}
        self.mode: Optional[str] = None
        self.overwrite_condition: Optional[Column] = None
        self.with_schema_evolution: bool = False

    def command(self, session: Any) -> proto.Command:
        assert self._child is not None
        plan = proto.Command()
        plan.write_operation_v2.input.CopyFrom(self._child.plan(session))
        if self.table_name is not None:
            plan.write_operation_v2.table_name = self.table_name
        if self.provider is not None:
            plan.write_operation_v2.provider = self.provider

        plan.write_operation_v2.partitioning_columns.extend(
            [c.to_plan(session) for c in self.partitioning_columns]
        )
        plan.write_operation_v2.clustering_columns.extend(self.clustering_columns)
        plan.write_operation_v2.with_schema_evolution = self.with_schema_evolution

        for k in self.options:
            if self.options[k] is None:
                plan.write_operation_v2.options.pop(k, None)
            else:
                plan.write_operation_v2.options[k] = cast(str, self.options[k])

        for k in self.table_properties:
            if self.table_properties[k] is None:
                plan.write_operation_v2.table_properties.pop(k, None)
            else:
                plan.write_operation_v2.table_properties[k] = cast(str, self.table_properties[k])

        if self.mode is not None:
            wm = self.mode.lower()
            if wm == "create":
                plan.write_operation_v2.mode = proto.WriteOperationV2.Mode.MODE_CREATE
            elif wm == "overwrite":
                plan.write_operation_v2.mode = proto.WriteOperationV2.Mode.MODE_OVERWRITE
                if self.overwrite_condition is not None:
                    plan.write_operation_v2.overwrite_condition.CopyFrom(
                        self.overwrite_condition.to_plan(session)
                    )
            elif wm == "overwrite_partitions":
                plan.write_operation_v2.mode = proto.WriteOperationV2.Mode.MODE_OVERWRITE_PARTITIONS
            elif wm == "append":
                plan.write_operation_v2.mode = proto.WriteOperationV2.Mode.MODE_APPEND
            elif wm == "replace":
                plan.write_operation_v2.mode = proto.WriteOperationV2.Mode.MODE_REPLACE
            elif wm == "create_or_replace":
                plan.write_operation_v2.mode = proto.WriteOperationV2.Mode.MODE_CREATE_OR_REPLACE
            else:
                raise PySparkValueError(
                    errorClass="UNSUPPORTED_OPERATION",
                    messageParameters={"operation": self.mode},
                )
        return plan


class WriteStreamOperation(LogicalPlan):
    def __init__(self, child: "LogicalPlan") -> None:
        super().__init__(child)
        self.write_op = proto.WriteStreamOperationStart()

    def command(self, session: Any) -> proto.Command:
        assert self._child is not None
        self.write_op.input.CopyFrom(self._child.plan(session))
        cmd = proto.Command()
        cmd.write_stream_operation_start.CopyFrom(self.write_op)
        return cmd
