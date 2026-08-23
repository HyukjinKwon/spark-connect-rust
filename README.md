<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
-->

# Spark Connect Rust Client

A Rust rewrite of the PySpark **Spark Connect** client (`pyspark.sql.connect.*`),
exposed through PyO3 as a drop-in `pyspark` package. It speaks the same Spark
Connect gRPC/protobuf protocol and returns the same results, so existing
Spark Connect code runs unchanged against the same server - with the plan
building, transport, and Arrow decoding done in Rust.

> **Status: alpha, work in progress.** The core DataFrame/Column/functions API,
> transport, and Arrow result path work end-to-end against a real Spark Connect
> server. It is developed and validated against **Apache Spark 4.2.0**. API parity
> with the reference client is being closed iteratively and validated by running
> the official Apache test suite (see [Parity & testing](#parity--testing)).

## Why

The reference Spark Connect client is pure Python: it builds protobuf plans,
manages the gRPC channel, and decodes Arrow results in Python. This project moves
that work into Rust while keeping a byte-for-byte compatible Python surface, so it
can be a **drop-in replacement** - same imports, same API, same server.

In an end-to-end benchmark against the same Spark Connect server, it is faster or at
parity on every workload measured (see [Performance](#performance)).

## Architecture

A Cargo workspace of four crates plus a Python skin:

| Component | Path | Responsibility |
|---|---|---|
| `spark-connect-proto` | `crates/spark-connect-proto` | gRPC/protobuf codegen for `spark.connect.*` |
| `spark-connect-core` | `crates/spark-connect-core` | Transport: channel, retries, reattach, artifacts, errors |
| `spark-connect` | `crates/spark-connect` | DataFrame API: session, dataframe, column, functions, plan, group, catalog, window, readwriter, streaming, udf, types |
| `pyspark-rs` | `crates/pyspark-rs` | PyO3 bindings - builds the `_pyspark` extension module |
| Python skin | `python/pyspark` | The drop-in `pyspark` package (+ vendored `pyspark.pandas`, `cloudpickle`, `pyspark.testing`) |

See [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md) for detail.

## Installation / build

Requires a Rust toolchain, `protobuf-compiler`, and Python 3.9+.

```bash
# Build the wheel (mixed Python/Rust layout via maturin) and install it:
maturin build --release --out dist
pip install dist/pyspark_client_rust-*.whl
```

For local development without maturin, build the extension and copy it into the skin:

```bash
cargo build -p pyspark-rs --release
cp target/release/lib_pyspark.dylib python/pyspark/_pyspark.so   # .so/.dylib per platform
PYTHONPATH=python python -c "from pyspark.sql import SparkSession; ..."
```

## Usage

Identical to the reference Spark Connect client:

```python
from pyspark.sql import SparkSession, functions as sf

spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()

df = (
    spark.range(0, 1_000_000)
    .select((sf.col("id") * 2).alias("x"))
    .filter(sf.col("x") % 3 == 0)
)
print(df.count())
df.groupBy((sf.col("x") % 10).alias("k")).agg(sf.sum("x"), sf.avg("x")).show()
```

### Rust usage (native)

The pure-Rust API is synchronous and mirrors PySpark's surface:

```rust
use spark_connect::session::SparkSession;
use spark_connect::functions as f;

fn main() -> Result<()> {
    let spark = SparkSession::builder()
        .remote("sc://localhost:15002")
        .get_or_create()?;
    
    let df = spark
        .range(0, 1_000_000)?
        .select(vec![f::col("id") * 2])?
        .filter(f::col("id") % 3 == 0)?;
    
    println!("Count: {}", df.count()?);
    df.show(10)?;
    
    Ok(())
}
```

## Performance

End-to-end benchmark vs the reference `pyspark` client, both talking to the **same**
Spark Connect server, so the client implementation is the only variable.
Each workload is run in its own process (`--isolate`) with warmup to avoid
cross-workload and cold-start bias. Reproduce with:

```bash
python scripts/benchmark_e2e.py --isolate --label ours --out ours.json
python scripts/benchmark_e2e.py --compare reference.json ours.json
```

| workload | reference | ours | speedup |
|---|---|---|---|
| `range_collect_100k` | 163 ms | 35 ms | **4.6×** |
| `collect_wide_100k` | 243 ms | 62 ms | **3.9×** |
| `select_filter_count_500k` | 30 ms | 27 ms | 1.1× |
| `many_small_queries_50x` | 1631 ms | 1493 ms | 1.1× |
| `withcolumns_chain_200k` | 41 ms | 41 ms | 1.0× |
| `join_count_50k` | 59 ms | 51 ms | 1.1× |
| `groupby_agg_500k` | 384 ms | 391 ms | 1.0× |

The biggest wins are on result-transfer paths (Arrow decode in Rust). Aggregate-
heavy queries like `groupby_agg` are server-bound - the client does little, so
they land at parity, as expected. Numbers vary by hardware and server; treat them
as indicative, not absolute.

## Parity & testing

"Done" is defined empirically: **our client must match the reference client's
pass/skip/fail profile on the official Apache Spark connect test suite, run against
the same server.** A test the reference client also skips or fails (no local JVM,
single-node server, cluster-only feature) is environmental and not counted against
us - only a test the reference passes and ours fails is a regression.

```bash
# Runs each official Apache connect test file with the reference client AND ours,
# then diffs; exits non-zero on any regression.
python scripts/rust_parity_diff.py \
    --target-pyspark ./python \
    --remote sc://localhost:15002
```

Rust-side builder correctness is covered by **golden-proto tests** (plans, expressions,
and all SQL functions are asserted byte-exact against captured reference protos):

```bash
cargo test -p spark-connect -p spark-connect-core
```

See [`docs/design/ACCEPTANCE.md`](docs/design/ACCEPTANCE.md) and
[`docs/design/OFFICIAL_TESTS.md`](docs/design/OFFICIAL_TESTS.md).

## Getting started with Spark Connect server

You can run a Spark Connect server locally for development in two ways:

### Option 1: Docker Compose (recommended)

The repo includes a `docker-compose.yml` that starts a Spark Connect server on port 15002:

```bash
docker compose up --build -d
```

### Option 2: Local Spark distribution

1. [Download Spark 4.2.0](https://spark.apache.org/downloads.html) and unzip it
2. Set `SPARK_HOME` to the unzipped directory
3. Start the server with:

```bash
$SPARK_HOME/sbin/start-connect-server.sh \
  --packages "org.apache.spark:spark-connect_2.13:4.2.0,io.delta:delta-spark_2.13:4.2.0" \
  --conf "spark.driver.extraJavaOptions=-Divy.cache.dir=/tmp -Divy.home=/tmp" \
  --conf "spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension" \
  --conf "spark.sql.catalog.spark_catalog=org.apache.spark.sql.delta.catalog.DeltaCatalog"
```

The server listens on `sc://localhost:15002` by default.

### Sample data

The repo includes sample datasets in `datasets/` (people.csv, employees.json, kv1.txt, etc.)
mounted at `/opt/spark/work-dir/datasets` in the Docker container for use in tests and examples.

## Continuous integration

- **Build & test** - Rust build/test via `cargo test`
- **Golden-proto tests** - Validates all plans/expressions against reference protos
- **Parity gate** - Runs official Apache connect test suite against both clients
- **Benchmark** - End-to-end performance vs reference client
- **Lint** - `rustfmt --check`, `clippy`, and `ruff` for Python tooling

## Development

```bash
cargo fmt --all                 # format Rust
cargo clippy --workspace        # lint Rust
ruff check scripts/             # lint our Python tooling
ruff format scripts/
```

`ruff` is intentionally scoped to `scripts/`; the `python/pyspark` tree is largely
vendored from Apache Spark and kept byte-compatible with upstream.

Remaining parity work is tracked mechanically in the parity ledger at
`docs/parity/inventory.csv`; the official Apache connect test suite is the
authoritative gate (see [Parity & testing](#parity--testing)).

## License

Apache License 2.0.
