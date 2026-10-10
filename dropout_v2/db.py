"""Impala access: connection and streaming of a query result into a Parquet file with a fixed schema."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dropout_v2 import config, utils


def connect():
    """Kerberos (GSSAPI) connection to Impala. Settings: config.IMPALA_* (overridable by env variables)."""
    from impala.dbapi import connect as impala_connect

    return impala_connect(host=config.IMPALA_HOST, port=config.IMPALA_PORT, auth_mechanism=config.IMPALA_AUTH,
                          kerberos_service_name=config.IMPALA_KERBEROS_SERVICE)


def _as_text(value):
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if isinstance(value, (_dt.date, _dt.datetime)):
        return value.isoformat()
    return str(value)


def _to_array(values: Sequence, arrow_type: pa.DataType) -> pa.Array:
    if pa.types.is_string(arrow_type):
        return pa.array([_as_text(v) for v in values], type=arrow_type)
    numbers = pd.to_numeric(pd.Series(list(values), dtype=object).map(_as_number), errors="coerce")
    if pa.types.is_integer(arrow_type):
        return pa.array(numbers.round().astype("Int64"), type=arrow_type, from_pandas=True)
    return pa.array(numbers.to_numpy(dtype="float64"), type=arrow_type, from_pandas=True)


def _as_number(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    if hasattr(value, "is_finite") and not isinstance(value, float):   # decimal.Decimal
        return float(value)
    return value


def rows_to_table(rows: List[tuple], schema: pa.Schema) -> pa.Table:
    """DB-API rows -> Arrow table with exactly `schema`; values are coerced, bad values become null."""
    if not rows:
        return pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema)
    columns = list(zip(*rows))
    if len(columns) != len(schema):
        raise ValueError("Row width {} does not match schema width {}".format(len(columns), len(schema)))
    arrays = [_to_array(values, field.type) for field, values in zip(schema, columns)]
    return pa.Table.from_arrays(arrays, schema=schema)


def fetch_to_parquet(query: str, out_path: Path, schema: pa.Schema, batch_size: int,
                     connect_fn: Callable = connect) -> int:
    """Run `query` and stream all rows into `out_path`. Writes atomically; returns the row count."""
    logger = utils.get_logger()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    conn = connect_fn()
    cursor = conn.cursor()
    # impyla reads 1,024 rows per server round trip by default; larger buffers make big extracts much faster.
    fetch_size = min(int(batch_size), 50000)
    try:
        cursor.arraysize = fetch_size
    except Exception:
        pass
    if hasattr(cursor, "_buffersize"):
        cursor._buffersize = fetch_size
    writer = None
    total = 0
    try:
        cursor.execute(query)
        names = [str(d[0]).lower().split(".")[-1] for d in cursor.description]
        if names != list(schema.names):
            raise ValueError("Query columns do not match the expected schema.\n  got:      {}\n  expected: {}".format(
                names, list(schema.names)))
        writer = pq.ParquetWriter(str(tmp), schema, compression="snappy")
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            writer.write_table(rows_to_table(rows, schema))
            total += len(rows)
            if total % (batch_size * 10) < len(rows):
                logger.info("  fetched %s rows into %s", "{:,}".format(total), out_path.name)
        writer.close()
        writer = None
        tmp.replace(out_path)
    finally:
        if writer is not None:
            writer.close()
        if tmp.exists():
            tmp.unlink()
        try:
            cursor.close()
        finally:
            conn.close()
    return total
