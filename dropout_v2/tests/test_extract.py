from __future__ import annotations

import datetime as dt
import re
import shutil
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dropout_v2 import config, db, extract, schema, synthetic


def _tuples(frame: pd.DataFrame):
    clean = frame.astype(object).where(frame.notna(), None)
    return [tuple(r) for r in clean.itertuples(index=False)]


class FakeCursor:
    def __init__(self, years, schools, names_override=None):
        self.years, self.schools, self.names_override = years, schools, names_override
        self.rows, self.description = [], None

    def execute(self, query):
        if "GROUP BY" in query:
            frame = self.schools
        else:
            year = re.search(r"academic_year = '([0-9-]+)'", query).group(1)
            frame = self.years.get(year, pd.DataFrame(columns=schema.RAW_COLUMNS))
        names = self.names_override or list(frame.columns)
        self.description = [(n, None, None, None, None, None, None) for n in names]
        self.rows = _tuples(frame)

    def fetchmany(self, size):
        batch, self.rows = self.rows[:size], self.rows[size:]
        return batch

    def close(self):
        pass


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def close(self):
        pass


class TestQueries(unittest.TestCase):
    def test_year_query(self):
        q = extract.build_year_query("2024-25", 0)
        self.assertIn("academic_year = '2024-25'", q)
        self.assertIn("BETWEEN -3 AND 12", q)
        self.assertIn(config.TABLE, q)
        self.assertNotIn("fnv_hash", q)
        for col in schema.RAW_COLUMNS:
            self.assertRegex(q, r"AS {}\b".format(col))
        self.assertIn("fnv_hash", extract.build_year_query("2024-25", 2))
        self.assertIn("% 100 < 2", extract.build_year_query("2024-25", 2))

    def test_school_query(self):
        q = extract.build_school_query()
        for col in schema.SCHOOL_COLUMNS:
            self.assertRegex(q, r"AS {}\b".format(col))
        self.assertIn("GROUP BY", q)

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            extract.build_year_query("2024-25'; DROP TABLE x; --", 0)
        with self.assertRaises(ValueError):
            extract.build_year_query("2024-25", 101)


class TestConversion(unittest.TestCase):
    def test_rows_to_table_coerces_types(self):
        sch = pa.schema([("s", pa.string()), ("i", pa.int32()), ("f", pa.float64()), ("d", pa.string())])
        rows = [(b"abc", "5", Decimal("12.50"), dt.date(2016, 3, 1)),
                (None, None, "x", None),
                (123, 7.0, 3, dt.datetime(2015, 1, 2, 0, 0))]
        table = db.rows_to_table(rows, sch)
        self.assertEqual(table.column("s").to_pylist(), ["abc", None, "123"])
        self.assertEqual(table.column("i").to_pylist(), [5, None, 7])
        self.assertEqual(table.column("f").to_pylist(), [12.5, None, 3.0])
        self.assertEqual(table.column("d").to_pylist(), ["2016-03-01", None, "2015-01-02T00:00:00"])

    def test_empty_batch(self):
        sch = pa.schema([("s", pa.string())])
        self.assertEqual(db.rows_to_table([], sch).num_rows, 0)


class TestFetchAndExtract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.years, cls.schools = synthetic.make_raw_years(n_students=300, seed=4)

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(str(self.tmp), ignore_errors=True)

    def test_fetch_writes_all_batches(self):
        cursor = FakeCursor(self.years, self.schools)
        out = self.tmp / "2024-25.parquet"
        n = db.fetch_to_parquet(extract.build_year_query("2024-25", 0), out, schema.RAW_SCHEMA, batch_size=97,
                                connect_fn=lambda: FakeConnection(cursor))
        self.assertEqual(n, len(self.years["2024-25"]))
        table = pq.read_table(str(out))
        self.assertEqual(table.num_rows, n)
        self.assertEqual(table.schema.names, schema.RAW_COLUMNS)

    def test_fetch_uses_large_fetch_buffer(self):
        cursor = FakeCursor(self.years, self.schools)
        cursor._buffersize = None
        db.fetch_to_parquet(extract.build_year_query("2024-25", 0), self.tmp / "x.parquet", schema.RAW_SCHEMA,
                            batch_size=200000, connect_fn=lambda: FakeConnection(cursor))
        self.assertEqual(cursor.arraysize, 50000)
        self.assertEqual(cursor._buffersize, 50000)

    def test_fetch_rejects_column_mismatch(self):
        names = list(schema.RAW_COLUMNS)
        names[0] = "personid"
        cursor = FakeCursor(self.years, self.schools, names_override=names)
        out = self.tmp / "bad.parquet"
        with self.assertRaises(ValueError):
            db.fetch_to_parquet("SELECT ... academic_year = '2024-25'", out, schema.RAW_SCHEMA, 100,
                                connect_fn=lambda: FakeConnection(cursor))
        self.assertFalse(out.exists())
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_run_extract_writes_years_and_schools(self):
        cursor = FakeCursor(self.years, self.schools)
        summary = extract.run_extract(["2023-24", "2024-25"], sample_pct=0, refresh=False, batch_size=500,
                                      connect_fn=lambda: FakeConnection(cursor), root=self.tmp)
        raw = self.tmp / "full" / "raw"
        self.assertTrue((raw / "2023-24.parquet").exists())
        self.assertTrue((raw / "schools.parquet").exists())
        self.assertEqual(summary["2024-25"], len(self.years["2024-25"]))
        again = extract.run_extract(["2024-25"], 0, False, 500, connect_fn=lambda: FakeConnection(cursor),
                                    root=self.tmp)
        self.assertEqual(again["2024-25"], "skipped (exists)")

    def test_run_extract_fails_on_empty_year(self):
        cursor = FakeCursor({"2024-25": self.years["2024-25"].iloc[:0]}, self.schools)
        with self.assertRaises(RuntimeError):
            extract.run_extract(["2024-25"], 0, False, 500, connect_fn=lambda: FakeConnection(cursor), root=self.tmp)

    def test_sample_tag_folder(self):
        cursor = FakeCursor(self.years, self.schools)
        extract.run_extract(["2024-25"], sample_pct=2, refresh=False, batch_size=500,
                            connect_fn=lambda: FakeConnection(cursor), root=self.tmp)
        self.assertTrue((self.tmp / "sample2" / "raw" / "2024-25.parquet").exists())


if __name__ == "__main__":
    unittest.main()
