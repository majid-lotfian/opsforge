"""Regression tests for the SQL query orchestration (no Azure credentials required)."""
import importlib
import sys
import types
import unittest
from unittest.mock import patch

# Provide a minimal fake Databricks module to test read-only SQL creation.
databricks = types.ModuleType("databricks")
databricks.sql = types.SimpleNamespace(connect=lambda **kwargs: None)
sys.modules.setdefault("databricks", databricks)
sys.modules.setdefault("databricks.sql", databricks.sql)

# Use a stub config so CI never reads local/cloud credentials.
config = types.ModuleType("app.config")
config.settings = types.SimpleNamespace(
    databricks_catalog="dbw_opsforge",
    databricks_schema="opsforge_monitoring",
)
sys.modules["app.config"] = config
data = importlib.import_module("app.data")


class TestDatabricksReadQueries(unittest.TestCase):
    def test_index_coverage_uses_one_joined_query(self):
        with patch.object(data, "_query", return_value=[{"total": 1000, "indexed": 990}]) as query:
            self.assertTrue(data.index_is_complete())
        self.assertEqual(query.call_count, 1)
        sql = query.call_args.args[0]
        self.assertIn("LEFT JOIN", sql)
        self.assertIn("DISTINCT issue_id", sql)

    def test_index_coverage_rejects_incomplete(self):
        with patch.object(data, "_query", return_value=[{"total": 1000, "indexed": 989}]):
            self.assertFalse(data.index_is_complete())

    def test_batch_comments_are_grouped_by_issue_id(self):
        returned = [
            {"issue_id": 10, "body": "first", "url": "https://example.org/1"},
            {"issue_id": 11, "body": "second", "url": "https://example.org/2"},
        ]
        with patch.object(data, "_query", return_value=returned) as query:
            grouped = data.issue_comments_batch([10, 11, 12])
        self.assertEqual(len(grouped[10]), 1)
        self.assertEqual(len(grouped[11]), 1)
        self.assertEqual(grouped[12], [])
        self.assertEqual(query.call_count, 1)
        self.assertIn("ROW_NUMBER()", query.call_args.args[0])
        self.assertEqual(query.call_args.args[1], [10, 11, 12])

    def test_empty_batch_does_not_query(self):
        with patch.object(data, "_query") as query:
            self.assertEqual(data.issue_comments_batch([]), {})
            query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
