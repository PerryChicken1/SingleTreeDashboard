"""Check release metadata changes in a warm Streamlit process."""
import ast
import json
from pathlib import Path
import tempfile
import unittest

import streamlit as st

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "Dashboard.py"
if not SOURCE.exists():
    SOURCE = ROOT.parent / "SingleTreeOptTool/Dashboard.py"


class MetadataReloadTests(unittest.TestCase):
    def test_file_changes_are_visible_without_clearing_streamlit_cache(self):
        # Extract the actual loaders, including any decorators, without starting
        # the dashboard. This reproduces an update in an already-running process.
        module = ast.parse(SOURCE.read_text(encoding="utf-8"))
        loaders = [node for node in module.body if isinstance(node, ast.FunctionDef)
                   and node.name in {"load_static_text", "load_dashboard_metadata"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace = {"Path": Path, "json": json, "st": st, "STATIC_DIR": root}
            exec(compile(ast.Module(body=loaders, type_ignores=[]), str(SOURCE), "exec"), namespace)
            case_file = root / "test_datasets.json"
            case_file.write_text('{"basel": {"file": "basel.csv"}}', encoding="utf-8")
            before = json.loads(namespace["load_static_text"](case_file))
            self.assertNotIn("problem", before["basel"])
            case_file.write_text('{"basel": {"file": "basel.csv", "problem": "future_crop_tree_selection"}}', encoding="utf-8")
            after = json.loads(namespace["load_static_text"](case_file))
            self.assertEqual(after["basel"].get("problem"), "future_crop_tree_selection")

            metadata = root / "metadata.json"
            metadata.write_text('{"version": 1}', encoding="utf-8")
            self.assertEqual(namespace["load_dashboard_metadata"]()["version"], 1)
            metadata.write_text('{"version": 2}', encoding="utf-8")
            self.assertEqual(namespace["load_dashboard_metadata"]()["version"], 2)


if __name__ == "__main__":
    unittest.main()
