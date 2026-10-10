"""Run the v2 unit tests on this machine: python -m dropout_v2.selftest"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


def main() -> int:
    tests_dir = Path(__file__).resolve().parent / "tests"
    top_dir = tests_dir.parent.parent
    suite = unittest.defaultTestLoader.discover(str(tests_dir), pattern="test_*.py", top_level_dir=str(top_dir))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        print("\nSELFTEST PASSED: {} tests".format(result.testsRun))
        return 0
    print("\nSELFTEST FAILED: {} failures, {} errors".format(len(result.failures), len(result.errors)))
    return 1


if __name__ == "__main__":
    sys.exit(main())
