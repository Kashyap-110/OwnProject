"""Print the Python version, installed package versions and machine resources.

Read-only. Run on the server with:  python3 docs/data_audit/env_check.py
"""
import importlib
import os
import platform
import sys

PACKAGES = [
    "pandas", "numpy", "scipy", "sklearn", "lightgbm", "xgboost", "catboost",
    "shap", "pyarrow", "joblib", "matplotlib", "fastapi", "pydantic", "uvicorn",
    "streamlit", "impala", "pytest",
]

print("python    ", sys.version.replace("\n", " "))
print("platform  ", platform.platform())
print("cpu_count ", os.cpu_count())

try:
    with open("/proc/meminfo") as f:
        for line in f.readlines()[:3]:
            print("meminfo   ", line.strip())
except OSError:
    print("meminfo    not available")

for name in PACKAGES:
    try:
        module = importlib.import_module(name)
        print("{:<10} {}".format(name, getattr(module, "__version__", "installed (version unknown)")))
    except Exception as exc:
        print("{:<10} NOT AVAILABLE ({})".format(name, exc.__class__.__name__))
