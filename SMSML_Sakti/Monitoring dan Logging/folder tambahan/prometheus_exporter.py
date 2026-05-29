"""Import wrapper for the required numbered exporter file.

Run this module with uvicorn:

    uvicorn prometheus_exporter:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path


numbered_file = Path(__file__).resolve().parents[1] / "3.prometheus_exporter.py"
module = SourceFileLoader("numbered_prometheus_exporter", str(numbered_file)).load_module()
app = module.app
