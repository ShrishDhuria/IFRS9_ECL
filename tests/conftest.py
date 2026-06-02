"""Put the project root on sys.path so tests can import the flat ecl_engine_* modules."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
