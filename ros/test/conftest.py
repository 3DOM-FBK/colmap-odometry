import sys
from pathlib import Path

# Make the test helpers and the frozen legacy reference importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
