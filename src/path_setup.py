import sys
from pathlib import Path

# Find the project root (the folder containing 'src')
# This assumes this file is located at project_root/src/path_setup.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
