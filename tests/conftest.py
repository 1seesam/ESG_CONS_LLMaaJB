import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

LOCAL_TMP = ROOT / "tmp" / "pytest_tmp"
LOCAL_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TMPDIR", str(LOCAL_TMP))
os.environ.setdefault("TMP", str(LOCAL_TMP))
os.environ.setdefault("TEMP", str(LOCAL_TMP))
