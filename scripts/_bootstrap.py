from pathlib import Path
import sys


def project_root() -> Path:
    """Return project root and ensure it is on sys.path for src imports."""
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
