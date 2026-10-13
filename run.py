"""Run the wind tunnel design.

    python run.py                      # inputs from inputs.py
    python run.py examples/mach3.py    # another Python input file (or a .toml)
    python run.py --quick              # coarse resolution, fast check
    python run.py --help               # all options
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))  # works without installing the package

from windtunnel.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(default_config=str(ROOT / "inputs.py")))
