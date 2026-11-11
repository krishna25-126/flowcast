"""Step 1 sanity check: confirm the three raw tables load and match the data dictionary."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.utils import get_logger, load_config, resolve

log = get_logger("check_data")


def main() -> int:
    cfg = load_config()
    raw = resolve(cfg["paths"]["raw_dir"])
    ok = True
    for key, fname in cfg["raw_files"].items():
        p = raw / fname
        if not p.exists():
            log.error("MISSING %s", p)
            ok = False
            continue
        df = pd.read_csv(p)
        log.info("%-9s %7d rows x %2d cols  (%s)", key, *df.shape, fname)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
