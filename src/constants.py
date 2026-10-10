"""
src/constants.py
----------------
Shared constants: AAMI class mapping, MIT-BIH symbol groups, paths.
"""

from pathlib import Path

# Root of the repo
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MITDB_DIR = DATA_DIR / "mitdb"
SPLITS_DIR = DATA_DIR / "splits"
DOCS_DIR = ROOT / "docs"
CONFIGS_DIR = ROOT / "configs"

FS_DEFAULT = 360
WINDOW_BEFORE = 90
WINDOW_AFTER = 120
BEAT_WINDOW = WINDOW_BEFORE + WINDOW_AFTER  # 210

# AAMI 5-class mapping (per de Chazal et al. 2004)
AAMI_MAP = {
    "N": ["N", "L", "R", "e", "j"],       # Normal
    "S": ["A", "a", "J", "S"],            # Supraventricular ectopic
    "V": ["V", "E"],                      # Ventricular ectopic
    "F": ["F"],                           # Fusion
    "Q": ["/", "f", "Q"],                 # Unknown / paced
}

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# Symbols we drop (not beats)
NON_BEAT_SYMBOLS = {
    "+", "~", "|", "!", "[", "]", '"', "@", "x",
    "(", ")", "p", "t", "u", "`", "'", "^", "\\", "=", "§",
}