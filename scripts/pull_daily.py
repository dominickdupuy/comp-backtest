"""Pull the full 3-year daily universe from WRDS into the parquet cache."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from src.config import REPO_ROOT, load_config
from src.data.wrds_loader import load_data_bundle

load_dotenv(REPO_ROOT / ".env")
cfg = load_config()
comp = cfg.competition
t0 = time.time()
print(f"Pulling daily {cfg.start_date} -> {cfg.end_date}, "
      f"universe<= {comp['universe_size']} ...", flush=True)
bundle = load_data_bundle(
    cfg.start_date, cfg.end_date,
    universe_size=int(comp["universe_size"]),
    min_price=float(comp["min_price"]),
    use_cache=True,
)
print(f"DONE in {time.time()-t0:.0f}s | names={bundle.close.shape[1]} "
      f"days={bundle.close.shape[0]} "
      f"earnings_rows={0 if bundle.earnings is None else len(bundle.earnings)}",
      flush=True)
