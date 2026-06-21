"""Wait until at least N minute-months are cached, then run the intraday-vs-daily
experiment and exit. Lets the frequency comparison auto-produce on partial data
while the bulk pull continues."""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from src.config import REPO_ROOT, load_config
from src.data.relevance import liquid_universe_symbols
from src.data.taq_loader import cached_months
from src.data.wrds_loader import load_data_bundle

load_dotenv(REPO_ROOT / ".env")
NEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1

cfg = load_config()
comp = cfg.competition
data = load_data_bundle(cfg.start_date, cfg.end_date,
                        universe_size=int(comp["universe_size"]),
                        min_price=float(comp["min_price"]), use_cache=True)
symbols = sorted(set(liquid_universe_symbols(data, top_n=1000).values()))

while True:
    have = cached_months(symbols, cfg.start_date, cfg.end_date)
    if len(have) >= NEED:
        print(f"{len(have)} month(s) cached: {have}. Running experiment...", flush=True)
        break
    time.sleep(60)

subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "run_intraday_experiment.py"),
                "--top", "1000"], check=False)
