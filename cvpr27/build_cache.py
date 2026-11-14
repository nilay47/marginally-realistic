"""python -m cvpr27.build_cache   -> builds train/heldout caches under <root>/cache (skips if present)."""
import json

from cvpr27.data import build_all

if __name__ == "__main__":
    print(json.dumps(build_all(), indent=2))
