"""Hash WebShop data and index files without loading the environment."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--webshop", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite %s" % args.output)
    root = args.webshop.resolve()
    files = [root / "data" / name for name in (
        "items_shuffle.json", "items_ins_v2.json", "items_human_ins.json",
    )]
    index = root / "search_engine/indexes"
    if not index.is_dir():
        parser.error("Expected the full product search index: %s" % index)
    files += sorted(p for p in index.iterdir() if p.is_file() and p.name != "write.lock")
    records = []
    for path in files:
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        relative = str(path.relative_to(root))
        records.append({"file": relative, "bytes": path.stat().st_size, "sha256": h.hexdigest()})
        print(relative, h.hexdigest(), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "scope": "File hashes of the supplied WebShop data/index; not a historical pip or image lockfile.",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
