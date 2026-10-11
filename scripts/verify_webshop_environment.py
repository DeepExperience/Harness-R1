"""Compare a locally initialized WebShop environment to archived main-table tasks.

Run this in the actual WebShop worker Python environment, with web_agent_site
on PYTHONPATH and Java configured. This does not request any model or run agents.
"""

import argparse
import gzip
import hashlib
import importlib.metadata
import json
import random
import sys
from pathlib import Path


def canonical(value):
    if isinstance(value, dict):
        return [[str(k), canonical(v)] for k, v in sorted(value.items(), key=lambda item: str(item[0]))]
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted([canonical(item) for item in value], key=lambda item: json.dumps(item, ensure_ascii=False, separators=(",", ":")))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path,
        default=Path(__file__).resolve().parents[1] / "reproduction/webshop_e0/data/tasks.jsonl",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, help="Optional goals and prices snapshot; written only after all identities match")
    parser.add_argument("--outer-seed", action="store_true", help="Test the reporter's random.seed(233) before WebAgentTextEnv construction")
    args = parser.parse_args()
    for output in (args.output, args.snapshot):
        if output is not None and output.exists():
            parser.error("Refusing to overwrite %s" % output)
    from web_agent_site.envs import web_agent_text_env as module
    from web_agent_site import utils
    import inspect
    signature = inspect.signature(module.SimServer)
    if args.outer_seed:
        random.seed(233)
    print("Constructing WebShop with real assets and search index...", flush=True)
    env = module.WebAgentTextEnv(observation_mode="text", human_goals=True, goal_seed=233)
    server = env.server
    prices_hash = digest(server.product_prices)
    reference = [json.loads(line) for line in args.reference.read_text().splitlines() if line.strip()]
    mismatches = []
    tasks = []
    for expected in reference:
        index = expected["index"]
        env.reset(session=index)
        observation = env.observation
        instruction = observation.split("Instruction: [SEP] ", 1)[1].split(" [SEP]", 1)[0].strip()
        payload = {
            "protocol": "webshop_task_manifest_v1", "goal_seed": 233, "index": index,
            "instruction_sha256": digest(instruction), "goal_sha256": digest(server.goals[index]),
            "product_prices_sha256": prices_hash,
        }
        manifest = {**payload, "sha256": digest(payload)}
        changed = [k for k, v in expected["manifest"].items() if manifest.get(k) != v]
        if changed:
            mismatches.append({"index": index, "different_fields": changed, "actual": manifest})
        tasks.append({"index": index, "manifest": manifest})
    versions = {}
    for name in ("pyserini", "faiss-cpu", "numpy", "spacy", "en-core-web-lg", "torch", "agentrl-worker"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {
        "python": sys.version, "packages": versions,
        "webshop_module_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
        "simserver_accepts_goal_seed": "goal_seed" in signature.parameters,
        "outer_seed": args.outer_seed,
        "goals": len(server.goals), "products": len(server.product_prices),
        "product_prices_sha256": prices_hash, "checked_tasks": len(reference),
        "matching_tasks": len(reference) - len(mismatches), "mismatches": mismatches,
        "tasks": tasks,
        "data_filenames": [Path(p).name for p in (utils.DEFAULT_FILE_PATH, utils.DEFAULT_ATTR_PATH, utils.HUMAN_ATTR_PATH)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if not mismatches and args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.snapshot, "wt", encoding="utf-8") as stream:
            json.dump({
                "goal_seed": 233, "scope": "goals for archived test indices 0..499; full product price map",
                "goals": [server.goals[t["index"]] for t in reference],
                "product_prices": server.product_prices,
                "product_prices_sha256": prices_hash,
            }, stream, ensure_ascii=False)
    print(json.dumps({k: result[k] for k in ("goals", "products", "product_prices_sha256", "checked_tasks", "matching_tasks")}, indent=2))
    if mismatches:
        raise SystemExit("Task identities differ; inspect the output before any reward comparison")


if __name__ == "__main__":
    main()
