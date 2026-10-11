# WebShop E0 Main-Table Reference

These are archived inputs and outputs of the identity-clean E0 **iter143** run
that produced **156/500 -> 211/500** successes and mean reward
**0.6595866666666667 -> 0.69923**. All 50 patches were valid and evaluated;
all 500 baseline/patched task manifests match.

See [the reproduction guide](../../docs/WEBSHOP_MAIN_TABLE.md) for installation,
environment checks, and the patch replay/generation commands. The main-table
schema was **`webshop_code_only`** with the materialized prompts in this folder.

| Path | Contents |
|---|---|
| `data/tasks.jsonl` | 500 instructions, initial user messages, complete task manifests, and paired rewards |
| `data/engineer_inputs.jsonl` | 50 original engineer inputs; only local source paths in metadata were removed |
| `data/engineer_prompt_prefix.json` | Exact shared system/user prefix |
| `data/target_system_prompt.json` | Original target system prompt |
| `data/prompt_hashes.json` | Canonical hashes of the unmodified prompt messages |
| `data/evaluation_settings.json` | Archived decoding/evaluation settings, excluding local paths, endpoints, and credentials |
| `data/batch_results.json`, `data/summary.json` | Archived per-batch and aggregate results |
| `data/provenance.json` | Source-record hashes and export details |
| `patches/webshop_noharness_bNNN/` | Original patch JSON and raw engineer response for batch NNN |
| `verification/` | Environment identity reports, data/index hashes, and export checks |
| `fixes/webshop_goal_seed.patch` | Incremental dependency fix for environments built from the original release |
| `full-artifacts.tar.gz` | Full standalone bundle, including both rollout sets and a verified goals/prices snapshot |

[Download the full bundle](full-artifacts.tar.gz?raw=true) (13,324,017 bytes).
Its SHA256 is:

```text
ebf3ba1da361b0d3ef4737b7426562b1d06216edf418c20cf9ca669a345ce7d8
```

The full bundle is an immutable export dated 2026-10-11. Its standalone
instructions describe its own directory layout. Use the repository guide above
for the commands installed under `scripts/`.

From this directory, verify the checked-in files with:

```bash
sha256sum -c SHA256SUMS
```

The full product catalog and Lucene index are benchmark assets and are not
included. The goal/price snapshot is for environment verification only; it is
not part of the engineer prompt or an answer source for the target agent.
