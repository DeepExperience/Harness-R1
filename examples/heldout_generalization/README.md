# Held-Out Task Generalization

Patches from the held-out experiment in the
[paper](https://arxiv.org/abs/2608.02276). Each editor sees the same 10 sampled
failures per benchmark, writes one benchmark-level patch, and that patch is
applied to the rest of the split — 1,270 held-out tasks, three evidence seeds,
against a frozen vanilla Qwen3.5-9B.

```
<editor>/seed<SEED>/<bench>.json   3 editors x 3 seeds x 3 benchmarks = 27 patches
results.json                       held-out counts, sha256, evidence task ids
```

| Editor | Valid | Δ held-out (1,270) |
|---|---:|---:|
| **Harness-R1** | **9/9** | **+8.92 ± 1.50 pp** |
| Qwen3.5-397B | 8/9 | −4.33 ± 2.52 pp |
| DeepSeek-V4-Pro | 6/9 | −0.42 ± 3.58 pp |

Mean ± sample std over the three seeds. 23 of the 27 patches validated and ship
here; the four the validator rejected are scored as no-patch (zero delta), with
its error recorded in `results.json`.

They load like [`../webshop_patch.json`](../webshop_patch.json):

```python
import json, sys
sys.path.insert(0, "code/life-harness/AgentBench/scripts")
from harness_r1_patch import normalize_patch, require_code_hook_only_patch

p = json.load(open("examples/heldout_generalization/harness-r1/seed20260720/webshop.json"))
require_code_hook_only_patch(normalize_patch(p, bench="webshop"))
```
