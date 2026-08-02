# Patch Format

The active public protocol requires:

```text
<think>
short reusable failure analysis
</think>
<patch>
{...one JSON object...}
</patch>
```

The JSON object has this shape:

```json
{
  "schema_version": "harness-r1-patch-v1",
  "benchmark": "webshop",
  "description": "Short reusable behavior description.",
  "actions": [
    {
      "type": "add_code_hook",
      "hook": "on_init",
      "code": "def hook(ctx, nb):\n    nb['stage'] = 'inspect'\n    return None"
    }
  ]
}
```

`benchmark` is one of `webshop`, `alfworld`, or `dbbench`. Each lifecycle
hook may occur at most once, and each code string must define exactly one
top-level `hook(ctx, nb)` function. A small number of top-level helper
functions are allowed.

## Return Contracts

`on_init`:

```python
{"skills": [{"text": "..."}], "tool_hint": "..."}
```

`make_pre_hint`:

```python
{"message": "..."}
```

Hook bodies are ordinary Python — there is no fixed vocabulary for what the code
may compute. What is constrained is the **return value**: it is the hook's only
channel into the episode, and the host runtime honors just these effects.

| Hook | Effect `kind` it may return |
|---|---|
| `on_post_step` | `inject_hint`, `force_action` |
| `on_before_action` | `block_and_prompt`, `rewrite_action`, `force_action` |

Anything else is dropped and the step proceeds as if the hook returned `None`
(`_normalize_hook_result` in `code_runner.py`). The action string carried by
`rewrite_action` / `force_action` is free text; only numbered ALFWorld instance
actions are checked against the admissible set. DBBench ignores rewrite/force in
the v1 runtime, so prefer state updates and soft guidance there.

See [examples/webshop_patch.json](../examples/webshop_patch.json) and the
benchmark-specific `schema_prompt()` branches in
`code/life-harness/AgentBench/scripts/harness_r1_patch.py`.

## Validation

The validator checks the outer schema, benchmark-specific hook set, AST safety,
effect contract, action/tool syntax, code length, exact-answer leakage, and
runtime no-op behavior. `require_code_hook_only_patch()` enforces that a patch
contains nothing but `add_code_hook`; use it in all training and evaluation code.

Leakage checks are benchmark scoped. For example, numbered ALFWorld instance
actions are rejected only for ALFWorld hooks, so unrelated strings in WebShop
or DBBench are not rejected by an ALFWorld-specific rule.
