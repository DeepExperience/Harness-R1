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

`on_post_step` may return `inject_hint`; WebShop/ALFWorld can also support a
narrow `force_action` effect. Prefer state updates and soft guidance.

`on_before_action` may return `block_and_prompt`; where supported it can also
return `rewrite_action` or `force_action`. DBBench intentionally ignores
rewrite/force effects in the v1 runtime.

See [examples/webshop_patch.json](../examples/webshop_patch.json) and the
benchmark-specific `schema_prompt()` branches in
`code/life-harness/AgentBench/scripts/harness_r1_patch.py`.

## Validation

The validator checks the outer schema, benchmark-specific hook set, AST safety,
effect contract, action/tool syntax, code length, exact-answer leakage, and
runtime no-op behavior. Use `require_code_hook_only_patch()` in all new
training and evaluation code. Legacy DSL actions are accepted only by explicit
historical compatibility paths.

Leakage checks are benchmark scoped. For example, numbered ALFWorld instance
actions are rejected only for ALFWorld hooks, so unrelated strings in WebShop
or DBBench are not rejected by an ALFWorld-specific rule.
