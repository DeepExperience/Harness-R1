"""Sandboxed Python hook runner for Harness-R1 code patches.

The hooks accepted here are intentionally small, synchronous Python functions.
They are not a general plugin mechanism: imports, file/network access, dunder
escape paths, and unsafe builtins are rejected before execution.
"""

from __future__ import annotations

import ast
import copy
import math
import re
import signal
import sys
import threading
from difflib import SequenceMatcher
from types import FrameType
from typing import Any, Callable, Optional


class HookCompileError(ValueError):
    pass


class HookRuntimeTimeout(BaseException):
    pass


HOOK_NAMES = {"on_init", "make_pre_hint", "on_before_action", "on_post_step"}
ALLOWED_EFFECT_KINDS = {
    "on_before_action": {"block_and_prompt", "force_action", "rewrite_action"},
    "on_post_step": {"inject_hint", "force_action"},
}
SAFE_BUILTIN_NAMES = {
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "Exception",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
}
FORBIDDEN_NAMES = {
    "breakpoint",
    "classmethod",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "exit",
    "getattr",
    "globals",
    "help",
    "input",
    "locals",
    "memoryview",
    "object",
    "open",
    "property",
    "quit",
    "setattr",
    "staticmethod",
    "super",
    "type",
    "vars",
    "__import__",
}
FORBIDDEN_NODES = (
    ast.AsyncFunctionDef,
    ast.Await,
    ast.ClassDef,
    ast.Delete,
    ast.Global,
    ast.Import,
    ast.ImportFrom,
    ast.Lambda,
    ast.Nonlocal,
    ast.Raise,
    ast.With,
    ast.AsyncWith,
    ast.While,
    ast.Yield,
    ast.YieldFrom,
)
ALFWORLD_INSTANCE_ACTION_RE = re.compile(
    r"\b(?:go to|take|pick up|put|open|close|toggle|clean|heat|cool|slice|examine|use)\b.*\b\d+\b",
    re.IGNORECASE,
)
MAX_HOOK_SOURCE_CHARS = 8000
MAX_HOOK_AST_NODES = 1200
MAX_HOOK_STR_CHARS = 700
MAX_RETURN_TEXT_CHARS = 900
MAX_HELPER_FUNCTIONS = 5


class _AttrDict(dict):
    """Dict copy that also supports read-style attribute access in hooks."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _to_attrdict(value: Any) -> Any:
    if isinstance(value, dict):
        return _AttrDict({key: _to_attrdict(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_attrdict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_attrdict(item) for item in value)
    return value


def compile_hook(
    source: str,
    *,
    benchmark: Optional[str] = None,
) -> Callable[[dict[str, Any], dict[str, Any]], Any]:
    """Compile a source string containing def hook(ctx, nb) plus small helpers."""
    if not isinstance(source, str) or not source.strip():
        raise HookCompileError("hook code must be a non-empty string")
    if len(source) > MAX_HOOK_SOURCE_CHARS:
        raise HookCompileError(f"hook code exceeds {MAX_HOOK_SOURCE_CHARS} characters")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise HookCompileError(f"hook code has invalid syntax: {exc}") from exc

    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) != len(tree.body):
        raise HookCompileError("hook code may contain only top-level function definitions")
    hook_funcs = [node for node in funcs if node.name == "hook"]
    if len(hook_funcs) != 1:
        raise HookCompileError("hook code must contain exactly one top-level hook function")
    helpers = [node for node in funcs if node.name != "hook"]
    if len(helpers) > MAX_HELPER_FUNCTIONS:
        raise HookCompileError(f"hook code may contain at most {MAX_HELPER_FUNCTIONS} helper functions")
    func = hook_funcs[0]
    helper_names = set()
    for helper in helpers:
        if helper.name.startswith("_") or helper.name in FORBIDDEN_NAMES:
            raise HookCompileError(f"helper function uses forbidden name: {helper.name}")
        if helper.name in SAFE_BUILTIN_NAMES or helper.name in {"math", "re", "SequenceMatcher", "hook"}:
            raise HookCompileError(f"helper function shadows reserved name: {helper.name}")
        if helper.name in helper_names:
            raise HookCompileError(f"duplicate helper function: {helper.name}")
        helper_names.add(helper.name)
    _validate_function_signature(func, hook=True)
    for helper in helpers:
        _validate_function_signature(helper, hook=False)

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_HOOK_AST_NODES:
        raise HookCompileError(f"hook AST exceeds {MAX_HOOK_AST_NODES} nodes")
    top_level_func_ids = {id(node) for node in funcs}
    for node in nodes:
        if isinstance(node, FORBIDDEN_NODES):
            raise HookCompileError(f"hook uses forbidden syntax: {type(node).__name__}")
        if isinstance(node, ast.FunctionDef) and id(node) not in top_level_func_ids:
            raise HookCompileError("nested functions are not allowed")
        if isinstance(node, ast.Try):
            _validate_try_node(node)
        if isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in FORBIDDEN_NAMES:
                raise HookCompileError(f"hook uses forbidden name: {node.id}")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_NAMES:
                raise HookCompileError(f"hook uses forbidden attribute: {node.attr}")
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if len(node.value) > MAX_HOOK_STR_CHARS:
                raise HookCompileError("hook string literal is too long")
            if (
                benchmark == "alfworld"
                and ALFWORLD_INSTANCE_ACTION_RE.search(" ".join(node.value.lower().split()))
            ):
                raise HookCompileError("hook must not hard-code numbered ALFWorld instance actions")

    safe_builtins = {name: getattr(__builtins__, name, None) for name in SAFE_BUILTIN_NAMES}
    if isinstance(__builtins__, dict):
        safe_builtins = {name: __builtins__[name] for name in SAFE_BUILTIN_NAMES if name in __builtins__}
    namespace = {"__builtins__": safe_builtins, "math": math, "re": re, "SequenceMatcher": SequenceMatcher}
    try:
        compiled = compile(tree, "<harness_code_hook>", "exec")
        exec(compiled, namespace)
    except Exception as exc:
        raise HookCompileError(f"hook failed to compile: {exc}") from exc
    hook = namespace.get("hook")
    if not callable(hook):
        raise HookCompileError("hook object is not callable")
    return hook


def _validate_function_signature(func: ast.FunctionDef, *, hook: bool) -> None:
    if func.decorator_list:
        raise HookCompileError("decorators are not allowed")
    if func.returns is not None:
        raise HookCompileError("return annotations are not allowed")
    args = func.args
    if hook:
        if (
            args.posonlyargs
            or args.vararg
            or args.kwonlyargs
            or args.kw_defaults
            or args.kwarg
            or args.defaults
        ):
            raise HookCompileError("hook signature may use only plain positional arguments")
        arg_names = [arg.arg for arg in args.args]
        if arg_names != ["ctx", "nb"]:
            raise HookCompileError("hook signature must be exactly hook(ctx, nb)")
    all_args = list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs)
    if args.vararg is not None:
        all_args.append(args.vararg)
    if args.kwarg is not None:
        all_args.append(args.kwarg)
    seen_arg_names = set()
    for arg in all_args:
        name = arg.arg
        if name in seen_arg_names:
            raise HookCompileError(f"duplicate argument name: {name}")
        seen_arg_names.add(name)
        if name.startswith("_") or name in FORBIDDEN_NAMES:
            raise HookCompileError(f"helper argument uses forbidden name: {name}")
        if arg.annotation is not None:
            raise HookCompileError("argument annotations are not allowed")
    if hook:
        return
    defaults = list(args.defaults) + [value for value in args.kw_defaults if value is not None]
    for default in defaults:
        if not isinstance(default, ast.Constant) or not isinstance(default.value, (str, int, float, bool, type(None))):
            raise HookCompileError("helper defaults may only be simple scalar constants")


def _validate_try_node(node: ast.Try) -> None:
    if node.orelse or node.finalbody:
        raise HookCompileError("try/except may not use else or finally")
    if not node.handlers:
        raise HookCompileError("try must catch Exception explicitly")
    for handler in node.handlers:
        if not isinstance(handler.type, ast.Name) or handler.type.id != "Exception":
            raise HookCompileError("try/except may only use except Exception")
        if handler.name and (handler.name.startswith("_") or handler.name in FORBIDDEN_NAMES):
            raise HookCompileError(f"exception alias uses forbidden name: {handler.name}")


def run_hook(
    fn: Callable[[dict[str, Any], dict[str, Any]], Any],
    ctx: dict[str, Any],
    nb: dict[str, Any],
    *,
    hook_name: str,
    time_budget_s: float = 0.05,
    line_budget: int = 2000,
) -> Optional[dict[str, Any]]:
    """Run a compiled hook and normalize its return value.

    Runtime failures are intentionally downgraded to None so a bad hook cannot
    crash an episode.
    """
    if hook_name not in HOOK_NAMES:
        return None
    ctx_copy = _to_attrdict(copy.deepcopy(ctx)) if isinstance(ctx, dict) else _AttrDict()
    previous_handler = None
    use_signal = threading.current_thread() is threading.main_thread() and hasattr(signal, "setitimer")

    def timeout_handler(signum: int, frame: Optional[FrameType]) -> None:
        raise HookRuntimeTimeout("hook timed out")

    line_count = 0

    def trace(frame: FrameType, event: str, arg: Any) -> Any:
        nonlocal line_count
        if event == "line":
            line_count += 1
            if line_count > line_budget:
                raise HookRuntimeTimeout("hook line budget exceeded")
        return trace

    try:
        if use_signal:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, max(0.001, float(time_budget_s)))
        old_trace = sys.gettrace()
        sys.settrace(trace)
        try:
            result = fn(ctx_copy, nb)
        finally:
            sys.settrace(old_trace)
        return _normalize_hook_result(hook_name, result, ctx_copy)
    except HookRuntimeTimeout:
        return None
    except Exception:
        return None
    finally:
        if use_signal:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if previous_handler is not None:
                signal.signal(signal.SIGALRM, previous_handler)


def _normalize_hook_result(hook_name: str, result: Any, ctx: dict[str, Any]) -> Optional[dict[str, Any]]:
    if result is None:
        return None
    if not isinstance(result, dict):
        return None
    if hook_name == "on_init":
        out: dict[str, Any] = {}
        skills = result.get("skills") or []
        if isinstance(skills, list):
            cleaned_skills = []
            for item in skills[:5]:
                text = ""
                if isinstance(item, dict):
                    text = _clean_text(item.get("text", ""), required=False)
                elif isinstance(item, str):
                    text = _clean_text(item, required=False)
                if text:
                    cleaned_skills.append({"id": "code_hook_skill", "text": text, "trigger": "code_hook"})
            if cleaned_skills:
                out["skills"] = cleaned_skills
        tool_hint = _clean_text(result.get("tool_hint", ""), required=False)
        if tool_hint:
            out["tool_hint"] = tool_hint
        return out or None
    if hook_name == "make_pre_hint":
        message = _clean_text(result.get("message", ""), required=False)
        return {"message": message} if message else None

    kind = str(result.get("kind") or "").strip()
    if kind not in ALLOWED_EFFECT_KINDS.get(hook_name, set()):
        return None
    out = {"kind": kind}
    message = _clean_text(result.get("message", ""), required=False)
    if message:
        out["message"] = message
    if kind in {"force_action", "rewrite_action"}:
        action = _clean_action(result.get("action", ""), ctx)
        if not action:
            return {"kind": "inject_hint", "message": message} if hook_name == "on_post_step" and message else None
        out["action"] = action
    if kind in {"block_and_prompt", "inject_hint"} and not message:
        return None
    return out


def _clean_text(value: Any, *, required: bool = True) -> str:
    if not isinstance(value, str):
        return "" if not required else ""
    text = " ".join(value.strip().split())
    if not text:
        return ""
    return text[:MAX_RETURN_TEXT_CHARS]


def _clean_action(value: Any, ctx: dict[str, Any]) -> str:
    if not isinstance(value, str):
        return ""
    action = " ".join(value.strip().lower().split())
    if not action:
        return ""
    admissible = {" ".join(str(x).strip().lower().split()) for x in ctx.get("admissible", []) or []}
    if ALFWORLD_INSTANCE_ACTION_RE.search(action) and action not in admissible:
        return ""
    return action[:MAX_RETURN_TEXT_CHARS]
