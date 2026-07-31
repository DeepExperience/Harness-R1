#!/usr/bin/env python3
"""Run Harness-R1 batch rollout followed by per-batch Agent Debugger analysis.

This is intentionally an orchestration layer. It does not edit the harness.
Each batch gets its own AgentBench task name, worker port, output directory,
ADB records file, ADB results file, and batch overview.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from harness_r1_patch import compile_patch_to_task_definition, load_patch


BENCHES = {"webshop", "alfworld", "dbbench"}
SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[4]
LOCAL_JAVA_HOME = SCRIPT_REPO_ROOT / ".local/jdk"
DEFAULT_JAVA_HOME = Path("/usr/lib/jvm/java-11-openjdk-amd64")
DEFAULT_QUESTION = (
    "This is a no-harness baseline trajectory from Harness-R1 experiments. "
    "Analyze why it did not reach full reward. Focus only on observable failure "
    "modes, causal evidence from the trace, and uncertainty. Do not propose "
    "harness changes, guard rules, tool hints, code edits, or action schemas. "
    "Do not leak task answers; avoid product ids, exact product titles, and "
    "specific correct click values unless they are necessary to describe the "
    "agent's observed action. "
    "Return concise sections: failure summary, key evidence from the trace, "
    "likely causal factors, and uncertainty/generalization notes."
)
DEFAULT_REACT_PROMPT = """Use the ReAct procedure for every environment step.

Thought: reason step by step from the task, interaction history, latest observation, available actions, and tool definitions. Keep this reasoning concise and specific to the immediate decision.
Action: after the thought, issue exactly one valid structured tool call from the provided tools. Put the action in message.tool_calls rather than writing an action as plain text. Never invent an unavailable action or call multiple tools in one step.
Observation: treat the returned tool result as the next observation, then repeat Thought followed by one Action until the task is complete. When a final-answer or finish tool is available, use that tool rather than returning the answer only as plain text.
A direct tool call without preceding reasoning is not ReAct; always produce the Thought before the structured Action.
"""


def message_text(message: dict[str, Any]) -> str:
    content = str(message.get("content") or "")
    if content.strip():
        return content
    return str(message.get("reasoning_content") or "")


def resolve_java_home() -> Path | None:
    candidates = [
        os.environ.get("HARNESS_R1_JAVA_HOME"),
        os.environ.get("JAVA_HOME"),
        str(LOCAL_JAVA_HOME),
        str(DEFAULT_JAVA_HOME),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        java_home = Path(candidate)
        if (java_home / "bin/javac").exists():
            return java_home
    return None


@dataclass(frozen=True)
class BatchSpec:
    bench: str
    batch_id: int
    start: int
    end: int
    task_name: str
    worker_port: int
    output_dir: Path
    config_dir: Path
    task_ids: list[int] | None = None

    @property
    def size(self) -> int:
        if self.task_ids is not None:
            return len(self.task_ids)
        return max(0, self.end - self.start)

    @property
    def task_config(self) -> Path:
        return self.config_dir / f"{self.task_name}.task.json"

    @property
    def assignment_config(self) -> Path:
        return self.config_dir / f"{self.task_name}.assignment.json"

    @property
    def agent_config(self) -> Path:
        return self.config_dir / f"{self.task_name}.agent.json"

    @property
    def rollout_dir(self) -> Path:
        return self.output_dir / "rollout"

    @property
    def debug_dir(self) -> Path:
        return self.output_dir / "debug"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"


@dataclass
class ControllerHandle:
    process: subprocess.Popen[str] | None
    url: str


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[4]


def default_agentbench_dir(repo_root: Path) -> Path:
    external = repo_root / "external/harness_evolution/life-harness/AgentBench"
    if external.exists():
        return external
    return Path(__file__).resolve().parents[1]


def clean_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    java_home = resolve_java_home()
    if java_home is not None:
        env["JAVA_HOME"] = str(java_home)
        env["HARNESS_R1_JAVA_HOME"] = str(java_home)
        java_bin = str(java_home / "bin")
        path = env.get("PATH", "")
        if java_bin not in path.split(":"):
            env["PATH"] = f"{java_bin}:{path}" if path else java_bin
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def agentbench_env(args: argparse.Namespace, extra: dict[str, str] | None = None) -> dict[str, str]:
    pythonpath_parts = [
        *venv_site_packages_for_python(args.agentbench_python),
        str(args.agentbench_dir),
        str(args.agentbench_dir / "src"),
    ]
    venv_lib = args.agentbench_dir / ".venv/lib"
    if venv_lib.exists():
        pythonpath_parts.extend(str(path) for path in sorted(venv_lib.glob("python*/site-packages")))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    payload = {"PYTHONPATH": ":".join(pythonpath_parts)}
    if extra:
        payload.update(extra)
    return clean_env(payload)


def site_packages_for(venv: Path) -> list[str]:
    lib_dir = venv / "lib"
    if not lib_dir.exists():
        return []
    return [str(path) for path in sorted(lib_dir.glob("python*/site-packages"))]


def venv_site_packages_for_python(python: Path | None) -> list[str]:
    if python is None:
        return []
    # Normal virtualenv layout: <venv>/bin/python.
    return site_packages_for(python.parent.parent)


def local_setup_source_paths(args: argparse.Namespace, source_name: str) -> list[str]:
    candidates = [
        args.agentbench_dir / ".local_setup" / source_name,
        SCRIPT_REPO_ROOT
        / "external"
        / "harness_evolution"
        / "life-harness"
        / "AgentBench"
        / ".local_setup"
        / source_name,
    ]
    seen: set[Path] = set()
    paths: list[str] = []
    for candidate in candidates:
        candidate = candidate.absolute()
        if candidate in seen or not candidate.exists():
            continue
        seen.add(candidate)
        paths.append(str(candidate))
    return paths


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def append_jsonl(path: Path, rows: Iterable[Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def http_json(url: str, timeout: float = 2.0) -> Any | None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None


def wait_for_controller(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/api/list_workers"
    while time.time() < deadline:
        if http_json(url, timeout=2.0) is not None:
            return True
        time.sleep(1)
    return False


def wait_for_worker(controller_port: int, task_name: str, timeout: float) -> bool:
    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{controller_port}/api/list_workers"
    while time.time() < deadline:
        payload = http_json(url, timeout=3.0)
        if isinstance(payload, dict):
            workers = (payload.get(task_name) or {}).get("workers") or {}
            if any(item.get("status") == "ALIVE" for item in workers.values()):
                return True
        time.sleep(1)
    return False


def terminate_process(proc: subprocess.Popen[str] | None, timeout: float = 10.0) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=timeout)


def start_controller(args: argparse.Namespace, run_root: Path) -> ControllerHandle:
    url = f"http://127.0.0.1:{args.controller_port}/api"
    if wait_for_controller(args.controller_port, timeout=2):
        return ControllerHandle(process=None, url=url)
    script = args.agentbench_dir / "scripts/local_agentrl_controller.py"
    if not script.exists():
        raise FileNotFoundError(f"local controller script not found: {script}")
    log_path = run_root / "controller.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [str(args.agentbench_python), str(script), "--port", str(args.controller_port)],
        cwd=args.agentbench_dir,
        env=agentbench_env(args),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    if not wait_for_controller(args.controller_port, timeout=args.startup_timeout):
        terminate_process(proc)
        raise RuntimeError(f"controller did not start on port {args.controller_port}")
    return ControllerHandle(process=proc, url=url)


def webshop_task_definition(args: argparse.Namespace, spec: BatchSpec) -> dict[str, Any]:
    return {
        spec.task_name: {
            "module": "src.server.tasks.webshop.WebShop",
            "parameters": {
                "name": spec.task_name,
                "concurrency": args.rollout_concurrency,
                "round": args.webshop_rounds,
                "start": spec.start,
                "end": spec.end,
                "sample_size": spec.size,
                **({"task_ids": spec.task_ids} if spec.task_ids is not None else {}),
                "goal_seed": args.webshop_goal_seed,
                "enabled": args.harness_enabled,
                "h2": args.h2,
                "h3": args.h3,
                "h4": args.h4,
                "h5": args.h5,
                "h5_top_k": args.h5_top_k,
                "h5_score_threshold": args.h5_score_threshold,
                "system_prompt_prefix": args.rollout_system_prefix,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "search_action",
                            "description": "Use search functionality with specified keywords.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "keywords": {
                                        "type": "string",
                                        "description": "The keywords to use in the search function.",
                                    }
                                },
                                "required": ["keywords"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "click_action",
                            "description": "Click a button or link with a specified value.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "value": {
                                        "type": "string",
                                        "description": "The value to click from the list of available actions.",
                                    }
                                },
                                "required": ["value"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ],
            },
        }
    }


def alfworld_task_definition(args: argparse.Namespace, spec: BatchSpec) -> dict[str, Any]:
    data_path = args.agentbench_dir / "data/alfworld"
    return {
        spec.task_name: {
            "module": "src.server.tasks.alfworld.ALFWorld",
            "parameters": {
                "name": spec.task_name,
                "concurrency": args.rollout_concurrency,
                "data_path": str(data_path),
                "config_path": str(
                    args.agentbench_dir
                    / "src/server/tasks/alfworld/configs/base_config.yaml"
                ),
                "prompts_path": str(
                    args.agentbench_dir
                    / "src/server/tasks/alfworld/prompts/alfworld_multiturn_plan_first.json"
                ),
                "split": args.alfworld_split,
                "max_step": args.alfworld_max_step,
                "start": spec.start,
                "end": spec.end,
                "sample_size": spec.size,
                **({"task_ids": spec.task_ids} if spec.task_ids is not None else {}),
                "enabled": args.harness_enabled,
                "h2": args.h2,
                "h3": args.h3,
                "h4": args.h4,
                "h5": args.h5,
                "h5_top_k": args.h5_top_k,
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "take_action",
                            "description": "Take an action.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "action": {
                                        "type": "string",
                                        "description": "The action you would like to take",
                                    }
                                },
                                "required": ["action"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            },
        }
    }


def dbbench_task_definition(args: argparse.Namespace, spec: BatchSpec) -> dict[str, Any]:
    if args.dbbench_env_driver == "manual":
        env_options = {"urls": {"mysql": args.dbbench_manual_mysql_host}}
    else:
        env_options = {
            "network_name": args.docker_network_name,
            "state_driver": "local",
        }
    return {
        spec.task_name: {
            "module": "src.server.tasks.dbbench.DBBenchTask",
            "parameters": {
                "name": spec.task_name,
                "concurrency": args.rollout_concurrency,
                "max_round": args.dbbench_max_round,
                "start": spec.start,
                "end": spec.end,
                **({"task_ids": spec.task_ids} if spec.task_ids is not None else {}),
                "sample_size": spec.size,
                "data_file": str(args.dbbench_data_file),
                "env_driver": args.dbbench_env_driver,
                "env_options": env_options,
                "harness": {
                    "enabled": args.harness_enabled,
                    "h2": args.h2,
                    "h3": args.h3,
                    "h4": args.h4,
                    "h5": args.h5,
                    "h5_top_k": args.h5_top_k,
                },
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "execute_sql",
                            "description": "Executes a given SQL statement on the database and returns the result.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The SQL query to be executed.",
                                    }
                                },
                                "required": ["query"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "commit_final_answer",
                            "description": "Commits the final answer after all operations are completed.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "answers": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "The list of final answers to commit.",
                                    }
                                },
                                "required": ["answers"],
                                "additionalProperties": False,
                            },
                        },
                    },
                ],
            },
        }
    }


def write_batch_configs(args: argparse.Namespace, spec: BatchSpec) -> None:
    if spec.bench == "webshop":
        task_def = webshop_task_definition(args, spec)
    elif spec.bench == "alfworld":
        task_def = alfworld_task_definition(args, spec)
    elif spec.bench == "dbbench":
        task_def = dbbench_task_definition(args, spec)
    else:
        raise ValueError(f"unsupported bench: {spec.bench}")
    if args.harness_patch_obj:
        task_def = compile_patch_to_task_definition(
            task_definition=task_def,
            patch=args.harness_patch_obj,
            bench=spec.bench,
        )
    agent_def = {
        args.agent_name: {
            "import": "../../../agents/openai-chat.yaml",
            "parameters": {
                "name": args.agent_name,
                "url": args.rollout_base_url.rstrip("/") + "/chat/completions",
                "timeout": args.rollout_http_timeout,
                "headers": {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {args.rollout_api_key}",
                },
                "proxies": {"http": None, "https": None},
                "body": {
                    "model": args.rollout_model,
                    "max_tokens": args.rollout_max_tokens,
                    "temperature": args.rollout_temperature,
                },
            },
        }
    }
    if args.rollout_react:
        if args.rollout_react_mode in {"plan_then_act", "plan_then_act_v2"}:
            agent_def[args.agent_name]["module"] = (
                "src.client.agents.PlanActHTTPAgent"
            )
            agent_def[args.agent_name]["parameters"]["plan_max_tokens"] = (
                args.rollout_plan_max_tokens
            )
            agent_def[args.agent_name]["parameters"]["action_context_mode"] = (
                "observation_last_v2"
                if args.rollout_react_mode == "plan_then_act_v2"
                else "append_plan_v1"
            )
        else:
            agent_def[args.agent_name]["module"] = (
                "src.client.agents.ReActHTTPAgent"
            )
            agent_def[args.agent_name]["parameters"]["react_prompt"] = (
                args.rollout_react_prompt
            )
    elif args.rollout_self_refine:
        agent_def[args.agent_name]["module"] = (
            "src.client.agents.SelfRefineHTTPAgent"
        )
    elif args.rollout_retrieved_memory_file is not None:
        agent_def[args.agent_name]["module"] = (
            "src.client.agents.RetrievedMemoryHTTPAgent"
        )
        agent_def[args.agent_name]["parameters"]["retrieved_memory_path"] = str(
            args.rollout_retrieved_memory_file
        )
    elif args.rollout_episode_reflection_memory_file is not None:
        agent_def[args.agent_name]["module"] = (
            "src.client.agents.EpisodeReflexionHTTPAgent"
        )
        agent_def[args.agent_name]["parameters"][
            "episode_reflection_memory_path"
        ] = str(args.rollout_episode_reflection_memory_file)
    elif args.rollout_reflection:
        agent_def[args.agent_name]["module"] = (
            "src.client.agents.ReflectionHTTPAgent"
        )
    if args.rollout_chat_template_kwargs:
        agent_def[args.agent_name]["parameters"]["body"]["chat_template_kwargs"] = (
            args.rollout_chat_template_kwargs
        )
    if args.rollout_tool_choice:
        agent_def[args.agent_name]["parameters"]["body"]["tool_choice"] = (
            args.rollout_tool_choice
        )
    if args.rollout_disable_parallel_tool_calls:
        agent_def[args.agent_name]["parameters"]["body"]["parallel_tool_calls"] = False
    if args.rollout_single_tool_call_only:
        agent_def[args.agent_name]["parameters"]["single_tool_call_only"] = True
    assignment = {
        "definition": {
            "task": {
                "overwrite": {
                    "module": "src.client.TaskClient",
                    "parameters": {
                        "controller_address": f"http://127.0.0.1:{args.controller_port}/api"
                    },
                },
                "import": f"./{spec.task_config.name}",
            },
            "agent": {
                "import": [
                    f"./{spec.agent_config.name}",
                    "../../../agents/fs_agent.yaml",
                ]
            },
        },
        "concurrency": {
            "task": {spec.task_name: args.rollout_concurrency},
            "agent": {args.agent_name: args.rollout_concurrency},
        },
        "assignments": [{"agent": [args.agent_name], "task": [spec.task_name]}],
        "output": str(spec.rollout_dir),
        "trials": 1,
    }
    dump_json(spec.task_config, task_def)
    dump_json(spec.agent_config, agent_def)
    dump_json(spec.assignment_config, assignment)


def worker_python(args: argparse.Namespace, bench: str) -> Path:
    if bench == "webshop":
        if args.webshop_worker_python is not None:
            return args.webshop_worker_python
        return args.agentbench_dir / ".venvs/webshop-worker/bin/python"
    if bench == "alfworld":
        if args.alfworld_worker_python is not None:
            return args.alfworld_worker_python
        return args.agentbench_dir / ".venvs/alfworld-worker-clean/bin/python"
    if args.dbbench_worker_python is not None:
        return args.dbbench_worker_python
    if args.agentbench_python is not None:
        return args.agentbench_python
    return args.agentbench_dir / ".venv/bin/python"


def dbbench_worker_env(args: argparse.Namespace) -> dict[str, str]:
    pythonpath_parts = [
        *venv_site_packages_for_python(args.dbbench_worker_python),
        *site_packages_for(args.agentbench_dir / ".venv"),
        str(args.agentbench_dir / "src"),
        str(args.agentbench_dir),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath_parts.append(existing)
    return clean_env({"PYTHONPATH": ":".join(pythonpath_parts)})


def worker_env(args: argparse.Namespace, bench: str) -> dict[str, str]:
    if bench == "dbbench":
        return dbbench_worker_env(args)
    if bench == "webshop":
        pythonpath_parts = [
            *local_setup_source_paths(args, "webshop_src"),
            *venv_site_packages_for_python(args.webshop_worker_python),
            *site_packages_for(args.agentbench_dir / ".venvs/webshop-worker"),
            *site_packages_for(args.agentbench_dir / ".venv"),
            str(args.agentbench_dir / "src"),
            str(args.agentbench_dir),
        ]
        return clean_env(
            {
                "PYTHONPATH": ":".join(pythonpath_parts)
            }
    )
    pythonpath_parts = [
        *local_setup_source_paths(args, "alfworld_src"),
        *venv_site_packages_for_python(args.alfworld_worker_python),
        *site_packages_for(args.agentbench_dir / ".venvs/alfworld-worker-clean"),
        *site_packages_for(args.agentbench_dir / ".venvs/alfworld-worker"),
        *site_packages_for(args.agentbench_dir / ".venv"),
        str(args.agentbench_dir / "src"),
        str(args.agentbench_dir),
    ]
    return clean_env(
        {
            "ALFWORLD_DATA": str(args.agentbench_dir / "data/alfworld"),
            "PYTHONPATH": ":".join(pythonpath_parts),
        }
    )


def run_rollout(args: argparse.Namespace, spec: BatchSpec) -> dict[str, Any]:
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    worker_py = worker_python(args, spec.bench)
    if not worker_py.exists():
        raise FileNotFoundError(f"worker python not found: {worker_py}")
    worker_log = (spec.log_dir / "worker.log").open("w", encoding="utf-8")
    assigner_log = (spec.log_dir / "assigner.log").open("w", encoding="utf-8")
    worker_entry = ["-m", "agentrl.worker"]
    if spec.bench == "webshop":
        worker_entry = [
            str(SCRIPT_REPO_ROOT / "code/life-harness/AgentBench/scripts/webshop_agentrl_worker.py")
        ]
    worker_cmd = [
        str(worker_py),
        *worker_entry,
        "-c",
        str(spec.task_config),
        "--controller",
        f"http://127.0.0.1:{args.controller_port}/api",
        "--self",
        f"http://127.0.0.1:{spec.worker_port}/api",
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.worker_port),
        spec.task_name,
    ]
    worker = subprocess.Popen(
        worker_cmd,
        cwd=args.agentbench_dir,
        env=worker_env(args, spec.bench),
        stdout=worker_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    result: dict[str, Any] = {}
    try:
        if not wait_for_worker(args.controller_port, spec.task_name, args.startup_timeout):
            raise RuntimeError(f"worker did not register for {spec.task_name}")
        assigner_cmd = [
            str(args.agentbench_python),
            "-m",
            "src.assigner",
            "-c",
            str(spec.assignment_config),
        ]
        if args.auto_retry:
            assigner_cmd.insert(-2, "-r")
        assigner = subprocess.Popen(
            assigner_cmd,
            cwd=args.agentbench_dir,
            env=agentbench_env(args),
            stdout=assigner_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            assigner_rc = assigner.wait(timeout=args.rollout_timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            terminate_process(assigner)
            assigner_rc = assigner.returncode
        result = {
            "rollout_status": "timeout" if timed_out else "finished",
            "assigner_returncode": assigner_rc,
        }
        return result
    finally:
        terminate_process(worker)
        result["worker_returncode"] = worker.returncode
        worker_log.close()
        assigner_log.close()


def reward_of(row: dict[str, Any]) -> float:
    result = ((row.get("output") or {}).get("result") or {})
    if isinstance(result, dict):
        reward = result.get("reward")
        if isinstance(reward, (int, float)):
            return float(reward)
        for key in ("metric", "metrics"):
            metric = result.get(key)
            if isinstance(metric, dict) and isinstance(metric.get("score"), (int, float)):
                return float(metric["score"])
    return 0.0


def status_of(row: dict[str, Any]) -> str:
    return str((row.get("output") or {}).get("status") or row.get("status") or "unknown")


def messages_of(row: dict[str, Any]) -> list[dict[str, Any]]:
    result = ((row.get("output") or {}).get("result") or {})
    messages = result.get("openai_messages") if isinstance(result, dict) else None
    if not isinstance(messages, list) or not messages:
        messages = (row.get("output") or {}).get("history") or []
    cleaned: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        item = dict(message)
        for key in ("tool_calls", "tool_call_id", "name"):
            if item.get(key) is None:
                item.pop(key, None)
        cleaned.append(item)
    return cleaned


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def result_dir(args: argparse.Namespace, spec: BatchSpec) -> Path:
    return spec.rollout_dir / args.agent_name / spec.task_name


def build_adb_records(args: argparse.Namespace, spec: BatchSpec) -> dict[str, Any]:
    out_dir = result_dir(args, spec)
    runs = read_jsonl(out_dir / "runs.jsonl")
    errors = read_jsonl(out_dir / "error.jsonl")
    selected: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []

    def add_record(row: dict[str, Any], kind: str) -> None:
        idx = row.get("index")
        reward = reward_of(row)
        if kind == "non_full_reward" and reward >= args.debug_reward_threshold:
            return
        messages = messages_of(row)
        if not messages:
            return
        global_index = idx
        if spec.bench == "alfworld" and isinstance(idx, int):
            global_index = spec.start + idx
        meta = {
            "benchmark": spec.bench,
            "batch_id": spec.batch_id,
            "batch_start": spec.start,
            "batch_end": spec.end,
            "index": idx,
            "global_index": global_index,
            "kind": kind,
            "status": status_of(row),
            "reward": reward,
            "error": row.get("error"),
            "info": row.get("info"),
        }
        trace_id = (
            f"{spec.bench}-b{spec.batch_id:03d}-"
            f"{kind}-{global_index if global_index is not None else len(records)}"
        )
        records.append(
            {
                "queries": [args.debug_question],
                "traces": {
                    "trace_id": trace_id,
                    "messages": messages
                    + [
                        {
                            "role": "user",
                            "content": "Evaluation metadata: "
                            + json.dumps(meta, ensure_ascii=False),
                        }
                    ],
                },
            }
        )
        selected.append(meta)

    for row in runs:
        add_record(row, "non_full_reward")

    last_error_by_index: dict[str, dict[str, Any]] = {}
    for row in errors:
        last_error_by_index[str(row.get("index"))] = row
    for row in last_error_by_index.values():
        add_record(row, "worker_error")

    records_path = spec.debug_dir / "records.jsonl"
    count = append_jsonl(records_path, records)
    manifest = {
        "created_at": datetime.now().isoformat(),
        "benchmark": spec.bench,
        "batch_id": spec.batch_id,
        "task_name": spec.task_name,
        "task_range": [spec.start, spec.end],
        "rollout_result_dir": str(out_dir),
        "runs_count": len(runs),
        "errors_count": len(errors),
        "records_count": count,
        "selected": selected,
    }
    dump_json(spec.debug_dir / "manifest.json", manifest)
    return manifest


def run_adb(args: argparse.Namespace, spec: BatchSpec, records_count: int) -> dict[str, Any]:
    spec.debug_dir.mkdir(parents=True, exist_ok=True)
    if records_count == 0:
        return {"adb_status": "skipped", "adb_returncode": None}
    cmd = [
        str(args.adb_bin),
        "ask",
        "-f",
        str(spec.debug_dir / "records.jsonl"),
        "-j",
        str(args.adb_parallelism),
        "--format",
        "json",
    ]
    env = clean_env(
        {
            "HOME": str(args.adb_home),
            "AHE_HOME": str(args.ahe_home),
            "LLM_MODEL": args.debug_model,
            "LLM_BASE_URL": args.debug_base_url,
            "LLM_API_KEY": args.debug_api_key,
            "LLM_MAX_TOKENS": str(args.debug_max_tokens),
        }
    )
    stdout_path = spec.debug_dir / "results.jsonl"
    stderr_path = spec.debug_dir / "stderr.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open(
        "w", encoding="utf-8"
    ) as err:
        proc = subprocess.run(
            cmd,
            cwd=args.repo_root,
            env=env,
            stdout=out,
            stderr=err,
            text=True,
            timeout=args.adb_timeout,
            check=False,
        )
    return {"adb_status": "finished", "adb_returncode": proc.returncode}


def load_adb_results(spec: BatchSpec) -> list[dict[str, Any]]:
    return read_jsonl(spec.debug_dir / "results.jsonl")


def compact_response(text: str, limit: int = 2400) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def write_heuristic_overview(spec: BatchSpec, manifest: dict[str, Any]) -> Path:
    rows = load_adb_results(spec)
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") != "success"]
    selected = manifest.get("selected") or []
    avg_reward = 0.0
    if selected:
        avg_reward = sum(float(item.get("reward") or 0.0) for item in selected) / len(selected)

    lines = [
        f"# Harness-R1 Batch Debug Overview: {spec.bench} batch {spec.batch_id}",
        "",
        "## Batch",
        "",
        f"- Task range: `{spec.start}:{spec.end}`",
        f"- Task name: `{spec.task_name}`",
        f"- Rollout result dir: `{manifest.get('rollout_result_dir')}`",
        f"- Rollout rows: `{manifest.get('runs_count')}` runs, `{manifest.get('errors_count')}` errors",
        f"- Debug records: `{manifest.get('records_count')}`",
        f"- ADB parsed: `{len(successes)}` success, `{len(failures)}` failed",
        f"- Selected average reward: `{avg_reward:.4f}`",
        "",
        "## Selected Cases",
        "",
    ]
    for item in selected:
        lines.append(
            "- "
            f"{item.get('kind')} index={item.get('index')} global={item.get('global_index')} "
            f"status={item.get('status')} reward={item.get('reward')}"
        )

    lines.extend(["", "## Per-Trace Debugger Analyses", ""])
    if not rows:
        lines.append("No non-full-reward or error traces were selected for debugging.")
    for i, row in enumerate(rows, 1):
        trace_id = row.get("trace_id") or f"failed-{i}"
        lines.append(f"### {i}. {trace_id}")
        lines.append("")
        if row.get("status") == "success":
            lines.append(compact_response(row.get("response") or ""))
        else:
            lines.append("ADB record failed to parse cleanly:")
            lines.append("")
            lines.append("```text")
            lines.append(compact_response(row.get("error") or "", limit=1000))
            lines.append("```")
        lines.append("")

    out = spec.debug_dir / "overview.md"
    out.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return out


def call_overview_llm(args: argparse.Namespace, spec: BatchSpec, overview_path: Path) -> Path:
    if args.overview_mode != "llm":
        return overview_path
    content = overview_path.read_text(encoding="utf-8")
    prompt = (
        "Aggregate the following per-trace debugger analyses into one batch-level "
        "overview for a future harness engineer. Identify recurring failure modes, "
        "candidate typed harness actions, evidence strength, and generalization "
        "risks. Do not propose task-answer leakage.\n\n"
        + content[: args.overview_context_chars]
    )
    payload = {
        "model": args.debug_model,
        "messages": [
            {"role": "system", "content": "You summarize agent-debugger analyses for harness engineering."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": args.overview_max_tokens,
    }
    req = urllib.request.Request(
        args.debug_base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {args.debug_api_key}"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=args.overview_timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = message_text(data["choices"][0]["message"])
    out = spec.debug_dir / "overview.llm.md"
    out.write_text(text.strip() + "\n", encoding="utf-8")
    return out


def run_batch(args: argparse.Namespace, spec: BatchSpec) -> dict[str, Any]:
    started = time.time()
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    write_batch_configs(args, spec)
    status: dict[str, Any] = {
        "benchmark": spec.bench,
        "batch_id": spec.batch_id,
        "range": [spec.start, spec.end],
        "task_ids": spec.task_ids,
        "task_name": spec.task_name,
        "worker_port": spec.worker_port,
        "output_dir": str(spec.output_dir),
    }
    if args.dry_run:
        status.update({"status": "dry_run"})
        dump_json(spec.output_dir / "status.json", status)
        return status
    try:
        if spec.bench == "webshop" and resolve_java_home() is None:
            raise RuntimeError(
                "WebShop requires a JDK with bin/javac. Set HARNESS_R1_JAVA_HOME or install .local/jdk."
            )
        if not args.skip_rollout:
            rollout_status = run_rollout(args, spec)
            status.update(rollout_status)
            if rollout_status.get("rollout_status") == "timeout":
                raise RuntimeError("rollout timed out")
            if rollout_status.get("assigner_returncode") not in (0, None):
                raise RuntimeError(
                    f"assigner failed with return code {rollout_status.get('assigner_returncode')}"
                )
        manifest = build_adb_records(args, spec)
        status["debug_records"] = manifest["records_count"]
        if not args.skip_adb:
            status.update(run_adb(args, spec, manifest["records_count"]))
        overview = write_heuristic_overview(spec, manifest)
        overview = call_overview_llm(args, spec, overview)
        status.update(
            {
                "status": "ok",
                "overview": str(overview),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": repr(exc),
                "elapsed_sec": round(time.time() - started, 3),
            }
        )
    dump_json(spec.output_dir / "status.json", status)
    return status


def build_specs(args: argparse.Namespace, run_root: Path) -> list[BatchSpec]:
    specs: list[BatchSpec] = []
    ordinal = 0
    for bench in args.bench:
        if args.task_ids is not None:
            if bench not in {"alfworld", "webshop", "dbbench"}:
                raise SystemExit(
                    "--task-ids-file is currently supported only for "
                    "--bench alfworld/webshop/dbbench"
                )
            batch_ids = range((len(args.task_ids) + args.batch_size - 1) // args.batch_size)
        else:
            batch_ids = args.batch_ids if args.batch_ids is not None else range(args.num_batches)
        for batch_id in batch_ids:
            task_ids = None
            if args.task_ids is not None:
                offset = batch_id * args.batch_size
                task_ids = args.task_ids[offset : offset + args.batch_size]
                if not task_ids:
                    continue
                start = min(task_ids)
                end = max(task_ids) + 1
            else:
                start = args.start + batch_id * args.batch_size
                end = start + args.batch_size
            task_name = f"{bench}-noharness-b{batch_id:03d}"
            specs.append(
                BatchSpec(
                    bench=bench,
                    batch_id=batch_id,
                    start=start,
                    end=end,
                    task_name=task_name,
                    worker_port=args.worker_port_base + ordinal,
                    output_dir=run_root / bench / f"batch_{batch_id:03d}",
                    config_dir=args.agentbench_dir
                    / "configs/generated/harness_r1"
                    / args.run_id,
                    task_ids=task_ids,
                )
            )
            ordinal += 1
    return specs


def parse_args() -> argparse.Namespace:
    repo_root = repo_root_from_script()
    agentbench_dir = default_agentbench_dir(repo_root)
    parser = argparse.ArgumentParser(
        description="Parallel batch rollout + Agent Debugger analysis for Harness-R1."
    )
    parser.add_argument("--bench", nargs="+", choices=sorted(BENCHES), default=["webshop"])
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument(
        "--batch-ids",
        type=str,
        default=None,
        help="Optional comma-separated original batch ids to run. Preserves batch ids and uses start + batch_id * batch_size.",
    )
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        default=None,
        help="Optional file with explicit task indices to run, one integer per line or a JSON list. Supported for ALFWorld/WebShop/DBBench.",
    )
    parser.add_argument("--max-parallel-batches", type=int, default=1)
    parser.add_argument("--rollout-concurrency", type=int, default=2)
    parser.add_argument("--auto-retry", action="store_true")
    parser.add_argument("--rollout-timeout", type=int, default=1800)
    parser.add_argument("--startup-timeout", type=int, default=120)
    parser.add_argument("--controller-port", type=int, default=5020)
    parser.add_argument("--worker-port-base", type=int, default=5100)
    parser.add_argument("--agent-name", default="qwen35-4b-rollout")
    parser.add_argument("--rollout-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--rollout-model", default="Qwen3.5-4B")
    parser.add_argument("--rollout-api-key", default="EMPTY")
    parser.add_argument("--rollout-max-tokens", type=int, default=4096)
    parser.add_argument("--rollout-temperature", type=float, default=0.0)
    parser.add_argument("--rollout-http-timeout", type=int, default=120)
    parser.add_argument("--rollout-tool-choice", choices=["auto", "required", "none"])
    parser.add_argument("--rollout-chat-template-kwargs", type=json.loads, default={})
    parser.add_argument(
        "--rollout-disable-parallel-tool-calls",
        action="store_true",
        help="Send parallel_tool_calls=false for model templates that accept one tool call per turn.",
    )
    parser.add_argument(
        "--rollout-single-tool-call-only",
        action="store_true",
        help=(
            "Keep only the first returned tool call before storing assistant history. "
            "Use only for single-action environments and model templates that reject "
            "multi-call assistant turns."
        ),
    )
    parser.add_argument(
        "--rollout-react",
        action="store_true",
        help=(
            "Use the dedicated direct-agent ReAct baseline: visible concise Thought "
            "followed by exactly one structured tool Action per environment step."
        ),
    )
    parser.add_argument(
        "--rollout-react-prompt",
        default=DEFAULT_REACT_PROMPT,
        help="Exact system prompt appended by the dedicated ReAct agent.",
    )
    parser.add_argument(
        "--rollout-react-mode",
        choices=[
            "visible_thought_single_call",
            "plan_then_act",
            "plan_then_act_v2",
        ],
        default="visible_thought_single_call",
        help=(
            "Keep the historical one-call visible-Thought protocol, or use two "
            "calls that force a visible plan before the structured action. The "
            "v2 mode injects the plan once while keeping the current observation "
            "as the final action context."
        ),
    )
    parser.add_argument(
        "--rollout-plan-max-tokens",
        type=int,
        default=384,
        help="Maximum visible planning tokens for two-call plan_then_act ReAct.",
    )
    parser.add_argument(
        "--rollout-self-refine",
        action="store_true",
        help=(
            "Use three-stage action Self-Refine: draft, textual feedback, then "
            "conditional refinement with an action-admissibility gate."
        ),
    )
    parser.add_argument(
        "--rollout-retrieved-memory-file",
        type=Path,
        default=None,
        help=(
            "Inject a frozen train-only memory bank using the dedicated "
            "RetrievedMemoryHTTPAgent baseline."
        ),
    )
    parser.add_argument(
        "--rollout-episode-reflection-memory-file",
        type=Path,
        default=None,
        help=(
            "Run a failed-episode retry with a precomputed Reflexion memory file. "
            "This is the second trial of episode-level Reflexion, not the legacy "
            "per-action --rollout-reflection baseline."
        ),
    )
    parser.add_argument(
        "--rollout-reflection",
        action="store_true",
        help=(
            "Use trajectory Reflection: reflect once on the interaction history "
            "before each environment action, then execute one selected tool call."
        ),
    )
    parser.add_argument("--rollout-system-prefix", default="")
    parser.add_argument("--agentbench-dir", type=Path, default=agentbench_dir)
    parser.add_argument("--agentbench-python", type=Path, default=None)
    parser.add_argument("--webshop-worker-python", type=Path, default=None)
    parser.add_argument("--alfworld-worker-python", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-rollout", action="store_true")
    parser.add_argument("--skip-adb", action="store_true")
    parser.add_argument(
        "--allow-failed-batches",
        action="store_true",
        help=(
            "Return exit code 0 even if some batches failed. The summary still "
            "records failed batches; this is intended for large eval sweeps where "
            "failed batches are backfilled separately."
        ),
    )
    parser.add_argument("--debug-reward-threshold", type=float, default=1.0)
    parser.add_argument("--debug-question", default=DEFAULT_QUESTION)
    parser.add_argument("--adb-bin", type=Path, default=repo_root / "code/life-harness/.venv-adb/bin/adb")
    parser.add_argument("--adb-home", type=Path, default=repo_root / "code/life-harness/.adb-home")
    parser.add_argument("--ahe-home", type=Path, default=repo_root / "code/agentic-harness-engineering")
    parser.add_argument("--adb-parallelism", type=int, default=10)
    parser.add_argument("--adb-timeout", type=int, default=1800)
    parser.add_argument("--debug-model", default="Qwen3.5-27B")
    parser.add_argument("--debug-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--debug-api-key", default="EMPTY")
    parser.add_argument("--debug-max-tokens", type=int, default=32000)
    parser.add_argument("--overview-mode", choices=["heuristic", "llm"], default="heuristic")
    parser.add_argument("--overview-context-chars", type=int, default=40000)
    parser.add_argument("--overview-max-tokens", type=int, default=4096)
    parser.add_argument("--overview-timeout", type=int, default=300)
    parser.add_argument("--webshop-rounds", type=int, default=20)
    parser.add_argument(
        "--webshop-goal-seed",
        type=int,
        default=233,
        help=(
            "Seed product-price and goal generation before WebShop server "
            "construction. Baseline and patched reruns must use the same value."
        ),
    )
    parser.add_argument("--alfworld-split", default="new_std")
    parser.add_argument("--alfworld-max-step", type=int, default=50)
    parser.add_argument(
        "--dbbench-worker-python",
        type=Path,
        default=None,
        help="Python interpreter for DBBench workers. Defaults to --agentbench-python.",
    )
    parser.add_argument(
        "--dbbench-data-file",
        type=Path,
        default=agentbench_dir / "data/dbbench/standard.jsonl",
    )
    parser.add_argument("--dbbench-max-round", type=int, default=20)
    parser.add_argument("--dbbench-env-driver", choices=["docker", "manual"], default="docker")
    parser.add_argument("--dbbench-manual-mysql-host", default="127.0.0.1")
    parser.add_argument("--docker-network-name", default="harness_r1_agentbench")
    parser.add_argument("--harness-enabled", action="store_true")
    parser.add_argument("--h2", action="store_true")
    parser.add_argument("--h3", action="store_true")
    parser.add_argument("--h4", action="store_true")
    parser.add_argument("--h5", action="store_true")
    parser.add_argument("--h5-top-k", type=int, default=1)
    parser.add_argument("--h5-score-threshold", type=float, default=3.0)
    parser.add_argument(
        "--harness-patch",
        type=Path,
        default=None,
        help="Typed Harness-R1 patch JSON generated by harness_r1_edit.py.",
    )
    args = parser.parse_args()
    args.repo_root = repo_root
    args.agentbench_dir = args.agentbench_dir.resolve()
    if args.agentbench_python is None:
        args.agentbench_python = args.agentbench_dir / ".venv/bin/python"
    args.agentbench_python = Path(args.agentbench_python).absolute()
    if args.dbbench_worker_python is None:
        args.dbbench_worker_python = args.agentbench_python
    else:
        args.dbbench_worker_python = Path(args.dbbench_worker_python).expanduser().absolute()
    if not args.dbbench_data_file.is_absolute():
        args.dbbench_data_file = args.agentbench_dir / args.dbbench_data_file
    args.dbbench_data_file = args.dbbench_data_file.expanduser().resolve()
    if args.webshop_worker_python is not None:
        args.webshop_worker_python = args.webshop_worker_python.expanduser().absolute()
    if args.alfworld_worker_python is not None:
        args.alfworld_worker_python = args.alfworld_worker_python.expanduser().absolute()
    if args.rollout_episode_reflection_memory_file is not None:
        args.rollout_episode_reflection_memory_file = (
            args.rollout_episode_reflection_memory_file.expanduser().resolve()
        )
        if not args.rollout_episode_reflection_memory_file.is_file():
            raise SystemExit(
                "episode reflection memory file not found: "
                f"{args.rollout_episode_reflection_memory_file}"
            )
    if args.rollout_retrieved_memory_file is not None:
        args.rollout_retrieved_memory_file = (
            args.rollout_retrieved_memory_file.expanduser().resolve()
        )
        if not args.rollout_retrieved_memory_file.is_file():
            raise SystemExit(
                "retrieved memory file not found: "
                f"{args.rollout_retrieved_memory_file}"
            )
    if args.output_root is None:
        args.output_root = (
            args.agentbench_dir / "outputs/harness_r1_batch_debug" / args.run_id
        )
    args.output_root = args.output_root.resolve()
    if args.batch_ids:
        ids: list[int] = []
        for item in args.batch_ids.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                batch_id = int(item)
            except ValueError as exc:
                raise SystemExit(f"invalid --batch-ids item: {item!r}") from exc
            if batch_id < 0:
                raise SystemExit(f"batch ids must be non-negative: {batch_id}")
            ids.append(batch_id)
        if not ids:
            raise SystemExit("--batch-ids was provided but no ids were parsed")
        args.batch_ids = ids
    else:
        args.batch_ids = None
    args.task_ids = None
    if args.task_ids_file is not None:
        path = args.task_ids_file.expanduser().resolve()
        raw = path.read_text(encoding="utf-8").strip()
        if raw.startswith("["):
            parsed_items = json.loads(raw)
        else:
            parsed_items = [
                item.strip()
                for line in raw.splitlines()
                for item in line.replace(",", " ").split()
                if item.strip()
            ]
        task_ids: list[int] = []
        for item in parsed_items:
            try:
                task_id = int(item)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"invalid --task-ids-file item: {item!r}") from exc
            if task_id < 0:
                raise SystemExit(f"task ids must be non-negative: {task_id}")
            task_ids.append(task_id)
        if not task_ids:
            raise SystemExit("--task-ids-file was provided but no ids were parsed")
        args.task_ids_file = path
        args.task_ids = task_ids
    args.harness_patch_obj = None
    if args.harness_patch is not None:
        args.harness_patch = args.harness_patch.resolve()
        args.harness_patch_obj = load_patch(args.harness_patch)
    if args.batch_size <= 0 or args.num_batches <= 0 or args.max_parallel_batches <= 0:
        raise SystemExit("batch-size, num-batches, and max-parallel-batches must be positive")
    if args.rollout_plan_max_tokens <= 0:
        raise SystemExit("rollout-plan-max-tokens must be positive")
    rollout_methods = sum(
        (
            bool(args.rollout_react),
            bool(args.rollout_self_refine),
            bool(args.rollout_reflection),
            args.rollout_retrieved_memory_file is not None,
            args.rollout_episode_reflection_memory_file is not None,
        )
    )
    if rollout_methods > 1:
        raise SystemExit(
            "ReAct, Self-Refine, retrieved memory, legacy per-action Reflection, "
            "and episode-level Reflexion retry are mutually exclusive"
        )
    return args


def main() -> int:
    args = parse_args()
    run_root = args.output_root
    run_root.mkdir(parents=True, exist_ok=True)
    if not args.agentbench_python.exists():
        raise FileNotFoundError(f"AgentBench python not found: {args.agentbench_python}")
    if not args.dry_run and not args.skip_adb and not args.adb_bin.exists():
        raise FileNotFoundError(f"ADB binary not found: {args.adb_bin}")
    specs = build_specs(args, run_root)
    dump_json(
        run_root / "run_config.json",
        {
            "created_at": datetime.now().isoformat(),
            "agentbench_dir": str(args.agentbench_dir),
            "output_root": str(run_root),
            "controller_port": args.controller_port,
            "max_parallel_batches": args.max_parallel_batches,
            "rollout_react": args.rollout_react,
            "rollout_react_mode": (
                args.rollout_react_mode if args.rollout_react else None
            ),
            "rollout_plan_max_tokens": (
                args.rollout_plan_max_tokens
                if args.rollout_react_mode
                in {"plan_then_act", "plan_then_act_v2"}
                else None
            ),
            "rollout_react_prompt": (
                args.rollout_react_prompt
                if args.rollout_react
                and args.rollout_react_mode == "visible_thought_single_call"
                else None
            ),
            "rollout_self_refine": args.rollout_self_refine,
            "rollout_reflection": args.rollout_reflection,
            "rollout_retrieved_memory_file": (
                str(args.rollout_retrieved_memory_file)
                if args.rollout_retrieved_memory_file
                else None
            ),
            "rollout_episode_reflection_memory_file": (
                str(args.rollout_episode_reflection_memory_file)
                if args.rollout_episode_reflection_memory_file
                else None
            ),
            "webshop_goal_seed": args.webshop_goal_seed,
            "harness_patch": str(args.harness_patch) if args.harness_patch else None,
            "harness_patch_obj": args.harness_patch_obj,
            "specs": [spec.__dict__ | {"output_dir": str(spec.output_dir), "config_dir": str(spec.config_dir)} for spec in specs],
        },
    )
    controller = ControllerHandle(process=None, url=f"http://127.0.0.1:{args.controller_port}/api")
    if not args.dry_run and not args.skip_rollout:
        controller = start_controller(args, run_root)
        print(f"[harness-r1] controller ready: {controller.url}", flush=True)
    print(
        f"[harness-r1] running {len(specs)} batch(es), parallel={args.max_parallel_batches}, output={run_root}",
        flush=True,
    )
    results: list[dict[str, Any]] = []
    interrupted = False
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.max_parallel_batches
        ) as pool:
            future_to_spec = {pool.submit(run_batch, args, spec): spec for spec in specs}
            for future in concurrent.futures.as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "benchmark": spec.bench,
                        "batch_id": spec.batch_id,
                        "status": "failed",
                        "error": repr(exc),
                    }
                results.append(result)
                print(
                    f"[harness-r1] {spec.bench} batch {spec.batch_id} -> {result.get('status')}",
                    flush=True,
                )
    except KeyboardInterrupt:
        interrupted = True
        print("[harness-r1] interrupted; child processes will be terminated by their batch handlers.", flush=True)
    finally:
        if controller.process is not None:
            terminate_process(controller.process)
    dump_json(run_root / "summary.json", {"interrupted": interrupted, "results": results})
    failed = [item for item in results if item.get("status") != "ok" and item.get("status") != "dry_run"]
    if interrupted:
        return 130
    if failed and args.allow_failed_batches:
        print(
            f"[harness-r1] {len(failed)} batch(es) failed but --allow-failed-batches is set",
            flush=True,
        )
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
