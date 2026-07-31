import copy
import enum
import json
from typing import List, Optional

import requests
from tqdm import tqdm

from src.typings import *
from src.utils import *
from .agent import AgentClient


def _slim_outputs_for_worker_aggregate(results: List[TaskOutput]) -> List[TaskOutput]:
    """Worker aggregate only needs compact fields; drop huge traces like openai_messages."""
    slim: List[TaskOutput] = []
    for r in results:
        keep = None
        if isinstance(r.result, dict):
            keep = {
                k: r.result[k]
                for k in ("reward", "result", "metrics")
                if k in r.result
            }
        # Strip trial dimension from compound index before sending to worker.
        idx = r.index[0] if isinstance(r.index, list) and len(r.index) == 2 else r.index
        slim.append(
            TaskOutput(
                status=r.status,
                index=idx,
                result=keep,
                history=[],
            )
        )
    return slim


def _compute_trial_metrics(
    results: List["TaskOutput"],
    is_pass_fn,
    trials: int,
) -> dict:
    groups: dict = {}
    for r in results:
        if isinstance(r.index, list) and len(r.index) == 2:
            base, t = r.index[0], r.index[1]
        else:
            continue
        groups.setdefault(base, []).append((t, r))
    groups = {base: [r for _, r in sorted(entries)] for base, entries in groups.items()}
    n_q = len(groups)
    if n_q == 0:
        return {}
    all_scores = [1.0 if is_pass_fn(r) else 0.0 for g in groups.values() for r in g]
    avg_at_k = sum(all_scores) / len(all_scores) if all_scores else 0
    pass_at_k = sum(1 for g in groups.values() if any(is_pass_fn(r) for r in g)) / n_q
    pass_power = {}
    for j in range(2, trials + 1):
        count = sum(
            1 for g in groups.values()
            if len(g) >= j and all(is_pass_fn(r) for r in g[:j])
        )
        pass_power[f"pass^{j}"] = round(count / n_q, 4)
    return {
        "trials": trials,
        "questions": n_q,
        "avg_at_k": round(avg_at_k, 4),
        "pass_at_k": round(pass_at_k, 4),
        "pass_power": pass_power,
    }


def _local_calculate_overall(task_name: str, results: List[TaskOutput], trials: int = 1) -> dict:
    """Mirror server task aggregate logic when controller/worker HTTP is unavailable."""
    if task_name.startswith("alfworld"):
        def is_pass(x: TaskOutput) -> bool:
            if not x or not isinstance(x.result, dict):
                return False
            if "result" in x.result:
                return int(x.result.get("result", 0) == 1) == 1
            reward = x.result.get("reward", None)
            if reward is not None:
                try:
                    return float(reward) >= 1.0
                except Exception:
                    return False
            metrics = x.result.get("metrics", {})
            score = metrics.get("score", None) if isinstance(metrics, dict) else None
            if score is not None:
                try:
                    return float(score) >= 1.0
                except Exception:
                    return False
            return False

        total = sum(1 for x in results if x)
        passed = sum(1 for x in results if is_pass(x))
        wrong = total - passed

        usages = [
            x.result.get("token_usage", {})
            for x in results
            if x and isinstance(x.result, dict)
        ]
        n_ep = len(usages) or 1
        total_prompt = sum(u.get("prompt_tokens", 0) for u in usages)
        total_completion = sum(u.get("completion_tokens", 0) for u in usages)
        total_tokens = sum(u.get("total_tokens", 0) for u in usages)

        overall = {
            "total": total,
            "pass": passed,
            "wrong": wrong,
            "success_rate": passed / total if total else 0,
        }
        if trials > 1:
            overall.update(_compute_trial_metrics(results, is_pass, trials))
        return {
            "overall": overall,
            "token_usage": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "avg_prompt_tokens_per_episode": round(total_prompt / n_ep),
                "avg_completion_tokens_per_episode": round(total_completion / n_ep),
                "avg_total_tokens_per_episode": round(total_tokens / n_ep),
            },
        }
    if task_name.startswith("webshop"):
        rel = [x.result for x in results if x and isinstance(x.result, dict)]
        rewards = [x.get("reward") for x in rel if x.get("reward") is not None]
        total = len(results)
        completed = sum(1 for x in results if x and x.status == SampleStatus.COMPLETED)
        success_at_1 = sum(1 for r in rewards if float(r) >= 1.0)
        avg = sum(rewards) / len(rewards) if rewards else 0
        usages = [x.get("token_usage", {}) for x in rel]
        n_ep = len(usages) or 1
        total_prompt = sum(u.get("prompt_tokens", 0) for u in usages if u)
        total_completion = sum(u.get("completion_tokens", 0) for u in usages if u)
        total_tokens = sum(u.get("total_tokens", 0) for u in usages if u)

        def _ws_is_pass(x: TaskOutput) -> bool:
            if not x or not isinstance(x.result, dict):
                return False
            r = x.result.get("reward")
            try:
                return r is not None and float(r) >= 1.0
            except Exception:
                return False

        overall = {
            "total": total,
            "completed": completed,
            "completed_rate": completed / total if total else 0,
            "success_at_1": success_at_1,
            "success_at_1_rate": success_at_1 / total if total else 0,
            "average_reward": avg,
        }
        if trials > 1:
            overall.update(_compute_trial_metrics(results, _ws_is_pass, trials))
        return {
            "overall": overall,
            "token_usage": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "avg_prompt_tokens_per_episode": round(total_prompt / n_ep),
                "avg_completion_tokens_per_episode": round(total_completion / n_ep),
                "avg_total_tokens_per_episode": round(total_tokens / n_ep),
            },
        }
    if task_name.startswith("dbbench"):
        rel = [x.result for x in results if x and isinstance(x.result, dict)]
        rewards = [float(x.get("reward", 0)) for x in rel]
        total = len(results)
        passed = sum(1 for r in rewards if r >= 1.0)
        usages = [x.get("token_usage", {}) for x in rel]
        n_ep = len(usages) or 1
        total_prompt = sum(u.get("prompt_tokens", 0) for u in usages if u)
        total_completion = sum(u.get("completion_tokens", 0) for u in usages if u)
        total_tokens = sum(u.get("total_tokens", 0) for u in usages if u)

        def _db_is_pass(x: TaskOutput) -> bool:
            if not x or not isinstance(x.result, dict):
                return False
            try:
                return float(x.result.get("reward", 0)) >= 1.0
            except Exception:
                return False

        overall = {
            "total": total,
            "correct": passed,
            "wrong": total - passed,
            "acc": passed / total if total else 0,
        }
        if trials > 1:
            overall.update(_compute_trial_metrics(results, _db_is_pass, trials))
        return {
            "overall": overall,
            "token_usage": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "avg_prompt_tokens_per_episode": round(total_prompt / n_ep),
                "avg_completion_tokens_per_episode": round(total_completion / n_ep),
                "avg_total_tokens_per_episode": round(total_tokens / n_ep),
            },
        }
    return {
        "note": f"No built-in local aggregate for task {task_name!r}; use worker or extend _local_calculate_overall."
    }


class TaskError(enum.Enum):
    START_FAILED = "START_FAILED"
    INTERACT_FAILED = "INTERACT_FAILED"
    AGENT_FAILED = "AGENT_FAILED"
    NETWORK_ERROR = "NETWORK_ERROR"
    NOT_AVAILABLE = "NOT_AVAILABLE"


class TaskClient:
    def __init__(
        self,
        name: str,
        controller_address: str = "http://localhost:5000/api",
        *_,
        task_ids=None,
        **__,
    ) -> None:
        self.name = name
        self.controller_address = controller_address
        self.task_ids = self._parse_task_ids(task_ids)
        print("TaskClient created: {} ({})".format(name, controller_address))

    @staticmethod
    def _parse_task_ids(task_ids):
        if task_ids is None:
            return None
        if isinstance(task_ids, str):
            raw_items = task_ids.replace(",", " ").split()
        elif isinstance(task_ids, (list, tuple)):
            raw_items = task_ids
        else:
            raw_items = [task_ids]
        parsed = []
        for item in raw_items:
            if item is None or item == "":
                continue
            parsed.append(int(item))
        return parsed or None

    def get_indices(self) -> List[SampleIndex]:
        result = requests.get(
            self.controller_address + "/get_indices", params={"name": self.name}
        )
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        indices = result.json()
        if self.task_ids is None:
            return indices
        # The controller's /get_indices reflects the worker's configured
        # shuffle/sample window.  Explicit task ids are intended to override
        # that window, so do not reject ids just because they are not present in
        # the sampled list.  The worker will validate the actual id on
        # /start_sample.
        return list(self.task_ids)

    def get_concurrency(self) -> int:
        try:
            result = requests.get(
                self.controller_address + "/list_workers"
            )
        except Exception as e:
            print(ColorMessage.yellow(f"Warning task {self.name} cannot connect to controller {e}"))
            return 0
        if result.status_code != 200:
            raise AgentBenchException(result.text, result.status_code, self.name)
        result = result.json()
        if self.name not in result:
            print(ColorMessage.yellow(f"task {self.name} not found in worker list"))
            return 0
        concurrency = 0
        for worker in result[self.name]["workers"].values():
            status = worker.get("status")
            # Controller versions may return worker status as enum value (0) or string ("ALIVE").
            if (
                status == WorkerStatus.ALIVE
                or (isinstance(status, str) and status.upper() == "ALIVE")
                or (isinstance(status, int) and status == int(WorkerStatus.ALIVE))
            ):
                concurrency += worker["capacity"] - worker["current"]
        return concurrency

    def run_sample(self, index: SampleIndex, agent: AgentClient) -> TaskClientOutput:
        # tqdm.write(f"[ASSIGNER][SAMPLE_START] task={self.name} index={index}")

        def _extract_session_id(resp: requests.Response, payload: dict):
            sid = payload.get("session_id")
            if sid is not None:
                return sid
            # Newer controller versions return session id in response headers.
            for key in ("session_id", "Session_id", "Session-Id"):
                if key in resp.headers:
                    return int(resp.headers[key])
            return None

        def _normalize_output(payload: dict) -> TaskOutput:
            # Legacy format: {"output": {...}}
            if "output" in payload and isinstance(payload["output"], dict):
                return TaskOutput.parse_obj(payload["output"])
            # Some controller/worker versions return TaskOutput directly.
            # If history is already present, preserve it instead of re-parsing
            # as OpenAI-style protocol payload.
            if "history" in payload:
                return TaskOutput.parse_obj(payload)
            # New format: {"status", "messages", "finish", "reward", "metrics"}
            messages = []
            for item in payload.get("messages", []) or []:
                role = item.get("role", "user")
                messages.append(
                    {
                        "role": "agent" if role in ("assistant", "agent") else "user",
                        "content": item.get("content", ""),
                    }
                )
            passthrough = {}
            for k, v in payload.items():
                if k in ("status", "messages", "finish", "reward", "metrics", "tools"):
                    continue
                passthrough[k] = v
            return TaskOutput(
                status=payload.get("status", SampleStatus.RUNNING),
                history=messages,
                result={
                    "reward": payload.get("reward", 0),
                    "metrics": payload.get("metrics", {}),
                    **passthrough,
                },
            )

        def _normalize_history(output: TaskOutput):
            history = output.history or []
            normalized = []
            for item in history:
                role = getattr(item, "role", None)
                if role is None and isinstance(item, dict):
                    role = item.get("role", "user")
                role = role or "user"
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content", "")
                content = content or ""
                # Agent clients in this repo only understand user/agent roles.
                normalized_role = "agent" if role in ("assistant", "agent") else "user"
                normalized.append(
                    {
                        "role": normalized_role,
                        "content": content,
                    }
                )
            return normalized

        def _seed_openai_messages(output: Optional[TaskOutput]) -> List[dict]:
            """
            Build OpenAI-compatible messages from task output history as fallback.
            This avoids sending empty message arrays to agent LLM endpoints.
            """
            if output is None or not output.history:
                return []
            seeded: List[dict] = []
            for item in output.history:
                role = getattr(item, "role", None)
                if role is None and isinstance(item, dict):
                    role = item.get("role")
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content")
                if not content:
                    continue
                if role in ("assistant", "agent"):
                    seeded.append({"role": "assistant", "content": content})
                elif role == "system":
                    seeded.append({"role": "system", "content": content})
                else:
                    seeded.append({"role": "user", "content": content})
            return seeded

        def _build_interact_payload(content: str, use_header_sid: bool, sid):
            # New controller expects OpenAI-like messages and session_id in request header.
            try:
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    messages = parsed
                else:
                    messages = [{"role": "assistant", "content": content}]
            except Exception:
                messages = [{"role": "assistant", "content": content}]
            if use_header_sid:
                return {"messages": messages}, {"session_id": str(sid)}
            return InteractRequest(
                session_id=sid,
                agent_response=AgentOutput(content=content),
            ).dict(), {}

        def _openai_trace_to_history(items):
            hist = []
            for m in items:
                role = m.get("role", "user")
                mapped = "agent" if role == "assistant" else "user"
                chunks = []
                c = m.get("content")
                if c:
                    chunks.append(str(c))
                if m.get("tool_calls"):
                    chunks.append(json.dumps(m["tool_calls"], ensure_ascii=False))
                hist.append(ChatHistoryItem(role=mapped, content="\n".join(chunks)))
            return hist

        # When running multiple trials, index is [base_idx, trial]. Strip the trial
        # dimension before sending to the controller which only accepts int/str.
        task_index = index[0] if isinstance(index, list) and len(index) == 2 else index
        try:
            start_resp = requests.post(
                self.controller_address + "/start_sample",
                json=StartSampleRequest(name=self.name, index=task_index).dict(),
            )
        except Exception as e:
            return TaskClientOutput(error=TaskError.NETWORK_ERROR.value, info=str(e))
        if start_resp.status_code == 406:
            return TaskClientOutput(
                error=TaskError.NOT_AVAILABLE.value, info=start_resp.text
            )
        if start_resp.status_code != 200:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value, info=start_resp.text
            )
        start_payload = start_resp.json()
        sid = _extract_session_id(resp=start_resp, payload=start_payload)
        if sid is None:
            return TaskClientOutput(
                error=TaskError.START_FAILED.value,
                info="Cannot find session id in start_sample response.",
            )
        # No session_id in payload usually means new controller protocol.
        use_header_sid = "session_id" not in start_payload
        latest_output = _normalize_output(start_payload)
        api_messages = copy.deepcopy(
            start_payload.get("messages") or _seed_openai_messages(latest_output)
        )
        api_tools = start_payload.get("tools")
        # Accumulate LLM token usage across all turns in this episode.
        _ep_usage: dict = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        _ep_inference_metadata: List[dict] = []

        while SampleStatus(latest_output.status) == SampleStatus.RUNNING:
            if use_header_sid and hasattr(agent, "inference_openai"):
                if not api_messages:
                    api_messages = _seed_openai_messages(latest_output)
                if not api_messages:
                    requests.post(
                        self.controller_address + "/cancel",
                        json={},
                        headers={"session_id": str(sid)},
                    )
                    return TaskClientOutput(
                        error=TaskError.INTERACT_FAILED.value,
                        info="No messages available for inference_openai; refusing to send empty message list.",
                        output=latest_output,
                    )
                try:
                    assistant_msg, _turn_usage = agent.inference_openai(api_messages, api_tools)
                    for _k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        _ep_usage[_k] += _turn_usage.get(_k, 0)
                    _turn_metadata = _turn_usage.get("_agentbench_inference_metadata")
                    if isinstance(_turn_metadata, dict):
                        _ep_inference_metadata.append(copy.deepcopy(_turn_metadata))
                except AgentContextLimitException:
                    assistant_msg = {"role": "assistant", "content": ""}
                    _turn_usage = {}
                except Exception as e:
                    if hasattr(agent, "model_name"):
                        model_name = agent.model_name
                    elif hasattr(agent, "name"):
                        model_name = agent.name
                    else:
                        model_name = agent.__class__.__name__
                    print(f"ERROR: {model_name}/{self.name} agent error", e)
                    requests.post(
                        self.controller_address + "/cancel",
                        json={},
                        headers={"session_id": str(sid)},
                    )
                    return TaskClientOutput(
                        error=TaskError.AGENT_FAILED.value,
                        info=str(e),
                        output=latest_output,
                    )
                api_messages.append(copy.deepcopy(assistant_msg))
                payload = {"messages": [assistant_msg]}
                headers = {"session_id": str(sid)}
            else:
                try:
                    content = agent.inference(_normalize_history(latest_output))
                except AgentContextLimitException:
                    content = ""
                except Exception as e:
                    if hasattr(agent, "model_name"):
                        model_name = agent.model_name
                    elif hasattr(agent, "name"):
                        model_name = agent.name
                    else:
                        model_name = agent.__class__.__name__
                    print(f"ERROR: {model_name}/{self.name} agent error", e)
                    if use_header_sid:
                        requests.post(
                            self.controller_address + "/cancel",
                            json={},
                            headers={"session_id": str(sid)},
                        )
                    else:
                        requests.post(
                            self.controller_address + "/cancel",
                            json=CancelRequest(session_id=sid).dict(),
                        )
                    return TaskClientOutput(
                        error=TaskError.AGENT_FAILED.value,
                        info=str(e),
                        output=latest_output,
                    )
                payload, headers = _build_interact_payload(
                    content=content, use_header_sid=use_header_sid, sid=sid
                )

            try:
                result = requests.post(
                    self.controller_address + "/interact",
                    json=payload,
                    headers=headers,
                )
            except Exception as e:
                return TaskClientOutput(
                    error=TaskError.NETWORK_ERROR.value,
                    info=str(e),
                    output=latest_output,
                )
            if result.status_code != 200:
                if use_header_sid:
                    requests.post(
                        self.controller_address + "/cancel",
                        json={},
                        headers={"session_id": str(sid)},
                    )
                else:
                    requests.post(
                        self.controller_address + "/cancel",
                        json=CancelRequest(session_id=sid).dict(),
                    )
                return TaskClientOutput(
                    error=TaskError.INTERACT_FAILED.value,
                    info=result.text,
                    output=latest_output,
                )

            resp_payload = result.json()
            if use_header_sid and hasattr(agent, "inference_openai"):
                for m in resp_payload.get("messages", []):
                    api_messages.append(copy.deepcopy(m))
                if resp_payload.get("tools"):
                    api_tools = resp_payload["tools"]
            latest_output = _normalize_output(resp_payload)

        if use_header_sid and hasattr(agent, "inference_openai"):
            base = latest_output.result if isinstance(latest_output.result, dict) else {}
            result = {
                **base,
                "openai_messages": api_messages,
                "token_usage": _ep_usage,
            }
            if _ep_inference_metadata:
                result["agent_inference_trace"] = _ep_inference_metadata
            return TaskClientOutput(
                output=TaskOutput(
                    status=latest_output.status,
                    result=result,
                    history=_openai_trace_to_history(api_messages),
                )
            )
        return TaskClientOutput(output=latest_output)

    def calculate_overall(self, results: List[TaskOutput], trials: int = 1) -> JSONSerializable:
        statistics = {s: 0 for s in SampleStatus}
        for result in results:
            statistics[SampleStatus(result.status)] += 1
        for s in SampleStatus:
            statistics[s] /= len(results)
        statistics["average_history_length"] = sum(
            [len(result.history) for result in results]
        ) / len(results)
        statistics["max_history_length"] = max(
            [len(result.history) for result in results]
        )
        statistics["min_history_length"] = min(
            [len(result.history) for result in results]
        )
        ret = {
            "total": len(results),
            "validation": statistics,
        }
        local = _local_calculate_overall(self.name, results, trials=trials)
        slim_results = _slim_outputs_for_worker_aggregate(results)
        payload = CalculateOverallRequest(name=self.name, results=slim_results).dict()
        res = requests.post(
            self.controller_address + "/calculate_overall",
            json=payload,
            timeout=60,
        )
        if res.status_code == 200:
            remote = res.json()
            ret["custom"] = local
            ret["custom"]["_remote_custom"] = remote
            return ret
        last_worker_err: Optional[str] = None
        # Newer controller images often omit /calculate_overall; task workers still expose it.
        try:
            lw = requests.get(self.controller_address + "/list_workers", timeout=10)
            if lw.status_code == 200:
                task_info = lw.json().get(self.name) or {}
                for w in (task_info.get("workers") or {}).values():
                    st = w.get("status")
                    alive = (
                        st == WorkerStatus.ALIVE
                        or (isinstance(st, str) and st.upper() == "ALIVE")
                        or (isinstance(st, int) and st == int(WorkerStatus.ALIVE))
                    )
                    if not alive:
                        continue
                    base = (w.get("address") or "").rstrip("/")
                    if not base:
                        continue
                    calc_url = (
                        f"{base}/calculate_overall"
                        if base.endswith("/api")
                        else f"{base}/api/calculate_overall"
                    )
                    try:
                        wr = requests.post(calc_url, json=payload, timeout=120)
                        if wr.status_code == 200:
                            remote = wr.json()
                            ret["custom"] = local
                            ret["custom"]["_remote_custom"] = remote
                            return ret
                        last_worker_err = f"{calc_url} -> {wr.status_code}: {wr.text[:500]}"
                    except requests.RequestException as e:
                        last_worker_err = f"{calc_url} -> {e!r}"
        except Exception as e:
            last_worker_err = f"list_workers/worker loop: {e!r}; controller said: {res.text[:300]}"

        ret["custom"] = local
        if last_worker_err:
            ret["custom"] = {**local, "_worker_aggregate_error": last_worker_err}
        return ret
