import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .http_agent import HTTPAgent


MEMORY_PREFIX = """A previous complete attempt at this same task failed. Use the following reflection as private memory for this retry. Re-check it against the current observation; do not assume an action happened unless it appears in the current trajectory.

<previous_attempt_reflection>
{reflection}
</previous_attempt_reflection>"""


def trajectory_fingerprint(messages: List[Dict[str, Any]]) -> str:
    """Identify a task from the original system prompt and first user turn."""
    identity: List[Dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        content = message.get("content")
        if role not in {"system", "user"} or not isinstance(content, str):
            continue
        identity.append({"role": role, "content": content})
        if role == "user":
            break
    payload = json.dumps(identity, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_memories(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        raise ValueError("episode reflection memory must contain an entries list")
    memories: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        reflection = entry.get("reflection")
        if not isinstance(fingerprint, str) or not fingerprint:
            continue
        if not isinstance(reflection, str) or not reflection.strip():
            continue
        memories[fingerprint] = copy.deepcopy(entry)
    return memories


def _inject_memory(
    messages: List[Dict[str, Any]], reflection: str
) -> List[Dict[str, Any]]:
    injected = copy.deepcopy(messages)
    memory = MEMORY_PREFIX.format(reflection=reflection.strip())
    for message in injected:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        message["content"] = f"{content}\n\n{memory}" if content else memory
        return injected
    injected.insert(0, {"role": "system", "content": memory})
    return injected


class EpisodeReflexionHTTPAgent(HTTPAgent):
    """Run a retry with reflection memory produced after a failed episode."""

    def __init__(
        self,
        *args,
        episode_reflection_memory_path: str,
        episode_reflection_require_match: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.episode_reflection_memory_path = Path(
            episode_reflection_memory_path
        ).expanduser().resolve()
        self.episode_reflection_require_match = bool(
            episode_reflection_require_match
        )
        self._memories = _load_memories(self.episode_reflection_memory_path)

    def inference_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        fingerprint = trajectory_fingerprint(messages)
        entry = self._memories.get(fingerprint)
        if entry is None:
            if self.episode_reflection_require_match:
                raise RuntimeError(
                    "No episode reflection memory matches task fingerprint "
                    f"{fingerprint}"
                )
            action_messages = copy.deepcopy(messages)
        else:
            action_messages = _inject_memory(messages, entry["reflection"])

        action, usage = super().inference_openai(action_messages, tools)
        usage["_agentbench_inference_metadata"] = {
            "method": "episode_reflexion_retry",
            "protocol_version": "episode_reflexion_v1",
            "fingerprint": fingerprint,
            "memory_found": entry is not None,
            "source_index": entry.get("global_index") if entry else None,
            "reflection": entry.get("reflection") if entry else None,
        }
        return action, usage
