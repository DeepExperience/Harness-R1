import copy
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .http_agent import HTTPAgent


MEMORY_PREFIX = """Use the following training-only experience as optional guidance. Adapt object identifiers and locations to the current scene, and only execute actions that are currently available. The current observation always overrides the memory.

<retrieved_training_experience method=\"{method}\">
{memory}
</retrieved_training_experience>"""

_TASK_RE = re.compile(
    r"your task is to:\s*(.*?)(?:\.?\s+available actions:|$)",
    re.IGNORECASE | re.DOTALL,
)
_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]*")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "some",
    "the",
    "then",
    "to",
    "your",
}


def extract_task_query(messages: List[Dict[str, Any]]) -> str:
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = _TASK_RE.search(content)
        if match:
            return " ".join(match.group(1).strip().split())
        return " ".join(content.strip().split())
    return ""


def _tokens(text: str) -> List[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    return {token: float(count) * idf.get(token, 1.0) for token, count in counts.items()}


def _cosine(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _inject_memory(
    messages: List[Dict[str, Any]], method: str, memory: str
) -> List[Dict[str, Any]]:
    injected = copy.deepcopy(messages)
    block = MEMORY_PREFIX.format(method=method, memory=memory.strip())
    for message in injected:
        if message.get("role") != "system":
            continue
        content = message.get("content")
        message["content"] = f"{content}\n\n{block}" if content else block
        return injected
    injected.insert(0, {"role": "system", "content": block})
    return injected


class RetrievedMemoryHTTPAgent(HTTPAgent):
    """Inject frozen, train-only global or query-retrieved memories."""

    def __init__(
        self,
        *args,
        retrieved_memory_path: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.retrieved_memory_path = Path(retrieved_memory_path).expanduser().resolve()
        raw = json.loads(self.retrieved_memory_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("retrieved memory file must contain a JSON object")
        self.method = str(raw.get("method") or "retrieved_memory")
        self.global_memory = str(raw.get("global_memory") or "").strip()
        self.top_k = max(0, int((raw.get("retrieval") or {}).get("top_k", 1)))
        entries = raw.get("entries") or []
        if not isinstance(entries, list):
            raise ValueError("retrieved memory entries must be a list")
        self.entries = [
            copy.deepcopy(entry)
            for entry in entries
            if isinstance(entry, dict)
            and isinstance(entry.get("query"), str)
            and isinstance(entry.get("memory"), str)
            and entry["query"].strip()
            and entry["memory"].strip()
        ]
        if not self.global_memory and not self.entries:
            raise ValueError("retrieved memory file has no usable memory")

        document_tokens = [_tokens(entry["query"]) for entry in self.entries]
        document_count = max(1, len(document_tokens))
        document_frequency: Counter[str] = Counter()
        for tokens in document_tokens:
            document_frequency.update(set(tokens))
        self._idf = {
            token: math.log((document_count + 1) / (frequency + 1)) + 1.0
            for token, frequency in document_frequency.items()
        }
        self._entry_vectors = [
            _tfidf_vector(tokens, self._idf) for tokens in document_tokens
        ]

    def _retrieve(self, query: str) -> List[Tuple[Dict[str, Any], float]]:
        if self.top_k <= 0 or not self.entries:
            return []
        query_vector = _tfidf_vector(_tokens(query), self._idf)
        scored = [
            (entry, _cosine(query_vector, vector))
            for entry, vector in zip(self.entries, self._entry_vectors)
        ]
        scored.sort(key=lambda item: (-item[1], str(item[0].get("id") or "")))
        return scored[: min(self.top_k, len(scored))]

    def inference_openai(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        query = extract_task_query(messages)
        retrieved = self._retrieve(query)
        memory_parts = []
        if self.global_memory:
            memory_parts.append(self.global_memory)
        memory_parts.extend(entry["memory"].strip() for entry, _ in retrieved)
        memory = "\n\n".join(memory_parts)
        action_messages = _inject_memory(messages, self.method, memory)

        action, usage = super().inference_openai(action_messages, tools)
        usage["_agentbench_inference_metadata"] = {
            "method": self.method,
            "protocol_version": "retrieved_memory_v1",
            "query": query,
            "memory_path": str(self.retrieved_memory_path),
            "memory_chars": len(memory),
            "retrieved": [
                {
                    "id": entry.get("id"),
                    "query": entry.get("query"),
                    "score": score,
                }
                for entry, score in retrieved
            ],
        }
        return action, usage
