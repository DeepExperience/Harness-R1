import asyncio
import json
import logging
import os
import random
import weakref
from typing import Any, Dict, List, Optional, Tuple, Sequence, Union

from agentrl.worker.environment import create_controller
from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (AgentCancelledException,
                                    RewardHistoryItem,
                                    SampleIndex,
                                    SampleStatus,
                                    TaskSampleExecutionResult)
from openai.types.chat import (ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)

from .environment import DBBenchEnvironmentDelegation, TYPE_SQLITE
from .interaction import Database, MySQLDatabase, SQLiteDatabase
from .result_processor import DBResultProcessor
from src.server.harness import (
    DBBenchHarnessConfig,
    DBBenchHarnessRuntime,
    compile_hook,
    patch_dbbench_tool_descriptions,
    patch_dbbench_system_prompt,
    run_hook,
)
from src.server.harness.dbbench import rescue_tool_call_from_text
from src.server.harness.dsl import normalize_harness_overlay, overlay_cold_start_hints

SYSTEM_PROMPT = """I will ask you a question, then you should help me operate a MySQL database with SQL to answer the question.
You have to explain the problem and your solution to me and write down your thoughts.
After thinking and explaining thoroughly, every round you can choose to operate or to answer with the two specific tools provided.
If you should execute a SQL query, use the `execute_sql` function, Your SQL should be in one line.
Every time you can only execute one SQL statement. I will only execute the statement in the first SQL code block. Every time you write a SQL, I will execute it for you and give you the output.
If you are done operating, and you want to commit your final answer, then use the `commit_final_answer` function.
DO NOT use this tool unless you are sure about your answer. I expect an accurate and correct answer.
Your answer should be accurate. Your answer must be exactly the same as the correct answer.
If the question is about modifying the database, then after done operation, your answer field can be anything.
If your response cannot match any pattern I mentioned earlier, you will be judged as FAIL immediately.
You should always use the tools provided to submit your answer. Be careful not to write it in the content field.
Your input will be raw MySQL response, you have to deal with it by yourself."""


def _dbbench_code_context(
    harness_runtime: DBBenchHarnessRuntime,
    *,
    tool_name: str = "",
    sql: str = "",
    answers: Optional[List[str]] = None,
    db_response: str = "",
    remaining_rounds: Optional[int] = None,
    round_num: int = 0,
    h2_result: Optional[Dict[str, Any]] = None,
    gate_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the small observable context exposed to DBBench code hooks."""
    state = harness_runtime.state
    task_ctx = harness_runtime.task_ctx
    remaining = remaining_rounds if remaining_rounds is not None else 999
    last_error_kind = state.last_error_kind or ""
    repeated_sql = bool(
        len(state.sql_history) >= 2
        and state.sql_history[-1] == state.sql_history[-2]
    )
    no_sql_yet = not state.sql_history
    is_commit = tool_name == "commit_final_answer"
    is_execute = tool_name == "execute_sql"
    is_mutation = bool(task_ctx and task_ctx.task_type in {"INSERT", "UPDATE", "DELETE"})
    answers = answers or []
    return {
        "action": {
            "tool": tool_name,
            "query": sql or "",
            "answers": list(answers),
            "h2_action": (h2_result or {}).get("action"),
            "h2_blocked_reason": (h2_result or {}).get("blocked_reason", ""),
            "commit_gate_action": (gate_result or {}).get("action"),
            "commit_gate_reason": (gate_result or {}).get("blocked_reason", ""),
        },
        "state": {
            "sql_count": len(state.sql_history),
            "last_sql": state.last_sql or "",
            "last_result": state.last_result_raw or "",
            "last_error_kind": last_error_kind,
            "last_error_text": state.last_error_text or "",
            "last_result_was_error": bool(state.last_result_was_error),
            "error_streak": state.error_streak,
            "empty_streak": state.empty_streak,
            "text_only_streak": state.text_only_streak,
            "loop_streak": state.loop_streak,
            "mutation_attempted": bool(state.mutation_attempted),
            "candidate_answer": state.candidate_answer,
            "candidate_answer_shape": state.candidate_answer_shape,
            "candidate_implausible": bool(state.candidate_implausible),
            "remaining_rounds": remaining,
        },
        "task": {
            "task_type": task_ctx.task_type if task_ctx else None,
            "answer_shape": task_ctx.answer_shape if task_ctx else None,
            "target_table": task_ctx.target_table_sanitized if task_ctx else None,
            "description": task_ctx.raw_description if task_ctx else "",
        },
            "dbbench": {
                "sql_history": list(state.sql_history),
                "sql_history_raw": list(state.sql_history_raw),
                "discovered_columns": {
                    key: list(value)
                    for key, value in state.discovered_columns.items()
                },
                "db_response": db_response or "",
                "round": int(round_num),
            },
        "predicates": {
            "execute_sql": is_execute,
            "commit_final_answer": is_commit,
            "no_sql_yet": no_sql_yet,
            "commit_before_sql": is_commit and no_sql_yet,
            "commit_empty_answer": is_commit and not answers,
            "mutation_task": is_mutation,
            "mutation_not_attempted": is_mutation and not state.mutation_attempted,
            "last_result_error": bool(state.last_result_was_error),
            "last_result_empty": last_error_kind == "empty",
            "unknown_column_error": last_error_kind == "unknown_col",
            "syntax_error": last_error_kind == "syntax",
            "repeated_sql": repeated_sql or state.loop_streak > 0,
            "candidate_answer_available": state.candidate_answer is not None,
            "remaining_rounds_low": remaining <= 3,
            "text_only_loop": state.text_only_streak >= 2,
        },
    }


class DBBenchTask(Task):

    def __init__(self,
                 data_file: str,
                 db_file: Optional[str] = None,
                 db_password: str = 'password',
                 max_round: int = 20,
                 start: int = 0,
                 end: Optional[int] = None,
                 task_ids: Optional[List[int]] = None,
                 shuffle_seed: Optional[int] = None,
                 sample_size: Optional[int] = None,
                 env_driver: str = 'docker',
                 env_options: Optional[dict] = None,
                 harness: Optional[dict] = None,
                 **configs):
        self.harness_overlay = normalize_harness_overlay(
            configs.pop("harness_overlay", {}),
            benchmark="dbbench",
        )
        # ── Harness config ─────────────────────────────────────────────────
        h = harness or {}
        enabled    = bool(h.get("enabled", False))
        h2         = bool(h.get("h2", True))
        h3         = bool(h.get("h3", True))
        h4         = bool(h.get("h4", True))
        h5         = bool(h.get("h5", True))
        h5_top_k   = int(h.get("h5_top_k", 2))
        h2_repeat  = int(h.get("h2_repeat_sql_block_after", 2))
        h4_stall   = int(h.get("h4_stall_window", 3))
        h4_empty   = int(h.get("h4_empty_threshold", 2))
        h4_bwarn   = int(h.get("h4_budget_warn_threshold", 3))
        h4_bforce  = int(h.get("h4_budget_force_threshold", 2))
        h5_words   = int(h.get("h5_hint_max_words", 40))

        self.harness_config = DBBenchHarnessConfig(
            enabled=enabled,
            h2_enabled=h2,
            h3_enabled=h3,
            h4_enabled=h4,
            h5_enabled=h5,
            h5_top_k=h5_top_k,
            h2_repeat_sql_block_after=h2_repeat,
            h4_stall_window=h4_stall,
            h4_empty_threshold=h4_empty,
            h4_budget_warn_threshold=h4_bwarn,
            h4_budget_force_threshold=h4_bforce,
            h4_hint_max_words=h5_words,
        )
        self.overlay_has_skills = bool(self.harness_overlay.get("skills"))
        self.overlay_has_code_hooks = bool(self.harness_overlay.get("code_hooks"))
        self._code_hooks: Dict[str, List[Any]] = {}
        for item in self.harness_overlay.get("code_hooks", []) or []:
            hook_name = str(item.get("hook") or "")
            code = item.get("code")
            if not hook_name or not isinstance(code, str):
                continue
            try:
                self._code_hooks.setdefault(hook_name, []).append(
                    compile_hook(code, benchmark="dbbench")
                )
            except Exception:
                continue
        self.harness_substrate_enabled = bool(
            self.harness_config.enabled
            or self.overlay_has_skills
            or self.overlay_has_code_hooks
        )

        # H3: patch tool descriptions once at class init.
        tools = configs.pop("tools", None)
        if self.harness_config.enabled and self.harness_config.h3_enabled:
            tools = patch_dbbench_tool_descriptions(tools)
        configs["tools"] = tools

        super().__init__(**configs)
        self.full_async = True
        self.logger = logging.getLogger(__name__)

        self.max_round = max_round
        self.data_file = data_file
        self.db_root_dir = db_file
        self.start = int(start) if start is not None else 0
        self.end = int(end) if end is not None else None
        self.task_ids = [int(item) for item in task_ids] if task_ids is not None else None
        self.shuffle_seed = int(shuffle_seed) if shuffle_seed is not None else None
        self.sample_size = int(sample_size) if sample_size is not None else None

        self.dataset = []
        with open(self.data_file) as f:
            raw_data = f.read()
            if self.data_file.endswith("json"):
                data = json.loads(raw_data)
            else:
                data = [json.loads(line) for line in raw_data.strip().split('\n')]

        for entry in data:
            ans_key = "answer_md5" if entry["type"][0] in ("INSERT", "DELETE", "UPDATE") else "label"
            ans = entry.pop(ans_key, None)
            inp = entry
            self.dataset.append((inp, ans))

        self.env_delegation = DBBenchEnvironmentDelegation(db_password)
        self.env_controller = create_controller(env_driver, self.env_delegation, **(env_options or {}))
        self.env_controller_background_task = None

        self.logger.info(
            "DBBench initialized with %s samples. Root dir: %s. range=[%s:%s], "
            "shuffle_seed=%s, sample_size=%s, explicit_task_ids=%s, harness_enabled=%s",
            len(self.dataset), self.db_root_dir,
            self.start, self.end, self.shuffle_seed, self.sample_size,
            len(self.task_ids) if self.task_ids is not None else None,
            self.harness_config.enabled,
        )

    def _run_code_hooks(
        self,
        hook_name: str,
        ctx: Dict[str, Any],
        nb: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        outputs: List[Dict[str, Any]] = []
        for fn in self._code_hooks.get(hook_name, []) or []:
            out = run_hook(fn, ctx, nb, hook_name=hook_name)
            if out:
                outputs.append(out)
        return outputs

    @staticmethod
    def _apply_before_tool_effect(
        session: Session,
        effect: Dict[str, Any],
        call_id: str,
        harness_trace: Dict[str, List[Any]],
        *,
        round_num: int,
        tool_name: str,
    ) -> bool:
        """Apply SQL-safe before-tool effects. Returns True when blocked."""
        kind = effect.get("kind")
        if kind == "block_and_prompt":
            message = effect.get("message") or "Harness blocked this tool call. Choose a safer next step."
            session.inject(ChatCompletionToolMessageParam(
                role='tool',
                tool_call_id=call_id,
                content=message,
            ))
            harness_trace["code_hooks"].append({
                "round": round_num,
                "hook": "on_before_action",
                "tool": tool_name,
                "effect": "block_and_prompt",
                "message": str(message)[:200],
            })
            return True
        if kind in {"rewrite_action", "force_action"}:
            # Generic action rewrites lower-case action strings in code_runner;
            # that is unsafe for SQL literals, so DBBench v1 ignores them.
            harness_trace["code_hooks"].append({
                "round": round_num,
                "hook": "on_before_action",
                "tool": tool_name,
                "effect": f"ignored_{kind}",
                "reason": "dbbench_v1_sql_rewrite_not_supported",
            })
        return False

    @staticmethod
    def _apply_post_step_effect(
        session: Session,
        effect: Dict[str, Any],
        harness_trace: Dict[str, List[Any]],
        *,
        round_num: int,
        tool_name: str,
    ) -> None:
        kind = effect.get("kind")
        if kind == "inject_hint" and effect.get("message"):
            message = effect["message"]
            session.inject(ChatCompletionUserMessageParam(role='user', content=message))
            harness_trace["code_hooks"].append({
                "round": round_num,
                "hook": "on_post_step",
                "tool": tool_name,
                "effect": "inject_hint",
                "message": str(message)[:200],
            })
        elif kind == "force_action":
            harness_trace["code_hooks"].append({
                "round": round_num,
                "hook": "on_post_step",
                "tool": tool_name,
                "effect": "ignored_force_action",
                "reason": "dbbench_v1_force_action_not_supported",
            })

    def get_indices(self) -> List[SampleIndex]:
        total = len(self.dataset)
        if self.task_ids is not None:
            indices = [idx for idx in self.task_ids if 0 <= idx < total]
        else:
            start = max(0, self.start)
            end = total if self.end is None else min(total, max(start, self.end))
            indices = list(range(start, end))
        if self.shuffle_seed is not None:
            random.Random(self.shuffle_seed).shuffle(indices)
        if self.sample_size is not None:
            indices = indices[:self.sample_size]
        return indices

    async def start_sample(self, index: int, session: Session) -> TaskSampleExecutionResult:
        self.env_controller.loop = asyncio.get_running_loop()
        if not self.env_controller_background_task:
            self.env_controller_background_task = asyncio.create_task(
                self.env_controller.background_task()
            )
            weakref.finalize(self, self.env_controller_background_task.cancel)

        database: Optional[Database] = None
        harness_trace: Dict[str, List[Any]] = {
            "h0": [], "h2": [], "h2_commit": [], "h3": [], "h4": [], "h5": [], "code_hooks": [],
        }
        try:
            entry = self.dataset[index][0]
            ground_truth = self.dataset[index][1]

            # ── H0 + H_SCHEMA ─────────────────────────────────────────────
            harness_runtime: Optional[DBBenchHarnessRuntime] = None
            if self.harness_substrate_enabled:
                harness_runtime = DBBenchHarnessRuntime(self.harness_config)
                harness_runtime.init_task(entry)
                harness_trace["h0"].append({
                    "task_type": harness_runtime.task_ctx.task_type if harness_runtime.task_ctx else None,
                    "answer_shape": harness_runtime.task_ctx.answer_shape if harness_runtime.task_ctx else None,
                })
                if self.harness_config.enabled and self.harness_config.h3_enabled:
                    harness_trace["h3"].append({"applied": True})

            # ── DB init ────────────────────────────────────────────────────
            use_sqlite = entry.get("user_sqlite", False)
            if use_sqlite:
                db_dir = entry['create']['database']
                init_file = entry['create']['init']
                sqlite_path = os.path.join(self.db_root_dir, db_dir, init_file)
                database = SQLiteDatabase(sqlite_path)
                await database.initialize()
            else:
                init_sql = self._build_init_sql(entry)
                database = MySQLDatabase(self.env_controller)
                await database.initialize()
                await database.batch_execute(init_sql)

            # ── System prompt (H3 append if enabled) ──────────────────────
            sys_prompt = SYSTEM_PROMPT
            if (
                harness_runtime is not None
                and self.harness_config.enabled
                and self.harness_config.h3_enabled
            ):
                sys_prompt = patch_dbbench_system_prompt(sys_prompt)

            # ── H5 / overlay cold-start (append to initial system prompt) ─
            cold_skills = []
            code_hook_tool_hints: List[str] = []
            code_hook_nb: Dict[str, Any] = {}
            if harness_runtime is not None:
                if self.harness_config.enabled and self.harness_config.h5_enabled:
                    cold_skills.extend(harness_runtime.cold_start_skill_hints())
                if self.overlay_has_skills:
                    cold_skills.extend(
                        overlay_cold_start_hints(
                            self.harness_overlay,
                            task_type=harness_runtime.task_ctx.task_type if harness_runtime.task_ctx else None,
                        )
                    )
                if self.overlay_has_code_hooks and self._code_hooks.get("on_init"):
                    code_ctx = _dbbench_code_context(
                        harness_runtime,
                        remaining_rounds=self.max_round,
                        round_num=0,
                    )
                    for hook_out in self._run_code_hooks("on_init", code_ctx, code_hook_nb):
                        for skill in hook_out.get("skills", []) or []:
                            if isinstance(skill, dict):
                                cold_skills.append(skill)
                            elif isinstance(skill, str):
                                cold_skills.append({"text": skill})
                        if hook_out.get("tool_hint"):
                            code_hook_tool_hints.append(str(hook_out["tool_hint"]))
                        harness_trace["code_hooks"].append({
                            "round": 0,
                            "hook": "on_init",
                            "skills": len(hook_out.get("skills", []) or []),
                            "tool_hint": bool(hook_out.get("tool_hint")),
                        })
                if cold_skills:
                    skill_lines = [f"- {item['text']}" for item in cold_skills if item.get("text")]
                    if skill_lines:
                        sys_prompt += (
                            "\n\nSome tips that may help for this task:\n"
                            + "\n".join(skill_lines)
                        )
                if code_hook_tool_hints:
                    sys_prompt += (
                        "\n\nAdditional reusable tool-use hint from the harness:\n"
                        + " ".join(code_hook_tool_hints)
                    )
            session.inject(ChatCompletionSystemMessageParam(role='system', content=sys_prompt))

            # ── User prompt + schema card (H_SCHEMA) ──────────────────────
            user_prompt = ""
            if "evidence" in entry and entry['evidence'] != "":
                user_prompt += "Evidence about the question: " + entry["evidence"] + "\n"
            if "add_description" in entry and entry['add_description'] != "":
                user_prompt += "Additional table information about the question: " + entry["add_description"] + "\n"
            if (
                harness_runtime is not None
                and self.harness_config.enabled
                and self.harness_config.h3_enabled
            ):
                schema_card = harness_runtime.schema_card()
                if schema_card:
                    user_prompt += "\n" + schema_card + "\n"
            user_prompt += "Question: " + entry["description"] + "\n"
            session.inject(ChatCompletionUserMessageParam(role='user', content=user_prompt))

            # ── H5 trace ───────────────────────────────────────────────────
            if harness_runtime is not None and cold_skills:
                for item in cold_skills:
                    harness_trace["h5"].append(item)

            # ── Main loop ─────────────────────────────────────────────────
            for current_round in range(self.max_round):
                remaining = self.max_round - current_round

                # ── Force-action consumption (H2) ──────────────────────────
                if (
                    harness_runtime is not None
                    and self.harness_config.enabled
                    and self.harness_config.h2_enabled
                    and harness_runtime.force_next_action is not None
                ):
                    fa = harness_runtime.force_next_action
                    harness_runtime.force_next_action = None
                    self.logger.info(f"[H2] consuming force_next_action: {fa['name']}")
                    harness_trace["h2"].append({
                        "round": current_round + 1,
                        "reason": "force_consumed",
                        "action": fa["name"],
                    })
                    # Synthesise a fake tool result path — jump straight to
                    # commit handling for commit_final_answer, or inject as
                    # tool message for execute_sql.
                    if fa["name"] == "commit_final_answer":
                        force_answers = (fa.get("arguments") or {}).get("answers", [])
                        finish_result = await self._do_commit(
                            force_answers, entry, ground_truth, database,
                            harness_runtime=harness_runtime,
                            harness_trace=harness_trace,
                            current_round=current_round,
                        )
                        if finish_result is not None:
                            self._persist_harness_trace(session, harness_trace)
                            session.inject(RewardHistoryItem(
                                reward=1 if finish_result.get("is_correct") else 0,
                                score=1 if finish_result.get("is_correct") else 0,
                            ))
                            return TaskSampleExecutionResult(
                                status=SampleStatus.COMPLETED,
                                result={**finish_result, "harness_trace": harness_trace},
                            )
                    continue

                if (
                    harness_runtime is not None
                    and self.overlay_has_code_hooks
                    and self._code_hooks.get("make_pre_hint")
                ):
                    pre_ctx = _dbbench_code_context(
                        harness_runtime,
                        remaining_rounds=remaining,
                        round_num=current_round + 1,
                    )
                    for hook_out in self._run_code_hooks("make_pre_hint", pre_ctx, code_hook_nb):
                        message = hook_out.get("message")
                        if message:
                            session.inject(ChatCompletionUserMessageParam(role='user', content=message))
                            harness_trace["code_hooks"].append({
                                "round": current_round + 1,
                                "hook": "make_pre_hint",
                                "message": str(message)[:200],
                            })

                response = await session.action()

                tool_calls = []
                response_content = None
                for message in response.messages:
                    if response_content is None:
                        response_content = message.get("content")
                    tool_calls.extend(message.get("tool_calls", []) or [])

                # ── Rescue parser (H2) ─────────────────────────────────────
                if (
                    not tool_calls
                    and harness_runtime is not None
                    and self.harness_config.enabled
                    and self.harness_config.h2_enabled
                    and response_content
                ):
                    rescued = rescue_tool_call_from_text(response_content)
                    if rescued is not None:
                        self.logger.info(f"[H2] rescued tool call: {rescued['name']}")
                        harness_trace["h2"].append({
                            "round": current_round + 1,
                            "reason": "rescued_text_embedded",
                            "action": rescued["name"],
                        })
                        tool_calls = [{
                            "id": f"harness_rescued_{current_round}",
                            "type": "function",
                            "function": {
                                "name": rescued["name"],
                                "arguments": json.dumps(rescued.get("arguments", {})),
                            },
                        }]

                if not tool_calls:
                    if harness_runtime is not None:
                        harness_runtime.note_text_only_turn()
                    has_xml = bool(response_content and "<tool_call>" in (response_content or "").lower())
                    if has_xml:
                        nudge = (
                            "Harness: your <tool_call> XML was malformed or truncated. "
                            "Use function calling API directly — do NOT write tool calls as XML text."
                        )
                    else:
                        nudge = "No executable tool calls found. Please call a tool."
                    # H4: check text_only_loop even in no-tool-call turns so we can
                    # force-commit or escalate when the agent is stuck writing plain text.
                    if (
                        harness_runtime is not None
                        and self.harness_config.enabled
                        and self.harness_config.h4_enabled
                    ):
                        h4_result = harness_runtime.post_step_monitor(remaining_rounds=remaining)
                        if h4_result.get("audit_reason") == "text_only_loop":
                            self.logger.info("[H4] text_only_loop detected in no-tool-call turn")
                            harness_trace["h4"].append({
                                "round": current_round + 1,
                                "reason": "text_only_loop",
                                "budget_branch": None,
                            })
                            if h4_result.get("force_action"):
                                harness_runtime.force_next_action = h4_result["force_action"]
                            if h4_result.get("recovery_prompt"):
                                nudge = h4_result["recovery_prompt"]
                    session.inject(ChatCompletionUserMessageParam(role='user', content=nudge))
                    continue

                if harness_runtime is not None:
                    harness_runtime.reset_text_only_streak()

                # Process each tool call (usually just one)
                committed = False
                for tool_call in tool_calls:
                    call_id = tool_call.get("id", "")
                    try:
                        function_name = tool_call.get("function", {}).get("name", "")
                        arguments_raw = tool_call.get("function", {}).get("arguments", "{}")
                        arguments = json.loads(arguments_raw)
                    except Exception:
                        session.inject(ChatCompletionToolMessageParam(
                            role='tool',
                            tool_call_id=call_id,
                            content="Error: Failed to parse tool call arguments.",
                        ))
                        self.logger.warning(f"Error parsing tool call: {tool_call}", exc_info=True)
                        continue

                    if function_name == "execute_sql":
                        sql = list(arguments.values())[0] if arguments else ""

                        # ── H2: SQL pre-validate ───────────────────────────
                        h2_audit: Optional[Dict[str, Any]] = None
                        if (
                            harness_runtime is not None
                            and self.harness_config.enabled
                            and self.harness_config.h2_enabled
                        ):
                            sql_check = harness_runtime.pre_validate_sql(sql)
                            h2_audit = sql_check
                            if sql_check["action"] == "block":
                                self.logger.info(f"[H2] blocked SQL: {sql_check['blocked_reason']}")
                                harness_trace["h2"].append({
                                    "round": current_round + 1,
                                    "reason": "sql_blocked",
                                    "detail": sql_check["blocked_reason"],
                                })
                                session.inject(ChatCompletionToolMessageParam(
                                    role='tool',
                                    tool_call_id=call_id,
                                    content=f"Error: SQL blocked by harness ({sql_check['blocked_reason']}).",
                                ))
                                continue
                            elif sql_check["action"] == "force_commit":
                                force_ans = (sql_check.get("force_args") or {}).get("answers", [])
                                self.logger.info(f"[H2] force_commit from repeated-SQL gate")
                                harness_trace["h2"].append({
                                    "round": current_round + 1,
                                    "reason": "repeated_sql_force_commit",
                                    "answers": force_ans,
                                })
                                finish_result = await self._do_commit(
                                    force_ans, entry, ground_truth, database,
                                    harness_runtime=harness_runtime,
                                    harness_trace=harness_trace,
                                    current_round=current_round,
                                )
                                if finish_result is not None:
                                    self._persist_harness_trace(session, harness_trace)
                                    session.inject(RewardHistoryItem(
                                        reward=1 if finish_result.get("is_correct") else 0,
                                        score=1 if finish_result.get("is_correct") else 0,
                                    ))
                                    return TaskSampleExecutionResult(
                                        status=SampleStatus.COMPLETED,
                                        result={**finish_result, "harness_trace": harness_trace},
                                    )
                                committed = True
                                break
                            else:
                                # Apply rewrites if any
                                if sql_check["sql"] != sql:
                                    self.logger.info(f"[H2] SQL rewritten: {sql_check['rule_hits']}")
                                    harness_trace["h2"].append({
                                        "round": current_round + 1,
                                        "reason": "sql_rewritten",
                                        "hits": [str(r) for r in sql_check["rule_hits"]],
                                        "before": sql[:200],
                                        "after": sql_check["sql"][:200],
                                    })
                                    sql = sql_check["sql"]
                                # Surface inline hints from pre_validate (e.g. INSERT partial columns).
                                for hit in sql_check.get("rule_hits", []):
                                    if isinstance(hit, dict) and hit.get("hint"):
                                        harness_trace["h2"].append({
                                            "round": current_round + 1,
                                            "reason": hit["rule"],
                                            "hint": hit["hint"][:200],
                                        })
                                        session.inject(ChatCompletionUserMessageParam(
                                            role='user',
                                            content=hit["hint"],
                                        ))

                        if (
                            harness_runtime is not None
                            and self.overlay_has_code_hooks
                            and self._code_hooks.get("on_before_action")
                        ):
                            before_ctx = _dbbench_code_context(
                                harness_runtime,
                                tool_name=function_name,
                                sql=sql,
                                remaining_rounds=remaining,
                                round_num=current_round + 1,
                                h2_result=h2_audit,
                            )
                            blocked_by_code = False
                            for hook_out in self._run_code_hooks("on_before_action", before_ctx, code_hook_nb):
                                if self._apply_before_tool_effect(
                                    session,
                                    hook_out,
                                    call_id,
                                    harness_trace,
                                    round_num=current_round + 1,
                                    tool_name=function_name,
                                ):
                                    blocked_by_code = True
                                    break
                            if blocked_by_code:
                                continue

                        # ── Execute SQL ────────────────────────────────────
                        try:
                            self.logger.info(f"Executing SQL: {sql}")
                            db_response = await asyncio.wait_for(database.execute(sql), 60)
                            self.logger.info(
                                f"DB response: {db_response[:100]}{'...' if len(db_response) > 100 else ''}"
                            )
                            if not db_response:
                                db_response = "No response from database."
                        except asyncio.TimeoutError:
                            self.logger.warning(f"Timeout executing SQL: {sql}")
                            db_response = "Error: SQL execution timed out."
                        except Exception as e:
                            self.logger.exception("Error executing query")
                            db_response = f"Error executing query: {e}"

                        # ── H1: update state ───────────────────────────────
                        if harness_runtime is not None:
                            harness_runtime.update_state_after_sql(sql, db_response)

                        session.inject(ChatCompletionToolMessageParam(
                            role='tool',
                            tool_call_id=call_id,
                            content=db_response,
                        ))

                        if (
                            harness_runtime is not None
                            and self.overlay_has_code_hooks
                            and self._code_hooks.get("on_post_step")
                        ):
                            post_ctx = _dbbench_code_context(
                                harness_runtime,
                                tool_name=function_name,
                                sql=sql,
                                db_response=db_response,
                                remaining_rounds=remaining - 1,
                                round_num=current_round + 1,
                                h2_result=h2_audit,
                            )
                            for hook_out in self._run_code_hooks("on_post_step", post_ctx, code_hook_nb):
                                self._apply_post_step_effect(
                                    session,
                                    hook_out,
                                    harness_trace,
                                    round_num=current_round + 1,
                                    tool_name=function_name,
                                )

                        # ── H4: post-step monitor ──────────────────────────
                        h4_audit_active = False
                        if (
                            harness_runtime is not None
                            and self.harness_config.enabled
                            and self.harness_config.h4_enabled
                        ):
                            h4_result = harness_runtime.post_step_monitor(remaining_rounds=remaining)
                            if h4_result.get("recovery_prompt") or h4_result.get("force_action"):
                                self.logger.info(f"[H4] {h4_result['audit_reason']}")
                                harness_trace["h4"].append({
                                    "round": current_round + 1,
                                    "reason": h4_result["audit_reason"],
                                    "budget_branch": h4_result.get("budget_branch"),
                                })
                                h4_audit_active = True
                                if h4_result.get("force_action"):
                                    harness_runtime.force_next_action = h4_result["force_action"]
                                if h4_result.get("recovery_prompt"):
                                    session.inject(ChatCompletionUserMessageParam(
                                        role='user',
                                        content=h4_result["recovery_prompt"],
                                    ))

                        # ── H4-E: state-driven per-step guidance ───────────
                        if (
                            harness_runtime is not None
                            and self.harness_config.enabled
                            and self.harness_config.h4_enabled
                        ):
                            hint = harness_runtime.step_guidance(
                                round_num=current_round,
                                h4_audit_active=h4_audit_active,
                            )
                            if hint:
                                harness_trace["h4"].append({
                                    "round": current_round + 1,
                                    "sub": "h4e_step_guidance",
                                    "hint": hint[:200],
                                })
                                session.inject(ChatCompletionUserMessageParam(
                                    role='user',
                                    content=hint,
                                ))

                    elif function_name == "commit_final_answer":
                        raw_answers = list(arguments.values())[0] if arguments else []
                        if not isinstance(raw_answers, list):
                            raw_answers = [str(raw_answers)]

                        if (
                            harness_runtime is not None
                            and self.overlay_has_code_hooks
                            and self._code_hooks.get("on_before_action")
                        ):
                            before_ctx = _dbbench_code_context(
                                harness_runtime,
                                tool_name=function_name,
                                answers=raw_answers,
                                remaining_rounds=remaining,
                                round_num=current_round + 1,
                            )
                            blocked_by_code = False
                            for hook_out in self._run_code_hooks("on_before_action", before_ctx, code_hook_nb):
                                if self._apply_before_tool_effect(
                                    session,
                                    hook_out,
                                    call_id,
                                    harness_trace,
                                    round_num=current_round + 1,
                                    tool_name=function_name,
                                ):
                                    blocked_by_code = True
                                    break
                            if blocked_by_code:
                                continue

                        # ── H2: commit gate + answer normalisation ─────────
                        if (
                            harness_runtime is not None
                            and self.harness_config.enabled
                            and self.harness_config.h2_enabled
                        ):
                            gate = harness_runtime.gate_commit(raw_answers)
                            harness_trace["h2_commit"].append({
                                "round": current_round + 1,
                                "action": gate["action"],
                                "blocked_reason": gate.get("blocked_reason", ""),
                                "rule_hits": gate["rule_hits"],
                                "normalize": gate.get("normalize_audit"),
                            })
                            if gate["action"] == "block":
                                self.logger.info(f"[H2 commit] blocked: {gate['blocked_reason']}")
                                # Prefer recovery_prompt from the gate (richer message) when present.
                                _static_block_msgs = {
                                    "empty_answers_before_query": (
                                        "Harness: cannot commit empty answers. Run execute_sql first "
                                        "to retrieve the answer from the database."
                                    ),
                                    "commit_before_any_sql": (
                                        "Harness: you haven't run any SQL yet. Run execute_sql first "
                                        "to query the database before committing."
                                    ),
                                    "mutation_not_executed": (
                                        "Harness: for INSERT/UPDATE/DELETE tasks, you MUST actually "
                                        "execute the mutation SQL before committing — the answer is "
                                        "verified by a table hash."
                                    ),
                                    "give_up_text_on_mutation": (
                                        "Harness: do not commit explanatory text for mutation tasks. "
                                        "The answer is verified by a table hash — you MUST run the "
                                        "INSERT/UPDATE/DELETE SQL and succeed before committing. "
                                        "Try `SELECT * FROM `tbl` LIMIT 5;` to find the target row."
                                    ),
                                }
                                block_msg = (
                                    gate.get("recovery_prompt")
                                    or _static_block_msgs.get(gate["blocked_reason"],
                                       f"Harness: commit blocked ({gate['blocked_reason']}).")
                                )
                                session.inject(ChatCompletionToolMessageParam(
                                    role='tool',
                                    tool_call_id=call_id,
                                    content=block_msg,
                                ))
                                continue
                            else:
                                raw_answers = gate["answers"]

                        # ── Evaluate ───────────────────────────────────────
                        finish_result = await self._do_commit(
                            raw_answers, entry, ground_truth, database,
                            harness_runtime=harness_runtime,
                            harness_trace=harness_trace,
                            current_round=current_round,
                        )
                        if finish_result is not None:
                            self._persist_harness_trace(session, harness_trace)
                            session.inject(RewardHistoryItem(
                                reward=1 if finish_result.get("is_correct") else 0,
                                score=1 if finish_result.get("is_correct") else 0,
                            ))
                            return TaskSampleExecutionResult(
                                status=SampleStatus.COMPLETED,
                                result={**finish_result, "harness_trace": harness_trace},
                            )
                        committed = True
                        break

                    else:
                        self.logger.warning(f"Invalid function call: {function_name}")
                        session.inject(ChatCompletionToolMessageParam(
                            role='tool',
                            tool_call_id=call_id,
                            content="Invalid function call. Please use execute_sql or commit_final_answer.",
                        ))

                    if committed:
                        break

            # Round limit reached
            self._persist_harness_trace(session, harness_trace)
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_LIMIT_REACHED,
                result={"harness_trace": harness_trace},
            )

        except AgentCancelledException:
            self._persist_harness_trace(session, harness_trace)
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
        except Exception:
            self.logger.exception("Error during task execution")
            self._persist_harness_trace(session, harness_trace)
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(status=SampleStatus.TASK_ERROR)
        finally:
            if database:
                try:
                    await database.delete()
                except Exception:
                    self.logger.warning("Error during database cleanup", exc_info=True)

    async def _do_commit(
        self,
        answers: List[str],
        entry: Dict[str, Any],
        ground_truth: Any,
        database: Database,
        harness_runtime: Optional[DBBenchHarnessRuntime],
        harness_trace: Dict[str, List[Any]],
        current_round: int,
    ) -> Optional[Dict[str, Any]]:
        """Run the final evaluation and return a result dict, or None on internal error."""
        if not answers:
            self.logger.warning("Empty answer submitted to _do_commit")
        else:
            self.logger.info(
                f"Final answer submitted: {str(answers)[:100]}{'...' if len(str(answers)) > 100 else ''}"
            )

        std_sql = entry.get("sql", {}).get("query")
        db_type = database.type

        if entry["type"][0] in ("INSERT", "DELETE", "UPDATE"):
            self.logger.info(f"Calculating table hash ({db_type})...")
            if db_type == TYPE_SQLITE:
                self.logger.warning("Table hash calculation for SQLite not implemented.")
                answer_to_compare = "SQLite hash not implemented"
            else:
                answer_to_compare = await DBResultProcessor.calculate_tables_hash_async(database, entry)

            if ground_truth == "":
                answer_db: Optional[Database] = None
                try:
                    answer_db = MySQLDatabase(self.env_controller)
                    await answer_db.initialize()
                    init_sql = self._build_init_sql(entry)
                    await answer_db.batch_execute(init_sql)
                    await answer_db.execute(std_sql)
                    ground_truth = await DBResultProcessor.calculate_tables_hash_async(answer_db, entry)
                finally:
                    if answer_db:
                        await answer_db.delete()
        else:
            # For query tasks, the submitted answers list is used as-is.
            answer_to_compare = answers

        self.logger.info(
            f"Final Answer: {str(answer_to_compare)[:100]}{'...' if len(str(answer_to_compare)) > 100 else ''}"
        )
        self.logger.info(
            f"Ground Truth: {str(ground_truth)[:100]}{'...' if len(str(ground_truth)) > 100 else ''}"
        )
        is_correct = DBResultProcessor.compare_results(answer_to_compare, ground_truth, entry["type"][0])
        self.logger.info(f"Correct: {is_correct}")

        return {
            "is_correct": is_correct,
            "answer": answers,
            "ground_truth": ground_truth,
            "std_sql": std_sql,
            "type": entry["type"][0],
        }

    @staticmethod
    def _persist_harness_trace(session: Session, harness_trace: Dict[str, List[Any]]) -> None:
        """Persist harness_trace via a [HARNESS_TRACE_V1] tagged user message."""
        try:
            session.inject(ChatCompletionUserMessageParam(
                role='user',
                content='[HARNESS_TRACE_V1]\n' + json.dumps(harness_trace, ensure_ascii=False),
            ))
        except Exception as e:
            logging.warning(f"Failed to inject harness_trace audit message: {e}")

    def calculate_overall(self, results) -> dict:
        total = len(results)
        if not total:
            return {"total": 0, "correct": 0, "acc": 0}
        correct = sum(
            1 for r in results
            if isinstance(getattr(r, 'result', None), dict)
            and float(r.result.get('reward', 0)) >= 1.0
        )
        return {
            "total": total,
            "correct": correct,
            "wrong": total - correct,
            "acc": correct / total,
        }

    @staticmethod
    def _sanitize_identifier(name: str, fallback_prefix: str = "col") -> str:
        """Normalize for MySQL: replace escape sequences, truncate to 64-char limit."""
        name = name.replace('\\n', ' ').replace('\\t', ' ').replace('\\r', ' ')
        name = name.strip()
        if not name:
            name = fallback_prefix
        return name[:64]

    @staticmethod
    def _build_init_sql(entry: dict) -> List[Union[str, Tuple[str, Sequence[str]]]]:
        """Builds initialization SQL for MySQL."""
        tables = entry["table"] if isinstance(entry["table"], list) else [entry["table"]]
        final_sql = []
        for table_idx, table in enumerate(tables):
            raw_name = table["table_name"]
            name = DBBenchTask._sanitize_identifier(raw_name, fallback_prefix=f"table_{table_idx}")
            cols = table["table_info"]["columns"]
            seen: dict = {}
            sanitized_col_names = []
            for col_idx, c in enumerate(cols):
                sname = DBBenchTask._sanitize_identifier(c['name'], fallback_prefix=f"col_{col_idx}")
                base = sname
                counter = 1
                while sname in seen:
                    sname = f"{base[:61]}_{counter}"
                    counter += 1
                seen[sname] = True
                sanitized_col_names.append(sname)
            columns = ",".join([f"`{sname}` TEXT" for sname in sanitized_col_names])
            column_names = ",".join([f"`{sname}`" for sname in sanitized_col_names])
            items = []
            items_data = ()
            for row in table["table_info"]["rows"]:
                item = "(" + ",".join(["%s"] * len(row)) + ")"
                items_data += tuple(str(col) for col in row)
                items.append(item)
            items_str = ",".join(items)
            final_sql.append(f'CREATE TABLE IF NOT EXISTS `{name}` ({columns})')
            final_sql.append((
                f'INSERT INTO `{name}` ({column_names}) VALUES {items_str}',
                items_data
            ))
        return final_sql
