import copy
import hashlib
import json
import logging
import re
from dataclasses import fields
from typing import Dict, List, Any, Optional
from uuid import uuid4

from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (AgentCancelledException,
                                    RewardHistoryItem,
                                    SampleStatus,
                                    TaskOutput,
                                    TaskSampleExecutionResult)
from openai.types.chat import (ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

from src.server.harness import (
    WebShopHarnessConfig,
    WebShopHarnessRuntime,
    compile_hook,
    patch_webshop_tool_descriptions,
    run_hook,
)
from src.server.harness.dsl import (
    apply_harness_rules,
    normalize_harness_overlay,
    overlay_cold_start_hints,
)

prompt_with_max_turn = """You are a web shopping agent. Follow the task instruction to find and buy the correct product.

CRITICAL RULES:
1. You MUST call a tool EVERY turn. NEVER respond with only text — always call search_action or click_action.
2. On search results: read product titles carefully. Click the product whose title best matches ALL key terms in the instruction (brand name, product type, specific features).
3. On a product page: select ALL required attributes (color, size, etc.) THEN click 'buy now' immediately.
4. After selecting attributes, click 'buy now' right away. Do NOT hesitate or deliberate — just buy.
5. If the Hint says "All checked" or "All attributes selected", your ONLY correct action is click 'buy now'.
6. Keywords in search should be the product name and key features, NOT prices or filler words.
7. The click value MUST be exactly one of the available clickable values.
"""


def _extract_instruction(observation: str) -> str:
    """Parse the task instruction out of the initial WebShop observation.

    WebShop text-mode format: "WebShop [SEP] Instruction: [SEP] <text> [SEP] Search"
    """
    # Primary: [SEP]-delimited format used by WebShop text env
    m = re.search(r"Instruction:\s*\[SEP\]\s*(.+?)\s*\[SEP\]", observation, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # Fallback: newline-delimited format
    m = re.search(r"Instruction:\s*\n(.+?)(?:\n\n|\[|$)", observation, re.DOTALL)
    if m:
        return m.group(1).strip()
    return observation[:300]


def _canonical_manifest_value(value: Any) -> Any:
    """Convert WebShop state into a stable, JSON-serializable value."""
    if isinstance(value, dict):
        return [
            [str(key), _canonical_manifest_value(item)]
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
    if isinstance(value, (list, tuple)):
        return [_canonical_manifest_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical_items = [_canonical_manifest_value(item) for item in value]
        return sorted(
            canonical_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _manifest_sha256(value: Any) -> str:
    canonical = json.dumps(
        _canonical_manifest_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _webshop_task_manifest(
    server: Any,
    *,
    index: int,
    instruction: str,
    goal_seed: int,
    product_prices_sha256: str,
) -> Dict[str, Any]:
    goal = server.goals[index]
    payload = {
        "protocol": "webshop_task_manifest_v1",
        "goal_seed": int(goal_seed),
        "index": int(index),
        "instruction_sha256": _manifest_sha256(instruction),
        "goal_sha256": _manifest_sha256(goal),
        "product_prices_sha256": product_prices_sha256,
    }
    return {**payload, "sha256": _manifest_sha256(payload)}


def _parse_available_actions(available_actions) -> tuple:
    """Return (has_search_bar: bool, clickables: list[str]) from env response."""
    if isinstance(available_actions, dict):
        return (
            bool(available_actions.get("has_search_bar", False)),
            list(available_actions.get("clickables", [])),
        )
    return False, []


def _page_type_for_dsl(page_type: str) -> str:
    return (page_type or "UNKNOWN").lower()


def _effect_action_for_webshop(effect_action: str, tool_name: str) -> str:
    action = (effect_action or "").strip()
    if "[" in action and action.endswith("]"):
        return action
    if tool_name == "search_action":
        return f"search[{action}]"
    return f"click[{action}]"


def _webshop_rule_context(
    harness_runtime: WebShopHarnessRuntime,
    tool_name: str = "",
    raw_value: str = "",
    h2_result: Optional[Dict[str, Any]] = None,
    has_search_bar: bool = False,
    clickables: Optional[List[str]] = None,
    remaining_steps: Optional[int] = None,
) -> Dict[str, Any]:
    clickables = clickables or []
    clickables_l = [str(c).lower() for c in clickables]
    state = harness_runtime.page_state
    req = harness_runtime.requirements
    value_norm = str(raw_value or "").strip().lower()
    final_action = (h2_result or {}).get("action")
    duplicate_search_count = harness_runtime._duplicate_search_count()
    buy_now_precheck = None
    if tool_name == "click_action" and value_norm == "buy now":
        buy_now_precheck = harness_runtime._buy_now_precheck(
            clickables,
            record_block=False,
            respect_safety_valve=False,
        )
    price_over_budget = (
        state.current_price is not None
        and req is not None
        and req.price_max is not None
        and state.current_price > req.price_max * (1 + harness_runtime.config.price_tolerance)
    )
    return {
        "action": {
            "tool": tool_name,
            "value": raw_value,
            "value_normalized": value_norm,
            "final_action": final_action,
        },
        "state": {
            "page_type": _page_type_for_dsl(state.page_type),
            "has_search_bar": has_search_bar,
            "clickables": clickables_l,
            "buy_now_available": "buy now" in clickables_l,
            "back_to_search_count": state.back_to_search_count,
            "duplicate_search_count": duplicate_search_count,
            "product_stall_turns": state._stall_turns,
            "same_click_count": harness_runtime._repeat_click_count,
            "remaining_steps": remaining_steps if remaining_steps is not None else 999,
            "current_price": state.current_price,
            "price_max": req.price_max if req else None,
        },
        "task": {
            "task_type": req.task_type if req else None,
            "required_color": req.color if req else None,
            "required_size": req.size if req else None,
            "required_material": req.material if req else None,
        },
        "predicates": {
            "required_options_unselected": bool(buy_now_precheck),
            "product_price_over_budget": bool(price_over_budget),
            "same_click_repeated": harness_runtime._repeat_click_count >= harness_runtime.config.h2_repeat_click_block_after,
            "duplicate_search_repeated": duplicate_search_count >= 2,
            "search_loop_detected": state.back_to_search_count >= 2 or duplicate_search_count >= 2,
            "product_page_stalled": state._stall_turns >= 2,
            "buy_now_available": "buy now" in clickables_l,
            "search_not_available": not has_search_bar,
            "action_not_admissible": bool((h2_result or {}).get("blocked")),
        },
    }


def _webshop_code_context(
    harness_runtime: WebShopHarnessRuntime,
    *,
    tool_name: str = "",
    raw_value: str = "",
    h2_result: Optional[Dict[str, Any]] = None,
    has_search_bar: bool = False,
    clickables: Optional[List[str]] = None,
    remaining_steps: Optional[int] = None,
    step: int = 0,
    max_step: int = 0,
    observation: str = "",
) -> Dict[str, Any]:
    ctx = _webshop_rule_context(
        harness_runtime=harness_runtime,
        tool_name=tool_name,
        raw_value=raw_value,
        h2_result=h2_result,
        has_search_bar=has_search_bar,
        clickables=clickables,
        remaining_steps=remaining_steps,
    )
    ctx.update(
        {
            "observation": observation or "",
            "step": int(step),
            "max_step": int(max_step),
            "remaining_steps": remaining_steps if remaining_steps is not None else 999,
            "webshop": {
                "search_queries": list(harness_runtime.page_state.search_queries),
                "asins_visited": list(harness_runtime.page_state.asins_visited),
                "current_asin": harness_runtime.page_state.current_asin,
                "selected_attributes": dict(harness_runtime.page_state.selected_attributes),
                "attribute_options": {
                    key: list(value)
                    for key, value in harness_runtime.page_state.attribute_options.items()
                },
            },
        }
    )
    return ctx


class WebShop(Task):
    def __init__(self, tools=None, **configs):
        # Extract harness config params before passing configs to super()
        self.goal_seed = int(configs.pop("goal_seed", 233))
        raw_task_ids = configs.pop("task_ids", None)
        self.task_ids = None if raw_task_ids is None else [int(item) for item in raw_task_ids]
        self.harness_overlay = normalize_harness_overlay(
            configs.pop("harness_overlay", {}),
            benchmark="webshop",
        )
        enabled = configs.pop("enabled", False)
        h2 = configs.pop("h2", True)
        h3 = configs.pop("h3", True)
        h4 = configs.pop("h4", True)
        h5 = configs.pop("h5", True)
        h5_top_k = configs.pop("h5_top_k", 2)
        h5_score_threshold = configs.pop("h5_score_threshold", 0.0)
        config_kwargs = dict(
            enabled=bool(enabled),
            h2_enabled=bool(h2),
            h3_enabled=bool(h3),
            h4_enabled=bool(h4),
            h5_enabled=bool(h5),
            h5_top_k=int(h5_top_k),
            h5_score_threshold=float(h5_score_threshold),
        )
        for item in fields(WebShopHarnessConfig):
            if item.name in configs:
                config_kwargs[item.name] = configs.pop(item.name)
        self.harness_config = WebShopHarnessConfig(**config_kwargs)
        self.overlay_has_skills = bool(self.harness_overlay.get("skills"))
        self.overlay_has_guard_rules = bool(self.harness_overlay.get("guard_rules"))
        self.overlay_has_recovery_rules = bool(self.harness_overlay.get("recovery_rules"))
        self.overlay_has_code_hooks = bool(self.harness_overlay.get("code_hooks"))
        self._code_hooks: Dict[str, List[Any]] = {}
        for item in self.harness_overlay.get("code_hooks", []) or []:
            hook_name = str(item.get("hook") or "")
            code = item.get("code")
            if not hook_name or not isinstance(code, str):
                continue
            try:
                self._code_hooks.setdefault(hook_name, []).append(
                    compile_hook(code, benchmark="webshop")
                )
            except Exception:
                continue
        self.harness_substrate_enabled = bool(
            self.harness_config.enabled
            or self.overlay_has_skills
            or self.overlay_has_guard_rules
            or self.overlay_has_recovery_rules
            or self.overlay_has_code_hooks
        )
        if self.harness_config.enabled and self.harness_config.h3_enabled:
            tools = patch_webshop_tool_descriptions(tools)

        super().__init__(**configs)
        self.logger = logging.getLogger(__name__)
        self.ranging = (configs.pop("start", 0), configs.pop("end", 500))
        self.shuffle_seed = configs.pop("shuffle_seed", None)
        self.sample_size = configs.pop("sample_size", None)
        print(
            f"[MOUNT_CHECK] WebShop mapped source active: range={self.ranging}, sample_size={self.sample_size}",
            flush=True,
        )
        self.logger.warning(
            "[MOUNT_CHECK] WebShop mapped source loaded: start=%s end=%s sample_size=%s",
            self.ranging[0],
            self.ranging[1],
            self.sample_size,
        )
        self.logger.info('Initializing WebShop environment...')
        self.server = WebAgentTextEnv(
            observation_mode="text",
            human_goals=True,
            goal_seed=self.goal_seed,
        ).server
        self._product_prices_sha256 = _manifest_sha256(self.server.product_prices)
        self._base_tools = copy.deepcopy(tools)
        self.tools = copy.deepcopy(self._base_tools)
        self.max_rounds = configs.get('round', 20)

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
    def _apply_before_action_effect(
        session: Session,
        effect: Dict[str, Any],
        h2_result: Dict[str, Any],
        action: str,
        reason: str,
        tool_name: str,
    ) -> tuple[str, bool]:
        kind = effect.get("kind")
        if kind == "block_and_prompt":
            session.inject(ChatCompletionUserMessageParam(
                role='user',
                content=effect.get("message") or "Harness blocked this action. Choose a better action.",
            ))
            session.inject(RewardHistoryItem(reward=0, score=0))
            return action, True
        if kind in {"force_action", "rewrite_action"} and effect.get("action"):
            action = _effect_action_for_webshop(effect.get("action", ""), tool_name)
            h2_result["action"] = action
            h2_result["blocked"] = False
            h2_result["reason"] = reason
            if effect.get("message"):
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content=effect["message"],
                ))
        return action, False

    @staticmethod
    def _apply_post_step_effect(
        session: Session,
        harness_runtime: WebShopHarnessRuntime,
        effect: Dict[str, Any],
    ) -> None:
        if effect.get("kind") == "inject_hint" and effect.get("message"):
            session.inject(ChatCompletionUserMessageParam(
                role='user',
                content=effect["message"],
            ))
        elif effect.get("kind") == "force_action" and effect.get("action"):
            harness_runtime.force_next_action = _effect_action_for_webshop(
                effect["action"],
                "click_action",
            )
            if effect.get("message"):
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content=effect["message"],
                ))

    def get_indices(self) -> List[Any]:
        if self.task_ids is not None:
            return list(self.task_ids)
        indices = list(range(*self.ranging))
        if self.shuffle_seed is not None:
            import random
            random.Random(self.shuffle_seed).shuffle(indices)
        if self.sample_size is not None:
            indices = indices[:self.sample_size]
        return indices

    def sync_start_sample(self, index: int, session: Session) -> TaskSampleExecutionResult:
        print(f"[MOUNT_CHECK][SAMPLE_START] webshop index={index}", flush=True)
        self.logger.warning("[MOUNT_CHECK][SAMPLE_START] webshop index=%s", index)
        history = []

        env = WebAgentTextEnv(
            observation_mode="text",
            server=self.server,
            human_goals=True,
            session_prefix=str(uuid4()) + '-'
        )
        task_manifest = None
        try:
            env.reset(index)
            task_manifest = _webshop_task_manifest(
                self.server,
                index=index,
                instruction=_extract_instruction(env.observation),
                goal_seed=self.goal_seed,
                product_prices_sha256=self._product_prices_sha256,
            )
            # Harness: initialise per-episode runtime
            harness_runtime = None
            cold_skills = []
            if self.harness_substrate_enabled:
                harness_runtime = WebShopHarnessRuntime(config=self.harness_config)
                harness_runtime.init_task(_extract_instruction(env.observation))
            code_hook_nb: Dict[str, Any] = {}
            episode_tools = copy.deepcopy(self._base_tools)

            # Cold-start hints: built-in H5 and overlay skills are independent.
            if harness_runtime:
                if self.harness_config.enabled and self.harness_config.h5_enabled:
                    cold_skills.extend(harness_runtime.cold_start_skill_hints())
                if self.overlay_has_skills:
                    cold_skills.extend(
                        overlay_cold_start_hints(
                            self.harness_overlay,
                            task_type=harness_runtime.requirements.task_type if harness_runtime.requirements else None,
                        )
                    )
                if self.overlay_has_code_hooks and self._code_hooks.get("on_init"):
                    code_ctx = _webshop_code_context(
                        harness_runtime=harness_runtime,
                        observation=env.observation,
                        remaining_steps=self.max_rounds,
                        step=0,
                        max_step=self.max_rounds,
                    )
                    for hook_out in self._run_code_hooks("on_init", code_ctx, code_hook_nb):
                        for skill in hook_out.get("skills", []) or []:
                            if isinstance(skill, dict):
                                cold_skills.append(skill)
                            elif isinstance(skill, str):
                                cold_skills.append({"text": skill})
                        if hook_out.get("tool_hint"):
                            episode_tools = patch_webshop_tool_descriptions(
                                episode_tools,
                                extra_hint=hook_out["tool_hint"],
                            )

            session.set_tools(episode_tools)

            system_prompt = prompt_with_max_turn
            if cold_skills:
                skill_lines = [f"- {item['text']}" for item in cold_skills]
                system_prompt += (
                    "\n\nSome tips that may help for this task:\n"
                    + "\n".join(skill_lines)
                )
            session.inject(ChatCompletionSystemMessageParam(
                role='system',
                content=system_prompt
            ))

            action = None
            observation = env.observation
            reward = 0
            call_id = None
            _no_tool_consecutive = 0

            for j in range(self.max_rounds):
                available_actions = env.get_available_actions()
                has_search_bar, clickables = _parse_available_actions(available_actions)

                if j == 0:
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content=f'The initial observation:\n{observation}\n\nAvailable Actions:\n{available_actions}'
                    ))
                    # H4-E: initial search hint on step 0
                    if harness_runtime and self.harness_config.enabled and self.harness_config.h4_enabled:
                        first_hint = harness_runtime.step_guidance(
                            step_num=0,
                            max_steps=self.max_rounds,
                            observation=observation,
                            has_search_bar=has_search_bar,
                            clickables=clickables,
                        )
                        if first_hint:
                            session.inject(ChatCompletionUserMessageParam(
                                role='user',
                                content=first_hint,
                            ))
                else:
                    if action is None:
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=f'Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}'
                        ))
                    else:
                        session.inject(ChatCompletionToolMessageParam(
                            role='tool',
                            content=f'Action: {action}\n\nObservation:\n{observation}\n\nAvailable Actions:\n{available_actions}',
                            tool_call_id=call_id
                            ))

                if harness_runtime and self.overlay_has_code_hooks and self._code_hooks.get("make_pre_hint"):
                    pre_ctx = _webshop_code_context(
                        harness_runtime=harness_runtime,
                        has_search_bar=has_search_bar,
                        clickables=clickables,
                        observation=observation,
                        remaining_steps=self.max_rounds - j,
                        step=j,
                        max_step=self.max_rounds,
                    )
                    for hook_out in self._run_code_hooks("make_pre_hint", pre_ctx, code_hook_nb):
                        message = hook_out.get("message")
                        if message:
                            session.inject(ChatCompletionUserMessageParam(role='user', content=message))

                response = session.sync_action()

                tool_calls = []
                for message in response.messages:
                    tool_calls.extend(message.get('tool_calls', []) or [])

                finish_reason = SampleStatus.COMPLETED
                if not tool_calls:
                    _no_tool_consecutive += 1
                    action = None

                    # Aggressive force: on product page with buy-now available,
                    # force buy-now after just 1 text-only turn (saves 265+ wasted turns).
                    buy_now_available = "buy now" in clickables
                    if self.harness_config.enabled and harness_runtime and buy_now_available and _no_tool_consecutive >= 1:
                        # Force buy now immediately — the agent is deliberating uselessly
                        harness_runtime.force_next_action = "click[buy now]"
                        no_exec_msg = (
                            "You must call a tool! Since 'buy now' is available, "
                            "buying now. Call click_action with 'buy now' next turn."
                        )
                    elif self.harness_config.enabled and harness_runtime and _no_tool_consecutive >= 2 and "back to search" in clickables:
                        harness_runtime.force_next_action = "click[back to search]"
                        no_exec_msg = (
                            "You must call a tool! Forcing back to search."
                        )
                    elif self.harness_config.enabled and harness_runtime and _no_tool_consecutive >= 1:
                        # Immediate directive message
                        if has_search_bar:
                            no_exec_msg = (
                                "You must call a tool! Use search_action to search for the product."
                            )
                        elif "back to search" in clickables:
                            no_exec_msg = (
                                "You must call a tool! Click 'back to search' or click a product."
                            )
                        else:
                            no_exec_msg = (
                                "You must call a tool! Click one of the available options."
                            )
                    else:
                        no_exec_msg = "You must call a tool! NEVER respond with only text."
                    observation = no_exec_msg
                else:
                    _no_tool_consecutive = 0
                    action = None
                    try:
                        tool_call = tool_calls[0]
                        func_name = tool_call["function"]["name"]
                        arguments = tool_call["function"]["arguments"]
                        arguments = json.loads(arguments)
                        arguments = list(arguments.values())
                        call_id = tool_call["id"]
                        raw_value = arguments[0]

                        if func_name == "search_action":
                            default_action = f"search[{raw_value}]"
                        elif func_name == "click_action":
                            default_action = f"click[{raw_value}]"
                        else:
                            default_action = None

                        h2_result = {
                            "action": default_action,
                            "blocked": False,
                            "reason": "",
                            "canonicalized": False,
                        }

                        builtin_h2_enabled = (
                            self.harness_config.enabled
                            and self.harness_config.h2_enabled
                        )
                        if harness_runtime and not builtin_h2_enabled and harness_runtime.force_next_action:
                            forced = harness_runtime.force_next_action
                            harness_runtime.force_next_action = None
                            h2_result["action"] = forced
                            h2_result["reason"] = "overlay_force_next_action"
                            h2_result["canonicalized"] = True

                        if harness_runtime and builtin_h2_enabled:
                            h2_result = harness_runtime.pre_validate_action(
                                tool_name=func_name,
                                raw_value=raw_value,
                                has_search_bar=has_search_bar,
                                clickables=clickables,
                            )
                        if harness_runtime and self.overlay_has_guard_rules:
                            overlay_guard = apply_harness_rules(
                                self.harness_overlay.get("guard_rules", []),
                                _webshop_rule_context(
                                    harness_runtime=harness_runtime,
                                    tool_name=func_name,
                                    raw_value=raw_value,
                                    h2_result=h2_result,
                                    has_search_bar=has_search_bar,
                                    clickables=clickables,
                                    remaining_steps=self.max_rounds - j,
                                ),
                                trigger="before_action",
                            )
                            if overlay_guard:
                                effect = overlay_guard.get("effect", {})
                                kind = effect.get("kind")
                                if kind == "block_and_prompt":
                                    session.inject(ChatCompletionUserMessageParam(
                                        role='user',
                                        content=effect.get("message") or "Harness blocked this action. Choose a better action.",
                                    ))
                                    session.inject(RewardHistoryItem(reward=0, score=0))
                                    continue
                                if kind in {"force_action", "rewrite_action"}:
                                    action = _effect_action_for_webshop(effect.get("action", ""), func_name)
                                    if effect.get("message"):
                                        session.inject(ChatCompletionUserMessageParam(
                                            role='user',
                                            content=effect["message"],
                                        ))
                                    h2_result["action"] = action
                                    h2_result["blocked"] = False
                                    h2_result["reason"] = f"overlay_{kind}:{overlay_guard.get('rule_id')}"
                        if harness_runtime and self.overlay_has_code_hooks and self._code_hooks.get("on_before_action"):
                            code_ctx = _webshop_code_context(
                                harness_runtime=harness_runtime,
                                tool_name=func_name,
                                raw_value=raw_value,
                                h2_result=h2_result,
                                has_search_bar=has_search_bar,
                                clickables=clickables,
                                observation=observation,
                                remaining_steps=self.max_rounds - j,
                                step=j,
                                max_step=self.max_rounds,
                            )
                            blocked_by_code_any = False
                            for hook_out in self._run_code_hooks("on_before_action", code_ctx, code_hook_nb):
                                action, blocked_by_code = self._apply_before_action_effect(
                                    session,
                                    hook_out,
                                    h2_result,
                                    h2_result["action"],
                                    f"code_hook_{hook_out.get('kind')}",
                                    func_name,
                                )
                                if blocked_by_code:
                                    blocked_by_code_any = True
                                    break
                            if blocked_by_code_any:
                                continue
                        if h2_result["blocked"]:
                            reason = h2_result["reason"]
                            # Use custom block_message if available (e.g. buy-now pre-check)
                            if "block_message" in h2_result:
                                block_msg = h2_result["block_message"]
                            elif "search_not_available" in reason:
                                block_msg = "Search is not available on this page. Click one of the available options."
                            elif "repeat_click" in reason:
                                block_msg = f"You already clicked '{raw_value}'. Choose a different action."
                            else:
                                block_msg = f"Invalid action: {reason}. Please choose a valid action from available options."
                            session.inject(ChatCompletionUserMessageParam(
                                role='user',
                                content=block_msg,
                            ))
                            session.inject(RewardHistoryItem(reward=0, score=0))
                            continue
                        action = h2_result["action"]
                    except:
                        self.logger.warning(f'Error processing tool call. {tool_calls=}', exc_info=True)
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=f"No valid tool call found from agent."
                        ))
                        session.inject(RewardHistoryItem(reward=0, score=0))
                        continue

                history.append(
                    {
                        "observation": observation,
                        "available_actions": available_actions,
                        "response": response,
                        "action": action,
                    }
                )

                if not action:
                    reward = 0
                    done = False
                    round_reward = 0
                else:
                    observation, reward, done, info = env.step(action)
                    round_reward = reward

                    if harness_runtime:
                        # Get post-step available actions for harness checks
                        post_available = env.get_available_actions()
                        post_has_sb, post_clickables = _parse_available_actions(post_available)

                        # H1: update page state
                        harness_runtime.update_state(action, observation, post_has_sb, post_clickables)

                        builtin_h4_enabled = (
                            self.harness_config.enabled
                            and self.harness_config.h4_enabled
                        )
                        if builtin_h4_enabled:
                            # H4: shopping monitor
                            h4_result = harness_runtime.post_step_monitor(
                                action, observation, post_has_sb, post_clickables
                            )
                            if h4_result.get("recovery_prompt"):
                                session.inject(ChatCompletionUserMessageParam(
                                    role='user',
                                    content=h4_result["recovery_prompt"],
                                ))

                        if self.overlay_has_recovery_rules:
                            overlay_recovery = apply_harness_rules(
                                self.harness_overlay.get("recovery_rules", []),
                                _webshop_rule_context(
                                    harness_runtime=harness_runtime,
                                    tool_name="",
                                    raw_value="",
                                    h2_result=None,
                                    has_search_bar=post_has_sb,
                                    clickables=post_clickables,
                                    remaining_steps=self.max_rounds - j - 1,
                                ),
                                trigger="post_step",
                            )
                            if overlay_recovery:
                                effect = overlay_recovery.get("effect", {})
                                if effect.get("kind") == "inject_hint" and effect.get("message"):
                                    session.inject(ChatCompletionUserMessageParam(
                                        role='user',
                                        content=effect["message"],
                                    ))
                                elif effect.get("kind") == "force_action" and effect.get("action"):
                                    harness_runtime.force_next_action = _effect_action_for_webshop(
                                        effect["action"], "click_action"
                                    )
                                    if effect.get("message"):
                                        session.inject(ChatCompletionUserMessageParam(
                                            role='user',
                                            content=effect["message"],
                                        ))
                        if self.overlay_has_code_hooks and self._code_hooks.get("on_post_step"):
                            code_ctx = _webshop_code_context(
                                harness_runtime=harness_runtime,
                                tool_name="",
                                raw_value="",
                                h2_result=None,
                                has_search_bar=post_has_sb,
                                clickables=post_clickables,
                                observation=observation,
                                remaining_steps=self.max_rounds - j - 1,
                                step=j,
                                max_step=self.max_rounds,
                            )
                            for hook_out in self._run_code_hooks("on_post_step", code_ctx, code_hook_nb):
                                self._apply_post_step_effect(session, harness_runtime, hook_out)

                        if builtin_h4_enabled:
                            # H4-E: state-driven per-step guidance (page state + requirements)
                            h4e_hint = harness_runtime.step_guidance(
                                step_num=j + 1,
                                max_steps=self.max_rounds,
                                observation=observation,
                                has_search_bar=post_has_sb,
                                clickables=post_clickables,
                            )
                            if h4e_hint:
                                session.inject(ChatCompletionUserMessageParam(
                                    role='user',
                                    content=h4e_hint,
                                ))

                            # H4-D: step-budget management
                            budget_result = harness_runtime.budget_check(
                                remaining_steps=self.max_rounds - j - 1,
                                clickables=post_clickables,
                            )
                            if budget_result.get("force_action"):
                                harness_runtime.force_next_action = budget_result["force_action"]
                            if budget_result.get("hint"):
                                session.inject(ChatCompletionUserMessageParam(
                                    role='user',
                                    content=budget_result["hint"],
                                ))

                history[-1]["reward"] = reward
                history[-1]["done"] = done
                rewardhistory = RewardHistoryItem(reward=round_reward, score=round_reward)
                session.inject(rewardhistory)
                if done:
                    break
            else:
                finish_reason = SampleStatus.TASK_LIMIT_REACHED
                rewardhistory = RewardHistoryItem(reward=0, score=0)
                session.inject(rewardhistory)
                if call_id:
                    session.inject(ChatCompletionToolMessageParam(
                        role='tool',
                        content='Task limit reached.',
                        tool_call_id=call_id
                    ))
                else:
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content='Task limit reached.'
                    ))

            return TaskSampleExecutionResult(
                status=finish_reason,
                result={
                    "reward": reward,
                    "history": history,
                    "webshop_task_manifest": task_manifest,
                },
            )
        except AgentCancelledException:
            session.inject(RewardHistoryItem(reward=0, score=0))
            return TaskSampleExecutionResult(
                status=SampleStatus.CANCELLED,
                result={
                    "reward": 0,
                    "history": history,
                    "webshop_task_manifest": task_manifest,
                },
            )
        except:
            self.logger.exception(f'Error during sample execution')
            return TaskSampleExecutionResult(
                status=SampleStatus.TASK_ERROR,
                result={
                    "reward": 0,
                    "history": history,
                    "webshop_task_manifest": task_manifest,
                },
            )
        finally:
            try:
                env.close()
            except:
                pass

    def calculate_overall(self, results: List[TaskOutput]) -> Dict:
        result_payloads = [x.result for x in results if x and isinstance(x.result, dict)]
        rewards = [x.get("reward") for x in result_payloads if x.get("reward") is not None]
        total = len(results)
        completed = len([x for x in results if x and x.status == SampleStatus.COMPLETED])
        success_at_1 = len([r for r in rewards if float(r) >= 1.0])
        average_reward = sum(rewards) / len(rewards) if rewards else 0
        usages = [x.get("token_usage", {}) for x in result_payloads]
        n_ep = len(usages) or 1
        total_prompt = sum(u.get("prompt_tokens", 0) for u in usages if u)
        total_completion = sum(u.get("completion_tokens", 0) for u in usages if u)
        total_tokens = sum(u.get("total_tokens", 0) for u in usages if u)
        return {
            "overall": {
                "total": total,
                "completed": completed,
                "completed_rate": completed / total if total else 0,
                "success_at_1": success_at_1,
                "success_at_1_rate": success_at_1 / total if total else 0,
                "average_reward": average_reward,
            },
            "token_usage": {
                "total_prompt_tokens": total_prompt,
                "total_completion_tokens": total_completion,
                "total_tokens": total_tokens,
                "avg_prompt_tokens_per_episode": round(total_prompt / n_ep),
                "avg_completion_tokens_per_episode": round(total_completion / n_ep),
                "avg_total_tokens_per_episode": round(total_tokens / n_ep),
            },
        }
