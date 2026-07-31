import logging
import os
import traceback
from copy import deepcopy
from dataclasses import fields
from typing import Dict, Any, List, Optional
import json

from agentrl.worker.task import Task, Session
from agentrl.worker.typings import (AgentCancelledException,
                                    RewardHistoryItem,
                                    SampleStatus,
                                    TaskOutput,
                                    TaskSampleExecutionResult)
from openai.types.chat import (ChatCompletionSystemMessageParam,
                               ChatCompletionToolMessageParam,
                               ChatCompletionUserMessageParam)

from .environment import AlfworldEnvWrapper
from .utils import *
from src.server.harness import (
    ALFWorldHarnessConfig,
    ALFWorldHarnessRuntime,
    compile_hook,
    patch_take_action_tool_description,
    run_hook,
)
from src.server.harness.dsl import (
    apply_harness_rules,
    normalize_harness_overlay,
    overlay_cold_start_hints,
)


def _trailing_count(items: List[str], value: str) -> int:
    count = 0
    for item in reversed(items):
        if item == value:
            count += 1
        else:
            break
    return count


def _alfworld_rule_context(
    harness_runtime: ALFWorldHarnessRuntime,
    raw_output: str = "",
    final_action: str = "",
    admissible: Optional[List[str]] = None,
    observation: str = "",
    remaining_steps: Optional[int] = None,
    h2_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    admissible = admissible or []
    normalized = (final_action or raw_output or "").strip().lower()
    obs_norm = (observation or "").strip().lower()
    repeated_obs = _trailing_count(harness_runtime.last_observations, obs_norm) if obs_norm else 0
    repeated_action = _trailing_count(harness_runtime.last_actions, normalized) if normalized else 0
    nothing_happens_count = sum(
        1 for item in harness_runtime.last_observations if "nothing happens" in item
    )
    task_ctx = harness_runtime.task_ctx
    blocked = bool((h2_result or {}).get("blocked"))
    in_admissible = normalized in admissible
    return {
        "action": {
            "raw": raw_output,
            "normalized": normalized,
            "final_action": final_action,
            "in_admissible": in_admissible,
        },
        "state": {
            "nothing_happens_count": nothing_happens_count,
            "repeated_observation_count": repeated_obs,
            "repeated_action_count": repeated_action,
            "invalid_action_count": harness_runtime.invalid_consecutive_count,
            "remaining_steps": remaining_steps if remaining_steps is not None else 999,
        },
        "task": {
            "task_type": task_ctx.task_type if task_ctx else None,
            "target_type": task_ctx.target_type if task_ctx else None,
            "destination_type": task_ctx.destination_type if task_ctx else None,
        },
        "predicates": {
            "action_not_admissible": blocked or not in_admissible,
            "nothing_happens_repeated": nothing_happens_count >= 2,
            "same_observation_repeated": repeated_obs >= 2,
            "same_action_repeated": repeated_action >= 2,
            "remaining_steps_low": (remaining_steps if remaining_steps is not None else 999) <= 5,
        },
    }


def _alfworld_code_context(
    harness_runtime: ALFWorldHarnessRuntime,
    *,
    raw_output: str = "",
    final_action: str = "",
    admissible: Optional[List[str]] = None,
    observation: str = "",
    remaining_steps: Optional[int] = None,
    h2_result: Optional[Dict[str, Any]] = None,
    step: int = 0,
    max_step: int = 0,
) -> Dict[str, Any]:
    ctx = _alfworld_rule_context(
        harness_runtime=harness_runtime,
        raw_output=raw_output,
        final_action=final_action,
        admissible=admissible,
        observation=observation,
        remaining_steps=remaining_steps,
        h2_result=h2_result,
    )
    ctx.update(
        {
            "observation": observation or "",
            "admissible": list(admissible or []),
            "output": raw_output or "",
            "step": int(step),
            "max_step": int(max_step),
            "remaining_steps": remaining_steps if remaining_steps is not None else 999,
            "history": {
                "last_actions": list(harness_runtime.last_actions),
                "last_observations": list(harness_runtime.last_observations),
            },
        }
    )
    w = getattr(harness_runtime, "world", None)
    if w is not None:
        ctx["world"] = {
            "current_location": w.current_location,
            "inventory": w.inventory,
            "object_at": dict(w.object_at),
            "visited": sorted(w.visited),
            "unvisited": list(w.unvisited),
            "target_found": w.target_found,
            "target_location": w.target_location,
            "placed_count": w.placed_count,
            "placed_items": list(w.placed_items),
            "placed_locations": list(w.placed_locations),
            "lamp_location": w.lamp_location,
        }
    else:
        ctx["world"] = {}
    return ctx


class ALFWorld(Task):

    def __init__(self,
                 data_path: Optional[str],
                 config_path: Optional[str],
                 prompts_path: Optional[str],
                 split: str = 'dev',
                 max_step: int = 20,
                 tools: Optional[List[Dict[str, Any]]] = None,
                 task_ids: Optional[List[int]] = None,
                 **kwargs):
        self.task_ids = (
            [int(item) for item in task_ids] if task_ids is not None else None
        )
        self.harness_overlay = normalize_harness_overlay(
            kwargs.pop("harness_overlay", {}),
            benchmark="alfworld",
        )
        enabled = kwargs.pop("enabled", False)
        h2 = kwargs.pop("h2", True)
        h3 = kwargs.pop("h3", True)
        h4 = kwargs.pop("h4", True)
        h5 = kwargs.pop("h5", True)
        h5_top_k = kwargs.pop("h5_top_k", 1)
        config_kwargs = dict(
            enabled=bool(enabled),
            h2_enabled=bool(h2),
            h3_enabled=bool(h3),
            h4_enabled=bool(h4),
            h5_enabled=bool(h5),
            h5_top_k=int(h5_top_k),
        )
        for item in fields(ALFWorldHarnessConfig):
            if item.name in kwargs:
                config_kwargs[item.name] = kwargs.pop(item.name)
        self.harness_config = ALFWorldHarnessConfig(**config_kwargs)
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
                    compile_hook(code, benchmark="alfworld")
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
            # H3 hint is task-type-aware; a generic runtime is used here since
            # per-episode TaskContext is not yet available at class init time.
            # The per-episode runtime will apply a task-specific hint via init_task.
            runtime = ALFWorldHarnessRuntime(self.harness_config)
            tools = patch_take_action_tool_description(tools, runtime.build_h3_hint())
        super().__init__(tools=tools, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.tools = tools

        # load data_path
        self.data_path = data_path
        if self.data_path is None:
            raise Exception("missing parameter data_path")
        os.environ["ALFWORLD_DATA"] = self.data_path

        # load config for alfworld benchmark
        self.config_path = config_path
        if self.config_path is None:
            raise Exception("missing parameter config_path")
        self.config = load_config(self.config_path)

        # load prompts
        self.prompts_path = prompts_path
        if self.prompts_path is None:
            raise Exception("missing parameter prompts_path")
        self.prompts = load_prompts(self.prompts_path)

        # prepare data_files
        self.data_files = []
        self.split = split
        data_path = os.path.join("data/alfworld", f"{self.split}.json")
        with open(data_path, "r") as f:
            content = json.loads(f.read())
        for _, v in content.items():
            self.data_files.extend(v)
        self.data_files = [os.path.join(self.data_path, file) for file in self.data_files]

        # Support deterministic shuffle with a fixed seed
        shuffle_seed = kwargs.get("shuffle_seed", None)
        if shuffle_seed is not None:
            import random
            random.Random(shuffle_seed).shuffle(self.data_files)
        
        if self.task_ids is not None:
            invalid = [
                index
                for index in self.task_ids
                if index < 0 or index >= len(self.data_files)
            ]
            if invalid:
                raise ValueError(f"invalid ALFWorld task ids: {invalid[:5]}")
        else:
            # Preserve the historical contiguous-slice behavior unless an
            # explicit failed-task retry set is requested.
            start_idx = kwargs.get("start", 0)
            end_idx = kwargs.get("end", len(self.data_files))
            self.data_files = self.data_files[start_idx:end_idx]

            sample_size = kwargs.get("sample_size", None)
            if sample_size is not None:
                self.data_files = self.data_files[:sample_size]

        self.logger.info(f"successfully loaded {len(self.data_files)} games")
        if len(self.data_files) > 0:
            self.logger.debug(f"{self.data_files[0]=}")

        # other configs
        self.max_step = max_step
        self.prefixes = {
            'pick_and_place': 'put',
            'pick_clean_then_place': 'clean',
            'pick_heat_then_place': 'heat',
            'pick_cool_then_place': 'cool',
            'look_at_obj': 'examine',
            'pick_two_obj': 'puttwo'
        }

        self.env = AlfworldEnvWrapper(self.config)

    def get_indices(self) -> List[Any]:
        if self.task_ids is not None:
            return list(self.task_ids)
        return list(range(len(self.data_files)))

    def calculate_overall(self, results: List[TaskOutput]) -> Dict[str, Any]:
        """
            TaskOutput.result 0/1
        """
        def is_pass(config: TaskOutput) -> bool:
            if not config or not isinstance(config.result, dict):
                return False
            # Legacy path: explicit success marker.
            if "result" in config.result:
                return int(config.result.get("result", 0) == 1) == 1
            # New controller protocol path: reward/score carries success.
            reward = config.result.get("reward", None)
            if reward is not None:
                try:
                    return float(reward) >= 1.0
                except Exception:
                    return False
            metrics = config.result.get("metrics", {})
            score = metrics.get("score", None) if isinstance(metrics, dict) else None
            if score is not None:
                try:
                    return float(score) >= 1.0
                except Exception:
                    return False
            return False

        overall = {
            "total": len([config for config in results if config]),
            "pass": len([config for config in results if is_pass(config)]),
        }
        overall["wrong"] = overall["total"] - overall["pass"]
        overall["success_rate"] = overall["pass"] / overall["total"] if overall["total"] else 0
        return {
            "overall": overall,
        }

    def sync_start_sample(self, index, session: Session) -> TaskSampleExecutionResult:
        data_item = self.data_files[index]
        env = self.env.create_env(data_item)
        try:
            result, log_info, finish_reason = self.alfworld_run(session, env)
        except AgentCancelledException:
            return TaskSampleExecutionResult(status=SampleStatus.CANCELLED)
        except Exception:
            traceback.print_exc()
            return TaskSampleExecutionResult(status=SampleStatus.TASK_ERROR)
        finally:
            self.env.close_env(env)
        log_info.update({"result": result})
        return TaskSampleExecutionResult(status=finish_reason, result=log_info)

    @staticmethod
    def get_task_instruction():
        return """Interact with a household to solve a task. Imagine you are an intelligent agent in a household environment and your target is to perform actions to complete) the task goal. At the beginning of your interactions, you will be given the detailed description of the current environment and your goal to accomplish. A tool will be provided for you to use to submit the action you want to take. This tool is the only tool you should and must take in order to operate any action in the environment. The way you perform action is to place the action chosen by you in the arguments field of your tool call. For each of your turn, you will be given a list of actions which you can choose one to perform in this turn. The action you would like to take should be offered in this format: "the name of your next action", and you should fill it in the argument field of your tool call. Note that you should always call a tool to operate an action from the given choices. After your each turn, the environment will give you immediate feedback based on which you plan your next few steps. if the environment output "Nothing happened", that means the previous action is invalid and you should try more options.
 Reminder:
1. the action must be chosen from the given available actions. Any actions except provided available actions will be regarded as illegal.
2. Always call the tool to hand in your next action and think when necessary."""

    def get_prompt(self, filename: str):
        # return []
        for k, v in self.prefixes.items():
            if filename.startswith(k):
                example = self.prompts[v]
                return deepcopy(example)
        raise Exception(f"unsupported name: {filename}")
        # return self.prompts["naive_example"]

    @staticmethod
    def get_available_actions(actions):
        actions = "\n".join(actions)
        return " AVAILABLE ACTIONS: " + actions + "\n"

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
        h2: Dict[str, Any],
        action: str,
        reason: str,
    ) -> tuple[str, bool]:
        kind = effect.get("kind")
        if kind == "block_and_prompt":
            session.inject(ChatCompletionUserMessageParam(
                role='user',
                content=effect.get("message") or "Harness blocked this action. Choose a valid action.",
            ))
            session.inject(RewardHistoryItem(reward=0, score=0))
            return action, True
        if kind in {"force_action", "rewrite_action"} and effect.get("action"):
            action = effect["action"].strip().lower()
            h2["action"] = action
            h2["blocked"] = False
            h2["reason"] = reason
            if effect.get("message"):
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content=effect["message"],
                ))
        return action, False

    @staticmethod
    def _apply_post_step_effect(
        session: Session,
        harness_runtime: ALFWorldHarnessRuntime,
        effect: Dict[str, Any],
    ) -> None:
        if effect.get("kind") == "inject_hint" and effect.get("message"):
            session.inject(ChatCompletionUserMessageParam(
                role='user',
                content=effect["message"],
            ))
        elif effect.get("kind") == "force_action" and effect.get("action"):
            harness_runtime.force_next_action = effect["action"].strip().lower()
            if effect.get("message"):
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content=effect["message"],
                ))

    def alfworld_run(self, session: Session, env):
        finish_reason = SampleStatus.COMPLETED
        # env init
        ob, info = self.env.reset_env(env)
        ob = '\n'.join(ob[0].split('\n\n')[1:])
        log_info = {
            "log": [],
            "harness_trace": {"h2": [], "h3": [], "h4": [], "h5": [], "code_hooks": []},
        }
        code_hook_nb: Dict[str, Any] = {}
        current_observation = ob
        harness_runtime = None
        if self.harness_substrate_enabled:
            harness_runtime = ALFWorldHarnessRuntime(self.harness_config)
        initial_admissible = info.get('admissible_commands', [[]])[0]
        init_prompt = "Here is your task. " + ob + self.get_available_actions(initial_admissible)
        log_info["init_prompt"] = init_prompt
        if harness_runtime:
            # H0: parse task context and seed world model
            harness_runtime.init_task(init_prompt, initial_admissible)

        # Cold-start hints: built-in H5 and overlay skills are independent.
        cold_skills: list = []
        if harness_runtime:
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
                code_ctx = _alfworld_code_context(
                    harness_runtime,
                    admissible=initial_admissible,
                    observation=current_observation,
                    remaining_steps=self.max_step,
                    step=0,
                    max_step=self.max_step,
                )
                for hook_out in self._run_code_hooks("on_init", code_ctx, code_hook_nb):
                    for skill in hook_out.get("skills", []) or []:
                        cold_skills.append(skill)
                        log_info["harness_trace"]["code_hooks"].append(
                            {"hook": "on_init", "effect": "skill", "text": skill.get("text", "")}
                        )
                    if hook_out.get("tool_hint"):
                        self.tools = patch_take_action_tool_description(self.tools, hook_out["tool_hint"])
                        log_info["harness_trace"]["code_hooks"].append(
                            {"hook": "on_init", "effect": "tool_hint", "text": hook_out["tool_hint"]}
                        )
            for item in cold_skills:
                log_info["harness_trace"]["h5"].append(item)

        system_prompt = self.get_task_instruction()
        if cold_skills:
            skill_lines = "\n".join(f"- {item['text']}" for item in cold_skills)
            system_prompt += f"\n\nSome tips that may help for this task:\n{skill_lines}"

        session.inject(ChatCompletionSystemMessageParam(
            role='system',
            content=system_prompt
        ))
        session.inject(ChatCompletionUserMessageParam(
            role='user',
            content=init_prompt
        ))

        # interact
        # Cache admissible commands from the previous env step so they are
        # available even on turns where the agent makes no tool call.
        _last_admissible: List[str] = initial_admissible
        # Count consecutive turns with no tool call so we can escalate hints.
        _no_tool_consecutive: int = 0

        for i in range(0, self.max_step):
            if harness_runtime and self.overlay_has_code_hooks and self._code_hooks.get("make_pre_hint"):
                pre_ctx = _alfworld_code_context(
                    harness_runtime,
                    admissible=_last_admissible,
                    observation=current_observation,
                    remaining_steps=self.max_step - i,
                    step=i,
                    max_step=self.max_step,
                )
                for hook_out in self._run_code_hooks("make_pre_hint", pre_ctx, code_hook_nb):
                    message = hook_out.get("message")
                    if message:
                        session.inject(ChatCompletionUserMessageParam(role='user', content=message))
                        log_info["harness_trace"]["code_hooks"].append(
                            {"round": i + 1, "hook": "make_pre_hint", "message": message}
                        )
            output = session.sync_action()

            tool_calls = []
            for message in output.messages:
                tool_calls.extend(message.get('tool_calls', []) or [])

            if not tool_calls:
                _no_tool_consecutive += 1
                # Do not set finish_reason here — this is a mid-episode event,
                # not a terminal state. The episode continues via `continue`.
                no_exec_msg = (
                    'You MUST call the take_action tool — '
                    'do NOT output plain text without a tool call.'
                )
                # After 2+ consecutive no-tool turns, inject a directed step hint
                # so the agent has a concrete action to take rather than narrating.
                if (
                    _no_tool_consecutive >= 2
                    and harness_runtime
                    and self.harness_config.enabled
                    and self.harness_config.h4_enabled
                ):
                    forced_hint = harness_runtime.step_guidance(
                        current_round=i + 1,
                        max_step=self.max_step,
                        admissible=_last_admissible,
                    )
                    if forced_hint:
                        no_exec_msg = no_exec_msg + '\n' + forced_hint
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content=no_exec_msg,
                ))
                session.inject(RewardHistoryItem(reward=0, score=0))
                continue

            _no_tool_consecutive = 0

            try:
                tool_call = tool_calls[0]
                arguments = tool_call["function"]["arguments"]
                arguments = json.loads(arguments)
                arguments = list(arguments.values())
                call_id = tool_call["id"]
                # process action
                admissible_commands = info.get('admissible_commands', [[]])[0]
                output = arguments[0]
                builtin_h2_enabled = (
                    self.harness_config.enabled
                    and self.harness_config.h2_enabled
                )
                if harness_runtime and builtin_h2_enabled:
                    h2 = harness_runtime.pre_validate_action(output, admissible_commands)
                    action = h2["action"]
                else:
                    action = process_action(output, admissible_commands)
                    h2 = {
                        "action": action,
                        "canonicalized": False,
                        "blocked": False,
                        "reason": "",
                        "raw_action": output,
                    }
                    if harness_runtime and harness_runtime.force_next_action:
                        action = harness_runtime.force_next_action
                        harness_runtime.force_next_action = None
                        h2["action"] = action
                        h2["canonicalized"] = True
                        h2["reason"] = "overlay_force_next_action"

                if harness_runtime and self.overlay_has_guard_rules:
                    overlay_guard = apply_harness_rules(
                        self.harness_overlay.get("guard_rules", []),
                        _alfworld_rule_context(
                            harness_runtime=harness_runtime,
                            raw_output=output,
                            final_action=action,
                            admissible=admissible_commands,
                            remaining_steps=self.max_step - i,
                            h2_result=h2,
                        ),
                        trigger="before_action",
                    )
                    if overlay_guard:
                        effect = overlay_guard.get("effect", {})
                        action, blocked_by_overlay = self._apply_before_action_effect(
                            session,
                            effect,
                            h2,
                            action,
                            f"overlay_{effect.get('kind')}:{overlay_guard.get('rule_id')}",
                        )
                        if blocked_by_overlay:
                            continue

                if harness_runtime and self.overlay_has_code_hooks and self._code_hooks.get("on_before_action"):
                    code_ctx = _alfworld_code_context(
                        harness_runtime=harness_runtime,
                        raw_output=output,
                        final_action=action,
                        admissible=admissible_commands,
                        observation=current_observation,
                        remaining_steps=self.max_step - i,
                        h2_result=h2,
                        step=i,
                        max_step=self.max_step,
                    )
                    blocked_by_code_any = False
                    for hook_out in self._run_code_hooks("on_before_action", code_ctx, code_hook_nb):
                        action, blocked_by_code = self._apply_before_action_effect(
                            session,
                            hook_out,
                            h2,
                            action,
                            f"code_hook_{hook_out.get('kind')}",
                        )
                        log_info["harness_trace"]["code_hooks"].append(
                            {
                                "round": i + 1,
                                "hook": "on_before_action",
                                "effect": hook_out,
                                "final_action": action,
                            }
                        )
                        if blocked_by_code:
                            blocked_by_code_any = True
                            break
                    if blocked_by_code_any:
                        continue

                if builtin_h2_enabled:
                    log_info["harness_trace"]["h2"].append(
                        {
                            "round": i + 1,
                            "canonicalized": h2["canonicalized"],
                            "blocked": h2["blocked"],
                            "reason": h2["reason"],
                            "raw_action": h2["raw_action"],
                            "final_action": action,
                        }
                    )
                if h2["blocked"]:
                    # Do not set finish_reason — mid-episode block, episode continues.
                    session.inject(ChatCompletionUserMessageParam(
                        role='user',
                        content=(
                            "Harness blocked an invalid action sequence. "
                            "Please choose a valid action from AVAILABLE ACTIONS."
                        ),
                    ))
                    session.inject(RewardHistoryItem(reward=0, score=0))
                    continue
            except:
                # Do not set finish_reason — mid-episode exception, episode continues.
                session.inject(ChatCompletionUserMessageParam(
                    role='user',
                    content='No valid tool calls found. Please call a tool instead.'
                ))
                session.inject(RewardHistoryItem(reward=0, score=0))
                continue

            observation, reward, done, info = self.env.step_env(env, action)
            observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
            current_observation = observation
            _last_admissible = info.get('admissible_commands', [[]])[0]
            session.inject(ChatCompletionToolMessageParam(
                role='tool',
                tool_call_id=call_id,
                content=observation + self.get_available_actions(_last_admissible)
            ))
            round_reward = reward
            if "Nothing happens" in observation:
                round_reward = 0
            session.inject(RewardHistoryItem(reward=round_reward, score=reward))

            # save
            payload = {
                "round": i + 1,
                "output": output,
                "action": action,
                "admissible_commands": admissible_commands,
                "observation": observation,
                "done": done,
            }
            log_info["log"].append(payload)

            post_admissible = info.get('admissible_commands', [[]])[0]
            if harness_runtime:
                builtin_h4_enabled = (
                    self.harness_config.enabled
                    and self.harness_config.h4_enabled
                )
                h4 = harness_runtime.post_step_monitor(
                    raw_output=output,
                    final_action=action,
                    observation=observation,
                    admissible=post_admissible,
                )
                if builtin_h4_enabled:
                    log_info["harness_trace"]["h4"].append({"round": i + 1, **h4})
                    if h4.get("recovery_prompt"):
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=h4["recovery_prompt"],
                        ))
                if self.overlay_has_recovery_rules:
                    overlay_recovery = apply_harness_rules(
                        self.harness_overlay.get("recovery_rules", []),
                        _alfworld_rule_context(
                            harness_runtime=harness_runtime,
                            raw_output=output,
                            final_action=action,
                            admissible=post_admissible,
                            observation=observation,
                            remaining_steps=self.max_step - i - 1,
                        ),
                        trigger="post_step",
                    )
                    if overlay_recovery:
                        effect = overlay_recovery.get("effect", {})
                        self._apply_post_step_effect(session, harness_runtime, effect)
                if self.overlay_has_code_hooks and self._code_hooks.get("on_post_step"):
                    code_ctx = _alfworld_code_context(
                        harness_runtime=harness_runtime,
                        raw_output=output,
                        final_action=action,
                        admissible=post_admissible,
                        observation=observation,
                        remaining_steps=self.max_step - i - 1,
                        step=i,
                        max_step=self.max_step,
                    )
                    for hook_out in self._run_code_hooks("on_post_step", code_ctx, code_hook_nb):
                        self._apply_post_step_effect(session, harness_runtime, hook_out)
                        log_info["harness_trace"]["code_hooks"].append(
                            {"round": i + 1, "hook": "on_post_step", "effect": hook_out}
                        )
                if builtin_h4_enabled:
                    # H4-D: step-budget management
                    budget = harness_runtime.budget_check(
                        remaining_steps=self.max_step - i - 1,
                        admissible=post_admissible,
                    )
                    if budget.get("force_action"):
                        harness_runtime.force_next_action = budget["force_action"]
                    if budget.get("hint"):
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=budget["hint"],
                        ))
                    if budget.get("force_action") or budget.get("hint"):
                        log_info["harness_trace"]["h4"].append({
                            "round": i + 1,
                            "sub": "h4d_budget",
                            "forced": budget.get("force_action"),
                            "hint": budget.get("hint"),
                        })
                    # H4-E: state-driven per-step guidance (WorldModel + SubgoalSM)
                    step_hint = harness_runtime.step_guidance(
                        current_round=i + 1,
                        max_step=self.max_step,
                        admissible=post_admissible,
                    )
                    if step_hint:
                        session.inject(ChatCompletionUserMessageParam(
                            role='user',
                            content=step_hint,
                        ))
                        log_info["harness_trace"]["h4"].append({
                            "round": i + 1,
                            "sub": "h4e_step_guidance",
                            "text": step_hint,
                            "token_cost": str(len(step_hint.split())),
                        })
            # failure test
            if len(log_info["log"]) > 3:
                pre_logs = log_info["log"][-3:]
                pre_acts = [pre_log["output"] for pre_log in pre_logs]
                if len(list(set(pre_acts))) == 1:
                    self.logger.info("repeat actions for 3 times: failure")
                    return 0, log_info, SampleStatus.AGENT_INVALID_ACTION

            if done:
                return reward, log_info, finish_reason
        else:
            finish_reason = SampleStatus.TASK_LIMIT_REACHED
            final_reward = 0
            reward_history = RewardHistoryItem(reward=final_reward, score=0)
            session.inject(reward_history)

        return 0, log_info, finish_reason
