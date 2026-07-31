"""AgentRL worker entrypoint that preserves per-session WebShop tools."""

from __future__ import annotations

import argparse
import logging

from agentrl.worker.config import ConfigLoader
from agentrl.worker.task_worker import TaskWorker, model_dump
from agentrl.worker.typings import InstanceFactory, InteractRequest

from webshop_agentrl_passthrough import inject_webshop_task_manifest


class SessionToolTaskWorker(TaskWorker):
    """Use the episode-local tool list on every interaction, not only turn one."""

    async def interact(self, parameters: InteractRequest):
        running = self.session_map.get(parameters.session_id)
        session_tools = running.session.tools if running is not None else None
        response = await super().interact(parameters)
        if session_tools is not None and isinstance(response, dict):
            env_out = response.get("env_out")
            if isinstance(env_out, dict):
                env_out["tools"] = model_dump(session_tools)
        return inject_webshop_task_manifest(response, running)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--controller", "-C", default="http://localhost:5020/api")
    parser.add_argument("--self", "-s", default="http://localhost:5021/api")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=5021)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S %Z",
    )
    logger = logging.getLogger("task_worker")
    logger.info(
        "Starting WebShop task %s config=%s controller=%s self=%s port=%s",
        args.name,
        args.config,
        args.controller,
        args.self,
        args.port,
    )
    config = ConfigLoader().load_from(args.config, args.name)
    task = InstanceFactory.model_validate(config[args.name]).create()
    worker = SessionToolTaskWorker(
        task,
        controller_address=args.controller,
        self_address=args.self,
        logger=logger,
    )
    worker.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
