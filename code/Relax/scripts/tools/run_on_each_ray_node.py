# Copyright (c) 2026 Relax Authors. All Rights Reserved.

import argparse
import os
import re
import subprocess
import sys
from typing import Any, List, Optional

import ray
from loguru import logger


# Dangerous command patterns (whitelist approach would be safer but harder to maintain)
DANGEROUS_PATTERNS = [
    # File deletion - most dangerous
    r"^\s*rm\s+-?r?[f]+",  # rm -rf, rm -rf /, rm -f, etc.
    r"\s+rm\s+-?r?[f]+",  # rm in middle of command
    r"^\s*del\s+",  # Windows delete
    r"^\s*unlink\s+",  # Unix unlink
    # Disk operations
    r"^\s*dd\s+",  # dd if=/dev/zero of=/dev/sda
    r"^\s*mkfs",  # mkfs.ext4 /dev/sda
    r"^\s*fdisk\s+.*delete",  # fdisk delete partition
    r"^\s*parted\s+.*rm",  # parted remove partition
    # Permission modification (can lock out)
    r"^\s*chmod\s+-R\s+0",  # chmod -R 000
    r"^\s*chmod\s+-R\s+777",  # chmod -R 777 (too permissive)
    r"^\s*chown\s+-R",  # chown -R recursively
    # Fork bomb - CPU exhaustion
    r":\(\)\s*:\|:&\}:",  # :(){:|:&};:
    r"fork\(\)\s*\{.*\|.*&\}",  # fork bomb variant
    # Shutdown/reboot
    r"^\s*shutdown",  # shutdown system
    r"^\s*reboot",  # reboot
    r"^\s*init\s+0",  # init 0 (halt)
    r"^\s*init\s+6",  # init 6 (reboot)
    r"^\s*systemctl\s+poweroff",  # systemctl poweroff
    r"^\s*systemctl\s+reboot",  # systemctl reboot
    # Network-based attacks
    r"curl\s+.*\|\s*bash",  # curl ... | bash (remote script execution)
    r"wget\s+.*\|\s*bash",  # wget ... | bash
    r"python\s+-m\s+http\.server",  # start HTTP server (potential security issue)
    # Environment modification
    r"^\s*export\s+PATH=.*\. .",  # suspicious PATH modification
    r"^\s*source\s+/etc/profile",  # source system profile
    # Sudo and privilege escalation
    r"^\s*sudo\s+rm",  # sudo rm
    r"^\s*sudo\s+dd",  # sudo dd
    r"^\s*sudo\s+mkfs",  # sudo mkfs
]

# Default timeout for commands (can be overridden via RAY_CMD_TIMEOUT env var)
DEFAULT_TIMEOUT = 300


def is_command_safe(cmd: str) -> tuple[bool, Optional[str]]:
    """Check if command contains dangerous patterns.

    Returns:
        (is_safe, reason_if_unsafe)
    """
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return False, f"Matched dangerous pattern: {pattern}"

    # Additional checks
    cmd_stripped = cmd.strip()

    # Check for direct root file deletion attempts
    if re.match(r"rm\s+-?[rf]+\s+/", cmd_stripped):
        return False, "Attempting to delete root directory"

    return True, None


@ray.remote
def run_command(cmd: str) -> dict[str, Any]:
    """Run command on remote Ray node with timeout."""
    # Get timeout from environment variable
    timeout = int(os.environ.get("RAY_CMD_TIMEOUT", str(DEFAULT_TIMEOUT)))

    # Safety check on remote node
    is_safe, reason = is_command_safe(cmd)
    if not is_safe:
        return {
            "node": os.uname()[1],
            "success": False,
            "output": "",
            "error": f"Command blocked by safety check: {reason}",
            "returncode": -1,
            "stdout": "",
            "stderr": "",
            "timed_out": False,
        }

    try:
        # Use Popen for better control, with timeout
        process = subprocess.Popen(
            ["bash", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # Line buffered
        )

        try:
            stdout, stderr = process.communicate(timeout=timeout)
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            returncode = -1
            return {
                "node": os.uname()[1],
                "success": False,
                "output": stdout,
                "stdout": stdout,
                "stderr": stderr,
                "error": f"Command timed out after {timeout} seconds",
                "returncode": returncode,
                "timed_out": True,
            }

        return {
            "node": os.uname()[1],
            "success": returncode == 0,
            "output": stdout if returncode == 0 else stderr,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": returncode,
            "error": None if returncode == 0 else stderr,
            "timed_out": False,
        }
    except Exception as e:
        return {
            "node": os.uname()[1],
            "success": False,
            "output": "",
            "stdout": "",
            "stderr": "",
            "error": str(e),
            "returncode": -1,
            "timed_out": False,
        }


def format_output(result: dict[str, Any], show_stdout_stderr: bool = True) -> str:
    """Format result for elegant output printing."""
    node = result.get("node", "unknown")
    success = result.get("success", False)
    returncode = result.get("returncode", -1)

    status_emoji = "✅" if success else "❌"
    status_text = "成功" if success else "失败"

    lines = [f"{status_emoji} [{node}] 状态: {status_text} (exit code: {returncode})"]

    if show_stdout_stderr:
        stdout = result.get("stdout", "").strip()
        stderr = result.get("stderr", "").strip()
        error = result.get("error")
        timed_out = result.get("timed_out", False)

        if stdout:
            lines.append(f"  stdout:\n{indent_text(stdout, 4)}")
        if stderr and not success:
            lines.append(f"  stderr:\n{indent_text(stderr, 4)}")
        if error and "blocked" in error.lower():
            lines.append(f"  ⚠️ {error}")
        if timed_out:
            lines.append("  ⏰ 执行超时")

    return "\n".join(lines)


def indent_text(text: str, spaces: int = 4) -> str:
    """Indent text by specified number of spaces."""
    indent = " " * spaces
    return "\n".join(indent + line for line in text.split("\n"))


def list_ray_nodes() -> List[dict[str, Any]]:
    """List all nodes in the Ray cluster with their information."""
    if not ray.is_initialized():
        ray.init()

    nodes = ray.nodes()
    gpu_nodes = []

    print("\n" + "=" * 80)
    print("📋 Ray Cluster Nodes")
    print("=" * 80)

    for i, node in enumerate(nodes):
        node_id = node.get("NodeID", "unknown")
        address = node.get("NodeManagerAddress", "unknown")
        alive = node.get("Alive", False)
        resources = node.get("Resources", {})
        gpu_count = resources.get("GPU", 0)

        status = "🟢 Alive" if alive else "🔴 Dead"
        is_gpu = "🚀 GPU" if gpu_count > 0 else "💻 CPU"

        print(f"\nNode {i + 1}:")
        print(f"  NodeID:              {node_id}")
        print(f"  NodeManagerAddress: {address}")
        print(f"  Status:              {status}")
        print(f"  GPUs:                {gpu_count}")
        print(f"  Type:                {is_gpu}")

        if gpu_count > 0:
            gpu_nodes.append(node)

    print("\n" + "=" * 80)
    print(f"Total: {len(nodes)} nodes, {len(gpu_nodes)} GPU nodes")
    print("=" * 80 + "\n")

    return gpu_nodes


def find_node_by_id(nodes: List[dict[str, Any]], node_id: str) -> Optional[dict[str, Any]]:
    """Find a node by NodeID or NodeManagerAddress (IP).

    Args:
        nodes: List of node info from ray.nodes()
        node_id: NodeID (hex string) or NodeManagerAddress (IP)

    Returns:
        Node dict if found, None otherwise
    """
    node_id_lower = node_id.lower()

    for node in nodes:
        # Match by NodeID (case-insensitive)
        if node.get("NodeID", "").lower() == node_id_lower:
            return node

        # Match by NodeManagerAddress (IP)
        if node.get("NodeManagerAddress", "") == node_id:
            return node

    return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run command on Ray cluster nodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on all GPU nodes (default behavior)
  %(prog)s "nvidia-smi"

  # Run on specific node by IP
  %(prog)s -n 10.0.0.1 "nvidia-smi"

  # Run on specific node by NodeID
  %(prog)s --node-id 2f3104563016741c3a7d6395cc7e9393aad6bb5cdb2b83c0a4e7c6c9 "nvidia-smi"

  # List available GPU nodes
  %(prog)s --list

  # Run with custom timeout (60 seconds)
  %(prog)s -t 60 "python train.py"
        """,
    )

    parser.add_argument(
        "command",
        nargs="*",
        help="Command to execute on nodes",
    )

    parser.add_argument(
        "-n",
        "--node-id",
        type=str,
        default=None,
        help="Specific node ID (NodeID or NodeManagerAddress/IP) to execute on. "
        "If not specified, runs on all GPU nodes.",
    )

    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List all available GPU nodes and exit",
    )

    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Command timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )

    parser.add_argument(
        "--include-cpu",
        action="store_true",
        help="Include CPU nodes (default: GPU nodes only)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Ray if not already done
    if not ray.is_initialized():
        ray.init()

    # Handle --list option
    if args.list:
        list_ray_nodes()
        return

    # Validate command
    if not args.command:
        logger.error("❌ No command provided. Use --help for usage information.")
        sys.exit(1)

    cmd = " ".join(args.command)
    timeout = args.timeout

    # Pre-execution safety check on local machine
    is_safe, reason = is_command_safe(cmd)
    if not is_safe:
        logger.error(f"⚠️ Command blocked: {reason}")
        logger.error("❌ 操作被拒绝")
        sys.exit(1)

    logger.info(f"{cmd=}")
    logger.info(f"timeout={timeout}s")

    # Get all nodes from cluster
    nodes = ray.nodes()

    # Filter nodes based on criteria
    target_nodes = []
    for node in nodes:
        # Check if node is alive
        if not node.get("Alive", False):
            continue

        # Check resource requirements
        resources = node.get("Resources", {})
        has_gpu = "GPU" in resources and resources.get("GPU", 0) > 0

        if args.include_cpu:
            # Include all nodes
            target_nodes.append(node)
        elif has_gpu:
            # GPU nodes only
            target_nodes.append(node)
        else:
            # Skip non-GPU nodes
            continue

    # If specific node_id is provided, filter to that node only
    if args.node_id:
        found_node = find_node_by_id(target_nodes, args.node_id)
        if found_node is None:
            logger.error(f"❌ Node not found: {args.node_id}")
            logger.error("Available nodes:")
            for node in target_nodes:
                logger.error(f"  - NodeID: {node.get('NodeID')}, Address: {node.get('NodeManagerAddress')}")
            sys.exit(1)
        target_nodes = [found_node]
        logger.info(f"🎯 Targeting specific node: {args.node_id}")
    elif not args.include_cpu:
        logger.info("Targeting all GPU nodes (default behavior)")
    else:
        logger.info("Targeting all nodes (--include-cpu enabled)")

    if not target_nodes:
        logger.warning("No target nodes found in Ray cluster!")
        return

    # Create tasks for each target node
    tasks = []
    node_addresses = []

    for node in target_nodes:
        # Use NodeManagerAddress for resource binding
        node_address = node["NodeManagerAddress"]
        node_id_full = node["NodeID"]

        # Use node address as resource label for scheduling
        task = run_command.options(resources={f"node:{node_address}": 1}).remote(cmd)
        tasks.append(task)
        node_addresses.append(node_address)

        logger.info(f"  📍 Node: {node_address} (ID: {node_id_full[:16]}...)")

    logger.info(f"✨ 已向 {len(tasks)} 个节点发送命令 (timeout={timeout}s)")

    # Collect all results
    results = ray.get(tasks)

    # Print formatted results
    print("\n" + "=" * 60)
    print("📋 执行结果")
    print("=" * 60)

    failed_results = []
    success_count = 0

    for i, result in enumerate(results):
        node_info = node_addresses[i] if i < len(node_addresses) else f"Node {i + 1}"
        print(f"\n--- {node_info} ---")
        print(format_output(result))

        if result.get("success"):
            success_count += 1
        else:
            failed_results.append(result)

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    print(f"📊 摘要: 成功 {success_count}/{total}, 失败 {len(failed_results)}/{total}")

    if failed_results:
        print("\n❌ 失败节点:")
        for fr in failed_results:
            print(f"  - {fr['node']}: {fr.get('error', fr.get('output', 'unknown error'))[:100]}")

    print("=" * 60 + "\n")

    if failed_results:
        raise RuntimeError(f"{len(failed_results)} task(s) failed")


if __name__ == "__main__":
    main()
