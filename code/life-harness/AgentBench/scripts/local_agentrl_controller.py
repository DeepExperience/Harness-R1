#!/usr/bin/env python3
"""Minimal local AgentRL controller for single-node AgentBench runs.

The pip package used here ships task workers, but not a standalone HTTP
controller. This wrapper implements the small controller API surface that
AgentBench's TaskClient needs and forwards session traffic to registered
workers.
"""

from __future__ import annotations

import argparse
import itertools
import time
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse


app = FastAPI()
workers: dict[str, dict[str, dict[str, Any]]] = {}
session_to_worker: dict[int, str] = {}
session_counter = itertools.count(1)


def worker_api(address: str, endpoint: str) -> str:
    base = address.rstrip("/")
    if base.endswith("/api"):
        return f"{base}/{endpoint.lstrip('/')}"
    return f"{base}/api/{endpoint.lstrip('/')}"


def alive_worker_items(name: str) -> list[tuple[str, dict[str, Any]]]:
    now = time.time()
    task_workers = workers.get(name) or {}
    return [
        (wid, info)
        for wid, info in task_workers.items()
        if now - float(info.get("last_seen", 0.0)) < 60.0
    ]


@app.post("/api/receive_heartbeat")
async def receive_heartbeat(payload: dict[str, Any]) -> dict[str, Any]:
    name = str(payload.get("name") or "")
    address = str(payload.get("address") or "")
    if not name or not address:
        raise HTTPException(status_code=400, detail="heartbeat requires name and address")
    task_workers = workers.setdefault(name, {})
    worker_id = address
    current = 0
    try:
        status = requests.get(worker_api(address, "worker_status"), timeout=2)
        if status.status_code == 200:
            current = int((status.json() or {}).get("current") or 0)
    except requests.RequestException:
        current = 0
    task_workers[worker_id] = {
        "address": address,
        "capacity": int(payload.get("concurrency") or 1),
        "current": current,
        "indices": payload.get("indices") or [],
        "status": "ALIVE",
        "last_seen": time.time(),
    }
    return {"ok": True}


@app.get("/api/list_workers")
async def list_workers() -> dict[str, Any]:
    out: dict[str, Any] = {}
    now = time.time()
    for name, task_workers in workers.items():
        out[name] = {"workers": {}}
        for wid, info in task_workers.items():
            alive = now - float(info.get("last_seen", 0.0)) < 60.0
            item = dict(info)
            item["status"] = "ALIVE" if alive else "DEAD"
            out[name]["workers"][wid] = item
    return out


@app.get("/api/get_indices")
async def get_indices(name: str) -> list[Any]:
    for _, info in alive_worker_items(name):
        address = info["address"]
        response = requests.get(worker_api(address, "get_indices"), timeout=30)
        if response.status_code == 200:
            return response.json()
    raise HTTPException(status_code=404, detail=f"no alive worker for task {name}")


@app.post("/api/start_sample")
async def start_sample(payload: dict[str, Any]) -> JSONResponse:
    name = str(payload.get("name") or "")
    if not name:
        raise HTTPException(status_code=400, detail="start_sample requires name")
    candidates = alive_worker_items(name)
    if not candidates:
        raise HTTPException(status_code=404, detail=f"no alive worker for task {name}")
    session_id = int(next(session_counter))
    worker_id, info = candidates[0]
    address = info["address"]
    worker_payload = {
        "index": payload.get("index"),
        "custom_task": payload.get("custom_task"),
        "session_id": session_id,
    }
    response = requests.post(worker_api(address, "start_sample"), json=worker_payload, timeout=300)
    if response.status_code != 200:
        return JSONResponse(
            status_code=response.status_code,
            content={"detail": response.text},
        )
    session_to_worker[session_id] = worker_id
    return JSONResponse(content=response.json(), headers={"session_id": str(session_id)})


@app.post("/api/interact")
async def interact(
    request: Request,
    session_id: str | None = Header(default=None, convert_underscores=False),
) -> JSONResponse:
    payload = await request.json()
    sid = session_id or payload.get("session_id")
    if sid is None:
        raise HTTPException(status_code=400, detail="interact requires session_id")
    sid_i = int(sid)
    worker_id = session_to_worker.get(sid_i)
    if worker_id is None:
        raise HTTPException(status_code=404, detail=f"unknown session {sid_i}")
    worker_payload = {"session_id": sid_i, "messages": payload.get("messages") or []}
    response = requests.post(worker_api(worker_id, "interact"), json=worker_payload, timeout=300)
    if response.status_code != 200:
        return JSONResponse(status_code=response.status_code, content={"detail": response.text})
    body = response.json()
    if isinstance(body, dict) and isinstance(body.get("env_out"), dict):
        body = body["env_out"]
    return JSONResponse(content=body, headers={"session_id": str(sid_i)})


@app.post("/api/cancel")
async def cancel(
    request: Request,
    session_id: str | None = Header(default=None, convert_underscores=False),
) -> dict[str, Any]:
    payload = await request.json() if request.headers.get("content-length") not in (None, "0") else {}
    sid = session_id or payload.get("session_id")
    if sid is None:
        return {"ok": True}
    sid_i = int(sid)
    worker_id = session_to_worker.pop(sid_i, None)
    if worker_id is None:
        return {"ok": True}
    try:
        requests.post(worker_api(worker_id, "cancel"), json={"session_id": sid_i}, timeout=30)
    except requests.RequestException:
        pass
    return {"session_id": sid_i}


@app.post("/api/cancel_notice")
async def cancel_notice(request: Request) -> dict[str, Any]:
    payload = await request.json()
    sid = payload.get("session_id")
    if sid is not None:
        session_to_worker.pop(int(sid), None)
    return {"ok": True}


@app.post("/api/calculate_overall")
async def calculate_overall(payload: dict[str, Any]) -> JSONResponse:
    name = str(payload.get("name") or "")
    for _, info in alive_worker_items(name):
        address = info["address"]
        response = requests.post(worker_api(address, "calculate_overall"), json=payload, timeout=300)
        if response.status_code == 200:
            return JSONResponse(content=response.json())
    raise HTTPException(status_code=404, detail=f"no alive worker for task {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
