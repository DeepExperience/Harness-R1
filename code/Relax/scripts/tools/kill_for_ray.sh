#!/bin/bash

# Copyright (c) 2026 Relax Authors. All Rights Reserved.

set -exo pipefail

echo "=== Cleaning up residual python/sglang processes ==="

ps -eo ppid,pid,cmd | awk '$1>1' | egrep -i "python|sglang" | egrep -v 'gpustat|raylet|gcs_server|plasma|grep|.sh' | awk '{print $2}' | xargs kill -9 2>/dev/null || true
