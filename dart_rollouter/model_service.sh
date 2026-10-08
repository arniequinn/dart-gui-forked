#!/usr/bin/env bash
# start_model.sh - Start the model service
# Usage: chmod +x start_model.sh

source ~/miniconda3/bin/activate 
conda activate verl
# cd /workspace/codes/verl/rollouter/
set -euo pipefail

# 1. GPU Monitoring
# mkdir -p /root/verl/rollouter/gpu_util_log
# MONITOR_ID="/root/verl/rollouter/gpu_util_log/gpu_monitor_model_$(date +%Y%m%d_%H%M%S)_$$"

# echo "[INFO] Attempting to start GPU monitoring for Model Service..."
# if command -v nvidia-smi >/dev/null 2>&1; then
#   if timeout 5 nvidia-smi > /dev/null 2>&1; then
#     nohup nvidia-smi --query-gpu=index,timestamp,name,pci.bus_id,driver_version,pstate,pcie.link.gen.max,pcie.link.gen.current,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 1 > "${MONITOR_ID}.csv" 2>&1 &
#     echo "[INFO] GPU monitoring started"
#   else
#     echo "[WARN] nvidia-smi execution failed, skipping."
#   fi
# else
#   echo "[WARN] nvidia-smi not found, skipping."
# fi

# 2. Config & Env
export PYTHONUNBUFFERED=1

# 3. Start Model Service
echo "[INFO] Starting model service: python -m src.run_model"
exec python -m src.run_model
