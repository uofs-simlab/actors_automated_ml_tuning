#!/usr/bin/env bash
# Sweeps worker-actor counts / beta lists / gpu counts over ./test_actor
# (no changes to test.cc), logging per-job (beta, MMD) results and
# per-run timing + GPU utilization (via nvidia-smi) to CSV.
#
# Must be run from the actors/ directory (test_actor resolves
# ../.venv/bin/python3 relative to its own cwd).
set -euo pipefail

BIN="./test_actor"

# --- edit these to define the sweep -----------------------------------
WORKERS_LIST=(1 2 4 8)                      # --workers values to try
GPUS_LIST=(1)                               # --gpus values to try
BETAS_LIST=(
  "0.1,0.5,1.0,2.0,4.0"
  "0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8"
)                                           # --betas values to try (each entry = one run)
# ------------------------------------------------------------------------

OUT_DIR="results"
LOG_DIR="$OUT_DIR/logs"
GPU_DIR="$OUT_DIR/gpu"
RUNS_CSV="$OUT_DIR/runs.csv"        # one row per completed job (beta, mmd)
SUMMARY_CSV="$OUT_DIR/summary.csv"  # one row per test_actor invocation

mkdir -p "$LOG_DIR" "$GPU_DIR"

[[ -f "$RUNS_CSV" ]] || echo "run_id,timestamp,workers,gpus,beta,mmd" > "$RUNS_CSV"
[[ -f "$SUMMARY_CSV" ]] || echo "run_id,timestamp,workers,gpus,num_betas,reported_seconds,wall_seconds,gpu_util_avg_pct,gpu_util_max_pct,gpu_mem_avg_mib" > "$SUMMARY_CSV"

have_nvidia_smi=0
command -v nvidia-smi >/dev/null 2>&1 && have_nvidia_smi=1

for gpus in "${GPUS_LIST[@]}"; do
  for workers in "${WORKERS_LIST[@]}"; do
    for betas in "${BETAS_LIST[@]}"; do
      run_id="w${workers}_g${gpus}_$(date +%Y%m%d_%H%M%S)"
      log_file="$LOG_DIR/${run_id}.log"
      gpu_file="$GPU_DIR/${run_id}.csv"
      ts="$(date -Iseconds)"

      echo "=== run $run_id : workers=$workers gpus=$gpus betas=$betas ==="

      gpu_pid=""
      if [[ $have_nvidia_smi -eq 1 ]]; then
        nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used \
          --format=csv,noheader,nounits -l 1 > "$gpu_file" 2>/dev/null &
        gpu_pid=$!
      fi

      start=$(date +%s)
      "$BIN" --server-mode --port=0 --workers="$workers" --gpus="$gpus" --betas="$betas" \
        > "$log_file" 2>&1
      end=$(date +%s)
      wall_seconds=$((end - start))

      if [[ -n "$gpu_pid" ]]; then
        kill "$gpu_pid" 2>/dev/null || true
        wait "$gpu_pid" 2>/dev/null || true
      fi

      # "Worker completed: beta = X, MMD = Y" -> one runs.csv row per job
      betas_out="$(mktemp)"; mmds_out="$(mktemp)"
      grep -oP 'Worker completed: beta = \K[0-9.eE+-]+(?=, MMD = )' "$log_file" > "$betas_out" || true
      grep -oP 'Worker completed: beta = [0-9.eE+-]+, MMD = \K[0-9.eE+-]+' "$log_file" > "$mmds_out" || true
      paste -d, "$betas_out" "$mmds_out" | while IFS=, read -r beta mmd; do
        [[ -z "$beta" ]] && continue
        echo "$run_id,$ts,$workers,$gpus,$beta,$mmd" >> "$RUNS_CSV"
      done
      rm -f "$betas_out" "$mmds_out"

      reported_seconds=$(grep -oP 'All jobs are completed in \K[0-9]+(?= seconds)' "$log_file" || echo "")
      num_betas=$(awk -F',' '{print NF}' <<< "$betas")

      gpu_avg=""; gpu_max=""; mem_avg=""
      if [[ -s "$gpu_file" ]]; then
        gpu_avg=$(awk -F',' '{gsub(/ /,"",$3); sum+=$3; n++} END{if(n) printf "%.1f", sum/n}' "$gpu_file")
        gpu_max=$(awk -F',' '{gsub(/ /,"",$3); if($3+0>max) max=$3+0} END{print max}' "$gpu_file")
        mem_avg=$(awk -F',' '{gsub(/ /,"",$5); sum+=$5; n++} END{if(n) printf "%.1f", sum/n}' "$gpu_file")
      fi

      echo "$run_id,$ts,$workers,$gpus,$num_betas,$reported_seconds,$wall_seconds,$gpu_avg,$gpu_max,$mem_avg" >> "$SUMMARY_CSV"
    done
  done
done

echo "Done."
echo "  per-job results:  $RUNS_CSV"
echo "  per-run summary:  $SUMMARY_CSV"
echo "  raw logs:         $LOG_DIR/"
echo "  raw gpu samples:  $GPU_DIR/"
