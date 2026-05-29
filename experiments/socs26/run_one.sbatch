#!/bin/bash
#SBATCH --job-name=socs26_one
#SBATCH --account=berzelius-2026-49
#SBATCH --partition=berzelius
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:15:00
#SBATCH --output=run_one.%j.out
#SBATCH --error=run_one.%j.err

# Single Lab-generated run, executed directly. Useful for verifying the
# end-to-end pipeline (Lab run dir layout, fast-downward.py call, GPU
# heuristic) without launching the full SLURM array.
#
# By default targets run #00106 = gpu-batched-h2 on blocks/probBLOCKS-4-0.pddl.
# Override RUN_ID to pick a different run (see static-properties).

set -e
RUN_ID=${RUN_ID:-00106}
EXP=/proj/gpuplan/projects/downward_socs26/experiments/socs26/data/run_paper_experiments
# Find which "runs-XXXXX-YYYYY" subdir holds RUN_ID
SHARD=$(printf '%s\n' $EXP/runs-* | while read d; do
  [ -d "$d/$RUN_ID" ] && echo "$d" && break
done)
if [ -z "$SHARD" ]; then
  echo "ERROR: run $RUN_ID not found under $EXP/runs-*" >&2
  exit 1
fi
RUN_DIR=$SHARD/$RUN_ID

echo "=== start: $(date) JOBID=$SLURM_JOB_ID NODE=$(hostname) ==="
echo "=== run dir: $RUN_DIR ==="
cat "$RUN_DIR/static-properties" | head -8

# Build env (matches the build step the Lab "build" target used).
export LIBTORCH=/proj/gpuplan/users/x_mfrit/libtorch
export CUDA_HOME=/software/sse/manual/CUDA/12.1.1_530.30.02
export PATH=/software/sse/manual/GCC/12.3.0/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LIBTORCH/lib:$CUDA_HOME/lib64:/software/sse/manual/GCC/12.3.0/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$CUDA_HOME/lib64
export CMAKE_PREFIX_PATH=$LIBTORCH
export TORCH_CUDA_ARCH_LIST=8.0

# Execute the exact Lab-generated run script.
cd "$RUN_DIR"
uv run --project /proj/gpuplan/projects/downward_socs26 python ./run
rc=$?
echo "=== run exit=$rc  date=$(date) ==="

# Print interesting output if present.
[ -f run.log ] && { echo "=== run.log (last 30) ==="; tail -30 run.log; }
[ -f run.err ] && { echo "=== run.err (last 10) ==="; tail -10 run.err; }

echo "=== sacct ==="
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed -P
