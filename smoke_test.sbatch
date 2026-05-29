#!/bin/bash
#SBATCH --job-name=socs26_smoke
#SBATCH --account=berzelius-2026-49
#SBATCH --partition=berzelius
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=00:15:00
#SBATCH --output=smoke_test.%j.out
#SBATCH --error=smoke_test.%j.err

set -e
echo "=== smoke test start: $(date) JOBID=$SLURM_JOB_ID NODE=$(hostname) ==="

export REPO=/proj/gpuplan/projects/downward_socs26
export LIBTORCH=/proj/gpuplan/users/x_mfrit/libtorch
export CUDA_HOME=/software/sse/manual/CUDA/12.1.1_530.30.02
export PATH=/software/sse/manual/GCC/12.3.0/bin:$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$LIBTORCH/lib:$CUDA_HOME/lib64:/software/sse/manual/GCC/12.3.0/lib64:$LD_LIBRARY_PATH

cd $REPO

# Single FD invocation on gripper/prob01.pddl with B-h2 on the GPU (matches
# the paper's main GPU config).
./fast-downward.py \
    /proj/gpuplan/users/x_mfrit/downward-benchmarks/gripper/domain.pddl \
    /proj/gpuplan/users/x_mfrit/downward-benchmarks/gripper/prob01.pddl \
    --search "astarbatched(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))"

rc=$?
echo "=== smoke test end: $(date)  rc=$rc ==="
echo "--- sacct ---"
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,ReqMem,Elapsed -P
