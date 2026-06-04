#!/usr/bin/env python
"""Reproduce all six configurations evaluated in the SoCS-2026 paper

  "Parallelizing Classical Planning: Critical Path Heuristics on the GPU"

The configurations correspond to Table 1 of the paper:

    CPU:                    GPU (PyTorch backend):
      h2()                    astar(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))   -- "h^2"
      lmcut()                 astarbatched(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))  -- "B-h^2"
                              astar(hm2torch(m=2,use_gpu=true,cp=1,n_cp=5))   -- "h^2_CPr"
                              astar(hm2torch(m=2,use_gpu=true,cp=2,n_cp=5))   -- "h^2_CPg"

IMPORTANT - one task per SLURM array job (see ../README.md):
Fast Downward and SLURM's OOM handler interact poorly when multiple FD runs
share a SLURM job allocation: a kernel OOM event on the first run can leave the
allocation in a state where subsequent runs in the same array task never get
scheduled. This script therefore forces a single FD run per array task
via `ENV.MAX_TASKS = len(exp.runs)`. If your cluster has a low maximum
array size, split the suite across multiple invocations of this script
(see SUITE_OPTIMAL_STRIPS1 / SUITE_OPTIMAL_STRIPS2 in project.py for a
ready-made split, mirroring the original paper experiments).
"""
import os
import subprocess

import project

from lab.environments import TetralithEnvironment


REPO = project.get_repo_base()


def _require_env_dir(name: str, hint: str) -> str:
    """Return $name pointing to an existing directory or abort with a hint."""
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Required environment variable ${name} is not set. {hint}"
        )
    if not os.path.isdir(value):
        raise SystemExit(
            f"${name}={value!r} is not a directory. {hint}"
        )
    return value


def _require_env(name: str, hint: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(
            f"Required environment variable ${name} is not set. {hint}"
        )
    return value


BENCHMARKS_DIR = _require_env_dir(
    "DOWNWARD_BENCHMARKS",
    "Set it to your local clone of github.com/aibasel/downward-benchmarks "
    "(the IPC optimal-track STRIPS suite).",
)

# Build environment. These must point to the libtorch unpacked tree and the
# CUDA toolkit root respectively; both are also needed at runtime to resolve
# shared libs.
CUDA_HOME = _require_env_dir(
    "CUDA_HOME",
    "Set it to your CUDA 12.1 toolkit root (the directory that contains "
    "bin/nvcc).",
)
LIBTORCH = _require_env_dir(
    "LIBTORCH",
    "Set it to the unpacked libtorch tree (libtorch 2.1.2 with the cxx11 "
    "ABI; cu121 variant required for the GPU heuristic).",
)

env_updates = {
    "CUDA_HOME": CUDA_HOME,
    "CPATH": f"{CUDA_HOME}/include:" + os.environ.get("CPATH", ""),
    "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{LIBTORCH}/lib:"
        + os.environ.get("LD_LIBRARY_PATH", ""),
    # A100 (Ampere). Set to your GPU's CC if reproducing elsewhere.
    "TORCH_CUDA_ARCH_LIST": "8.0",
    "CC": "gcc",
    "CXX": "g++",
    "CMAKE_PREFIX_PATH": LIBTORCH,
}
os.environ.update(env_updates)

# ---------------------------------------------------------------------------
# SLURM environment. The single-task-per-array policy is enforced below by
# pinning ENV.MAX_TASKS to the total number of runs.
# ---------------------------------------------------------------------------
SLURM_ACCOUNT = _require_env(
    "SLURM_ACCOUNT",
    "Set it to your SLURM account/project name (e.g., berzelius-2026-49).",
)
LAB_EMAIL = _require_env(
    "LAB_EMAIL",
    "Set it to the contact email Lab should put in #SBATCH directives.",
)

ENV = TetralithEnvironment(
    email=LAB_EMAIL,
    memory_per_cpu="8G",
    partition="berzelius",
    extra_options=(
        "#SBATCH --cpus-per-task=1\n"
        "#SBATCH --gpus=1\n"
        f"#SBATCH --account={SLURM_ACCOUNT}\n"
    ),
    time_limit_per_task="01:00:00",
)


# ---------------------------------------------------------------------------
# Configurations as evaluated in the paper.
# ---------------------------------------------------------------------------
CONFIGS = [
    # CPU baselines
    ("cpu-h2",   ["--search", "astar(h2())"]),
    ("cpu-lmcut", ["--search", "astar(lmcut())"]),
    # GPU
    ("gpu-h2",
     ["--search", "astar(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))"]),
    ("gpu-batched-h2",
     ["--search", "astarbatched(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))"]),
    ("gpu-h2-cp-random",
     ["--search", "astar(hm2torch(m=2,use_gpu=true,cp=1,n_cp=5))"]),
    ("gpu-h2-cp-goal-facts",
     ["--search", "astar(hm2torch(m=2,use_gpu=true,cp=2,n_cp=5))"]),
]

DRIVER_OPTIONS = [
    "--overall-memory-limit", "100G",   # not enforced; SLURM cgroup is the real cap
    "--overall-time-limit",  "30m",
    "--build", "release",
]
BUILD_OPTIONS = []

# Single revision = current HEAD. Lab requires a "revision nickname" used to
# disambiguate algorithms across revisions; the empty string is fine for a
# single-revision run, but Lab still requires a non-empty string -- use HEAD.
REV = "HEAD"


# Paper Table-1 attributes.
ATTRIBUTES = [
    project.EVALUATIONS_PER_TIME,
    "cost",
    "coverage",
    "initial_h_values",
    "error",
    "evaluations",
    "expansions",
    "expansions_until_last_jump",
    "generated",
    "memory",
    "planner_memory",
    "planner_time",
    "quality",
    "run_dir",
    "score_evaluations",
    "score_expansions",
    "score_generated",
    "score_memory",
    "score_search_time",
    "score_total_time",
    "search_time",
    "total_time",
]


def main():
    exp = project.FastDownwardExperiment(environment=ENV)
    for nick, search_args in CONFIGS:
        exp.add_algorithm(
            nick, REPO, REV, search_args,
            build_options=BUILD_OPTIONS,
            driver_options=DRIVER_OPTIONS,
        )

    # Full optimal STRIPS suite, identical to the paper.
    exp.add_suite(BENCHMARKS_DIR, project.SUITE_OPTIMAL_STRIPS)

    exp.add_parser(exp.EXITCODE_PARSER)
    exp.add_parser(exp.TRANSLATOR_PARSER)
    exp.add_parser(exp.SINGLE_SEARCH_PARSER)
    exp.add_parser(exp.PLANNER_PARSER)

    # --- enforce one FD run per SLURM array task ---
    # Pin MAX_TASKS to the total run count so Lab issues one SBATCH array
    # element per FD invocation (see README for the cgroup-OOM rationale).
    # If the total exceeds the cluster's MaxArraySize, the build still
    # succeeds but submission will fail; the user has to split the suite
    # (see project.SUITE_OPTIMAL_STRIPS1 / SUITE_OPTIMAL_STRIPS2).
    exp._add_runs()
    run_count = len(exp.runs)
    try:
        scontrol = subprocess.run(
            ["scontrol", "show", "config"], stdout=subprocess.PIPE, check=False
        )
        for line in scontrol.stdout.decode().splitlines():
            if "MaxArraySize" in line:
                max_array = int(line.split()[-1])
                if run_count > max_array:
                    print(
                        f"[run_paper_experiments] WARNING: {run_count} runs "
                        f"exceeds cluster MaxArraySize={max_array}. Build will "
                        "still produce all run directories, but `start` will be "
                        "rejected by sbatch. Split via SUITE_OPTIMAL_STRIPS1 / "
                        "SUITE_OPTIMAL_STRIPS2 in project.py and re-run the "
                        "script once per shard."
                    )
                break
    except FileNotFoundError:
        pass
    ENV.MAX_TASKS = run_count
    exp.runs = []
    # ------------------------------------------------

    exp.add_step("build",  exp.build)
    exp.add_step("start",  exp.start_runs)
    exp.add_step("parse",  exp.parse)
    exp.add_fetcher(name="fetch")
    project.add_absolute_report(exp, attributes=ATTRIBUTES,
                                filter=[project.add_evaluations_per_time])

    exp.run_steps()


if __name__ == "__main__":
    main()
