#!/usr/bin/env python
"""Local smoke test for SoCS-2026 GPU h^m heuristic.

Adapted from run_paper_experiments.py for a single local NixOS box:
  - LocalEnvironment (one FD process at a time, no SLURM).
  - Tiny suite (blocks + gripper) so a full pass finishes in minutes.
  - LIBTORCH / CUDA_HOME come from the surrounding nix-shell (see shell.nix).
  - GPU configs are still built so build issues surface, but they will crash
    at runtime on hosts without an NVIDIA device. That's expected for a
    smoke test -- the build path is what we care about here.

Launch from the repo root:

    nix-shell --run "uv sync && uv run python experiments/socs26/run_smoke.py"
"""
import os

import project

from lab.environments import LocalEnvironment


REPO = project.get_repo_base()

BENCHMARKS_DIR = os.environ.get(
    "DOWNWARD_BENCHMARKS",
    os.path.expanduser("~/data/downward-benchmarks"),
)

# shell.nix exports LIBTORCH (out, has lib/) and LIBTORCH_DEV (has include/,
# share/cmake/Torch/). Make sure both ends up on CMAKE_PREFIX_PATH and that
# lib/ is on LD_LIBRARY_PATH. shell.nix has already done this for the
# surrounding shell; we redo it defensively in case the script is invoked
# without sourcing the nix shellHook first.
LIBTORCH = os.environ.get("LIBTORCH")
LIBTORCH_DEV = os.environ.get("LIBTORCH_DEV")
prefixes = [p for p in (LIBTORCH, LIBTORCH_DEV) if p]
if prefixes:
    os.environ["CMAKE_PREFIX_PATH"] = (
        ":".join(prefixes) + ":" + os.environ.get("CMAKE_PREFIX_PATH", "")
    ).rstrip(":")
if LIBTORCH:
    os.environ["LD_LIBRARY_PATH"] = (
        f"{LIBTORCH}/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    ).rstrip(":")


ENV = LocalEnvironment(processes=1)


CONFIGS = [
    ("cpu-h2",    ["--search", "astar(h2())"]),
    ("cpu-lmcut", ["--search", "astar(lmcut())"]),
    ("gpu-h2",
     ["--search", "astar(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))"]),
    ("gpu-batched-h2",
     ["--search", "astarbatched(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))"]),
]

DRIVER_OPTIONS = [
    "--overall-memory-limit", "8G",
    "--overall-time-limit",  "5m",
    "--build", "release",
]
BUILD_OPTIONS = []
REV = "HEAD"


ATTRIBUTES = [
    project.EVALUATIONS_PER_TIME,
    "cost", "coverage", "error", "evaluations", "expansions",
    "generated", "memory", "planner_time", "run_dir", "search_time",
    "total_time",
]

SUITE = ["blocks", "gripper"]


def main():
    exp = project.FastDownwardExperiment(environment=ENV)
    for nick, search_args in CONFIGS:
        exp.add_algorithm(
            nick, REPO, REV, search_args,
            build_options=BUILD_OPTIONS,
            driver_options=DRIVER_OPTIONS,
        )

    exp.add_suite(BENCHMARKS_DIR, SUITE)

    exp.add_parser(exp.EXITCODE_PARSER)
    exp.add_parser(exp.TRANSLATOR_PARSER)
    exp.add_parser(exp.SINGLE_SEARCH_PARSER)
    exp.add_parser(exp.PLANNER_PARSER)

    exp.add_step("build", exp.build)
    exp.add_step("start", exp.start_runs)
    exp.add_step("parse", exp.parse)
    exp.add_fetcher(name="fetch")
    project.add_absolute_report(
        exp, attributes=ATTRIBUTES, filter=[project.add_evaluations_per_time]
    )

    exp.run_steps()


if __name__ == "__main__":
    main()
