## Repository layout

```
src/search/heuristics/h2_heuristic.{cc,h}      - classical CPU h^2 baseline
src/search/heuristics/hm2_base.{cc,h}          - shared base: hypergraph build, dominance pruning
src/search/heuristics/hyperedge.{cc,h}         - hyperedge data structure
src/search/heuristics/hm2_torch.{cc,h}         - PyTorch host-side h^m (batching, CP, fixed-point)
src/search/heuristics/hm2_torch_cuda.{cu,h}    - fused CUDA kernels (max-add, grouped-amin)
experiments/socs26/run_paper_experiments.py    - Lab script reproducing all six paper configs
experiments/socs26/project.py                  - shared Lab settings + benchmark suites
```
## Build

```bash
# 1. Download libtorch 2.1.2 (CUDA 12.1, cxx11 ABI -- the build is compiled with
#    -D_GLIBCXX_USE_CXX11_ABI=1).
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.2%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.2+cu121.zip
export LIBTORCH=$PWD/libtorch

# 2. Make sure CUDA 12.1 toolchain is available.
export CUDA_HOME=/path/to/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH                          # nvcc on PATH
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib:$LIBRARY_PATH
export TORCH_CUDA_ARCH_LIST="8.0"                         # A100; adjust to your GPU CC

# 3. Build (requires cmake >= 3.16 and gcc/g++ >= 12.3 supporting C++20).
CMAKE_PREFIX_PATH=$LIBTORCH \
LD_LIBRARY_PATH=$LIBTORCH/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
./build.py release
```

## Running a single planner call

```bash
LD_LIBRARY_PATH=$LIBTORCH/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH \
./fast-downward.py domain.pddl problem.pddl \
    --search "astarbatched(hm2torch(m=2,use_gpu=true,cp=0,n_cp=1))"
```

The six paper configurations are:

## Reproducing the paper experiments

The paper used [Lab](https://lab.readthedocs.io/) (from the AI Group, University
of Basel) to drive batched IPC-suite runs on SLURM (Tetralith / Berzelius at
NSC, Sweden). The reproducer script `experiments/socs26/run_paper_experiments.py`
defines exactly the six configurations above and runs them over the full
optimal STRIPS suite.

**Required environment variables** (the script aborts with a hint if any
are missing):

| Variable               | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `DOWNWARD_BENCHMARKS`  | Local clone of github.com/aibasel/downward-benchmarks (the IPC suite). |
| `CUDA_HOME`            | CUDA 12.1 toolkit root (must contain `bin/nvcc`).                       |
| `LIBTORCH`             | Unpacked libtorch 2.1.2 cu121 cxx11-ABI tree.                           |
| `SLURM_ACCOUNT`        | Your SLURM account / project (e.g. `berzelius-2026-49`).                |
| `LAB_EMAIL`            | Contact email Lab puts in `#SBATCH` directives.                         |

```bash
export DOWNWARD_BENCHMARKS=/path/to/downward-benchmarks
export CUDA_HOME=/path/to/cuda-12.1
export LIBTORCH=/path/to/libtorch
export SLURM_ACCOUNT=your-project
export LAB_EMAIL=you@example.org

# Build the planner + create one Lab run dir per (config, instance):
uv run --project . python experiments/socs26/run_paper_experiments.py build
# Submit the runs to SLURM (rejected if total > MaxArraySize -- see below):
uv run --project . python experiments/socs26/run_paper_experiments.py start
# Parse + fetch results:
uv run --project . python experiments/socs26/run_paper_experiments.py parse fetch
```

