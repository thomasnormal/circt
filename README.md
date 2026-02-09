<p align="center"><img src="docs/includes/img/circt-logo.svg"/></p>

[![](https://github.com/llvm/circt/actions/workflows/buildAndTest.yml/badge.svg?event=push)](https://github.com/llvm/circt/actions?query=workflow%3A%22Build+and+Test%22)
[![Nightly integration tests](https://github.com/llvm/circt/workflows/Nightly%20integration%20tests/badge.svg)](https://github.com/llvm/circt/actions?query=workflow%3A%22Nightly+integration+tests%22)

---

## Fork Features (thomasnormal/circt vs llvm/circt)

This fork extends CIRCT with enhanced **UVM and SystemVerilog** support for running UVM testbenches. Key additions:

### UVM/Class Support
- **Class OOP** - Full class hierarchy, inheritance, virtual methods, vtable resolution
- **Parameterized classes** - UVM-style parameterized classes with proper specialization
- **Static properties** - Per-specialization static variables with correct scoping
- **Factory pattern support** - Virtual class vtables, method lookup in parameterized classes

### Interface & Module Support
- **Virtual interfaces** - Full virtual interface support with ref→vif conversion
- **Interface member lvalue access** - Proper llhd::ProbeOp generation for interface signals
- **Modport support** - Complete modport handling

### Constraint Randomization
- **94% constraint coverage** - Range, multi-range, soft constraints
- **All 10 constraint ops** - Lower to runtime solver calls
- **Inline implication constraints** - Supported via runtime
- **Complex predicates** - SMT solver integration for remaining 6%

### File I/O & System Functions
- **$fopen/$fwrite/$fclose** - Complete file I/O support
- **$display/$format** - Full format specifier support including width
- **$finish/$fatal** - Proper sim.terminate handling in seq.initial

### Initial Blocks & Processes
- **seq.initial support** - Simple initial blocks run through arcilator
- **$finish in seq.initial** - No longer forces llhd.process fallback
- **sim.proc.print** - $display works in arcilator simulation

### Coverage & Assertions
- **Covergroup infrastructure** - CovergroupHandleType, CovergroupInstOp, CovergroupSampleOp
- **SVA dialect** - Basic assertion ops for verification

### LSP/IDE Support
- **UVM path support** - `--uvm-path` flag and `UVM_HOME` environment variable
- **Interface symbols** - LSP returns proper interface structure
- **Debounce fix** - LSP no longer hangs on large files

### MooreToCore Lowering
- **100% UVM parsing** - 161K+ lines of Moore IR from uvm_pkg.sv
- **0 lowering errors** - Full MooreToCore conversion for UVM
- **All AVIPs pass** - APB, AHB, AXI4, AXI4-Lite, UART, I2S, I3C, SPI

### Misc Fixes
- **Array locator inline loops** - Complex predicates via scf.for
- **llhd.time data layout** - Structs with time fields handled correctly
- **RefType for dynamic structs** - Proper GEP for strings/queues
- **Mem2Reg loop-local variables** - Dominance errors fixed
- **realtobits/bitstoreal** - Conversion patterns for real types

For full details, see [PROJECT_PLAN.md](PROJECT_PLAN.md) and [docs/CHANGES.md](docs/CHANGES.md).

---

# ⚡️ "CIRCT" / Circuit IR Compilers and Tools

"CIRCT" stands for "Circuit Intermediate Representations (IR) Compilers and Tools".  One might also interpret
it as the recursively as "CIRCT IR Compiler and Tools".  The T can be
selectively expanded as Tool, Translator, Team, Technology, Target, Tree, Type,
... we're ok with the ambiguity.

The CIRCT community is an open and welcoming community.  If you'd like to
participate, you can do so in a number of different ways:

1) Join our [Discourse Forum](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/) 
on the LLVM Discourse server.  To get a "mailing list" like experience click the 
bell icon in the upper right and switch to "Watching".  It is also helpful to go 
to your Discourse profile, then the "emails" tab, and check "Enable mailing list 
mode".  You can also do chat with us on [CIRCT channel](https://discord.com/channels/636084430946959380/742572728787402763) 
of LLVM discord server.

2) Join our weekly video chat.  Please see the
[meeting notes document](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#)
for more information.

3) Contribute code.  CIRCT follows all of the LLVM Policies: you can create pull
requests for the CIRCT repository, and gain commit access using the [standard LLVM policies](https://llvm.org/docs/DeveloperPolicy.html#obtaining-commit-access).

## Motivation

The EDA industry has well-known and widely used proprietary and open source
tools.  However, these tools are inconsistent, have usability concerns, and were
not designed together into a common platform.  Furthermore
these tools are generally built with
[Verilog](https://en.wikipedia.org/wiki/Verilog) (also
[VHDL](https://en.wikipedia.org/wiki/VHDL)) as the IRs that they
interchange.  Verilog has well known design issues, and limitations, e.g.
suffering from poor location tracking support.

The CIRCT project is an (experimental!) effort looking to apply MLIR and
the LLVM development methodology to the domain of hardware design tools.  Many
of us dream of having reusable infrastructure that is modular, uses
library-based design techniques, is more consistent, and builds on the best
practices in compiler infrastructure and compiler design techniques.

By working together, we hope that we can build a new center of gravity to draw
contributions from the small (but enthusiastic!) community of people who work
on open hardware tooling.  In turn we hope this will propel open tools forward,
enables new higher-level abstractions for hardware design, and
perhaps some pieces may even be adopted by proprietary tools in time.

For more information, please see our longer [charter document](docs/Charter.md).

## Getting Started

To get started hacking on CIRCT quickly, run the following commands. If you want to include `circt-verilog` in the build, add `-DCIRCT_SLANG_FRONTEND_ENABLED=ON` to the cmake call:

```sh
# Clone the repository and its submodules
git clone git@github.com:llvm/circt.git --recursive
cd circt

# Configure the build
cmake -G Ninja llvm/llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
    -DLLVM_ENABLE_LLD=ON
```

If you want to build everything about the CIRCT tools and libraries, run below command(also runs all tests):
```
ninja -C build check-circt
```

If you want to only build a specific part, for example the `circt-opt` tool:
```sh
ninja -C build bin/circt-opt
```

or the `firtool` tool:
```sh
ninja -C build bin/firtool
```

This will only build the necessary parts of LLVM, MLIR, and CIRCT, which can be a lot quicker than building everything.

### Dependencies

If you have git, ninja, python3, cmake, and a C++ toolchain installed, you should be able to build CIRCT.
For a more detailed description of dependencies, take a look at:

- [Getting Started with MLIR](https://mlir.llvm.org/getting_started/)
- [LLVM Requirements](https://llvm.org/docs/GettingStarted.html#requirements)

### Useful Options

The `-DCMAKE_BUILD_TYPE=Debug` flag enables debug information, which makes the whole tree compile slower, but allows you to step through code into the LLVM
and MLIR frameworks.
The `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON` flag generates a `compile_commands.json` file, which can be used by editors and language servers for autocomplete and other IDE-like features.

To get something that runs faster but is still easy to debug, use the `-DCMAKE_BUILD_TYPE=RelWithDebInfo` flag to do a release build with debug info.

To do a release build that runs very fast, use the `-DCMAKE_BUILD_TYPE=Release` flag.
Release mode makes a very large difference in performance.

#### Tool Resource Guard (Memory/Time Caps)

Most CIRCT command-line tools enable a built-in "resource guard" by default to
avoid runaway memory growth (e.g., IR explosion / non-convergent pipelines)
from consuming tens of GB of RAM and hanging a machine.

- Disable: `--no-resource-guard`
- Adjust limits: `--max-rss-mb=...` / `--max-malloc-mb=...` / `--max-vmem-mb=...` /
  `--max-wall-ms=...` (or `CIRCT_MAX_RSS_MB`, `CIRCT_MAX_WALL_MS`, ...)
- Note: when enabled, the guard also applies a best-effort address-space cap by
  default (derived from the effective RSS limit). Use `--max-vmem-mb=0` (or
  `CIRCT_MAX_VMEM_MB=0`) to disable it.
- Diagnose effective limits: `--resource-guard-verbose` (or `CIRCT_RESOURCE_GUARD_VERBOSE=1`)

Consult the [Getting Started](docs/GettingStarted.md) page for detailed information on configuring and compiling CIRCT.

Consult the [Python Bindings](docs/PythonBindings.md) page if you are mainly interested in using CIRCT from a Python prompt or script.

### Mutation Tools (Cover + Matrix)

This fork includes mutation tooling for certitude-style fault classification
using formal pre-qualification (BMC/LEC) and dynamic tests.

Build the tools:

```sh
ninja -C build circt-mut circt-bmc circt-lec circt-verilog
```

Preferred frontend: `circt-mut` (subcommands: `cover`, `matrix`, `generate`).
Legacy script entrypoints under `utils/` remain supported for compatibility.
`circt-mut generate` now has a native execution path for core generation
options, including native generated-mutation caching via `--cache-dir`.
Unsupported future options still fall back to the script backend during
migration.

Run a single mutation campaign:

```sh
circt-mut cover \
  --design /path/to/design.il \
  --mutations-file /path/to/mutations.txt \
  --tests-manifest /path/to/tests.tsv \
  --formal-global-propagate-circt-chain auto \
  --formal-global-propagate-timeout-seconds 60 \
  --work-dir /tmp/mutation-cover
```

`circt-lec` / `circt-bmc` discovery is automatic when you enable built-in
global filters:
- first from install-tree sibling `bin/` (when running installed scripts)
- then from `PATH`
- then from `./build/bin` in this CIRCT checkout
- you can still override with explicit paths.
`circt-mut cover` now performs this built-in-tool resolution natively and
rewrites bare `--formal-global-propagate-circt-lec` /
`--formal-global-propagate-circt-bmc` to explicit executables before dispatch,
so unresolved toolchains fail fast with a direct CLI diagnostic.
It also runs native global-filter preflight checks:
- validates `--formal-global-propagate-circt-chain` mode values
- auto-injects LEC/BMC built-in tools for chain mode when omitted
- rejects conflicting non-chain global filter mode combinations early.
`circt-mut matrix` now applies the same preflight model for default global
filter options (`--default-formal-global-propagate-circt-*`) before dispatch.
It also pre-resolves `--default-mutations-yosys` so generated-mutation lanes
fail fast if the default Yosys executable is unavailable.
Generated lanes in `--lanes-tsv` now also get native preflight validation for
lane `mutations_yosys` values before script dispatch.
Lane-level formal tool fields in `--lanes-tsv` now also get native preflight
validation (`global_propagate_circt_lec`, `global_propagate_circt_bmc`,
`global_propagate_z3`, `global_propagate_bmc_z3`) with effective default
fallback semantics.
Both `circt-mut cover` and `circt-mut matrix` now also pre-resolve explicit Z3
options for built-in filters (`--formal-global-propagate-z3`,
`--formal-global-propagate-bmc-z3`, and default matrix variants) and fail fast
if unresolved.
`circt-mut cover` also pre-resolves `--mutations-yosys`, and native
`circt-mut generate` now fail-fast resolves `--yosys` before execution.

For CI robustness, set `--formal-global-propagate-timeout-seconds <N>` to cap
global formal filter wall time per mutant. Timeout outcomes are classified
conservatively as `propagated` (mutants are not pruned).
Use `--formal-global-propagate-lec-timeout-seconds <N>` and
`--formal-global-propagate-bmc-timeout-seconds <N>` when LEC/BMC need different
budgets; they override the global timeout for built-in LEC/BMC modes.

For installed toolchains, `circt-mut` expects mutation workflow scripts under
`<prefix>/share/circt/utils` and CIRCT now installs them there by default.

Use auto-generated mutations instead of a prebuilt list:

```sh
circt-mut cover \
  --design /path/to/design.il \
  --tests-manifest /path/to/tests.tsv \
  --generate-mutations 1000 \
  --mutations-yosys yosys \
  --mutations-modes arith,control \
  --mutations-seed 1
```

Run multiple lanes with `circt-mut matrix`:

```sh
circt-mut matrix \
  --lanes-tsv /path/to/lanes.tsv \
  --out-dir /tmp/mutation-matrix \
  --default-mutations-yosys yosys \
  --default-formal-global-propagate-circt-lec \
  --default-formal-global-propagate-circt-bmc \
  --default-formal-global-propagate-circt-chain auto \
  --default-formal-global-propagate-timeout-seconds 60 \
  --skip-baseline \
  --fail-on-undetected \
  --include-lane-regex '^sv-tests|^verilator' \
  --exclude-lane-regex 'slow'
```

Lane TSVs can now override strict matrix gates per lane via tail columns:
`skip_baseline`, `fail_on_undetected`, `fail_on_errors`.

Mutation materialization is built in by default via
`utils/create_mutated_yosys.sh`. Override with `--create-mutated-script` only
if you need MCY/external compatibility behavior.

Command mapping by workflow:

1. Single mutation campaign

`circt-mut cover`:

```sh
circt-mut cover \
  --design /path/to/design.il \
  --mutations-file /path/to/mutations.txt \
  --tests-manifest /path/to/tests.tsv \
  --formal-global-propagate-circt-chain auto \
  --formal-global-propagate-timeout-seconds 60 \
  --work-dir /tmp/mutation-cover
```

Equivalent `mcy` flow:

```sh
cd /path/to/mcy_project
mcy init
mcy run -j8
```

Equivalent Certitude-style flow (schematic; actual command names/flags vary by release/site integration):

```sh
certitude_run \
  -rtl /path/to/filelist.f \
  -tb /path/to/testlist.tcl \
  -fault_model rtl_mutation \
  -out /tmp/certitude-run
```

2. Increase generated mutation count

`circt-mut cover`:

```sh
circt-mut cover \
  --design /path/to/design.il \
  --tests-manifest /path/to/tests.tsv \
  --generate-mutations 1000 \
  --mutations-yosys yosys \
  --mutations-modes arith,control \
  --mutations-seed 1
```

Equivalent `mcy` flow (set `size 1000` in `config.mcy`, then regenerate tasks):

```sh
cd /path/to/mcy_project
mcy reset
mcy run -j8
```

Equivalent Certitude-style flow (schematic):

```sh
certitude_run \
  -rtl /path/to/filelist.f \
  -tb /path/to/testlist.tcl \
  -num_faults 1000 \
  -fault_scope arith,control \
  -out /tmp/certitude-run
```

3. Multi-lane/CI orchestration

`circt-mut matrix`:

```sh
circt-mut matrix \
  --lanes-tsv /path/to/lanes.tsv \
  --out-dir /tmp/mutation-matrix \
  --default-mutations-yosys yosys \
  --default-formal-global-propagate-circt-lec \
  --default-formal-global-propagate-circt-bmc \
  --default-formal-global-propagate-circt-chain auto \
  --default-formal-global-propagate-timeout-seconds 60 \
  --include-lane-regex '^sv-tests|^verilator' \
  --exclude-lane-regex 'slow'
```

Equivalent `mcy` flow (one project directory per lane, orchestrated by shell/CI):

```sh
for lane_dir in /path/to/lanes/*; do
  (
    cd "$lane_dir"
    mcy init
    mcy run -j8
  )
done
```

Equivalent Certitude-style flow (schematic):

```sh
for lane in lane_svtests lane_verilator; do
  certitude_run -config "/path/to/${lane}.cfg" -out "/tmp/${lane}"
done
```

Core input formats:
- `tests.tsv`:
  `test_id<TAB>run_cmd<TAB>result_file<TAB>kill_regex<TAB>survive_regex`
- `mutations.txt`:
  `mutation_id<space>mutate_spec`
- `lanes.tsv`:
  first columns are
  `lane_id<TAB>design<TAB>mutations_file<TAB>tests_manifest<...>`.
  Generated-mutation lanes can leave `mutations_yosys` as `-` to inherit
  matrix `--default-mutations-yosys`.
  with optional tail override `global_propagate_timeout_seconds` to set
  per-lane global formal timeout.
  Additional optional tail overrides:
  `global_propagate_lec_timeout_seconds`,
  `global_propagate_bmc_timeout_seconds`.

Outputs are written under `--work-dir` / `--out-dir` and include
`summary.tsv`, `pair_qualification.tsv`, `results.tsv`, `metrics.tsv`, and
`summary.json`. Matrix runs also emit gate-status aggregation in
`gate_summary.tsv`.

`metrics.tsv`/`summary.json` include global-filter runtime telemetry:
- `global_filter_lec_runtime_ns`
- `global_filter_bmc_runtime_ns`
- `global_filter_cmd_runtime_ns`
- `global_filter_lec_runs`
- `global_filter_bmc_runs`
- `global_filter_cmd_runs`

When `--reuse-cache-dir` is set, generated mutation lists are now cached by
design+generator options to avoid repeated Yosys mutation-list synthesis across
runs/lanes.
Within each generation round, CIRCT now batches all mode/profile `mutate -list`
requests into a single Yosys invocation to reduce process startup overhead.
Generation cache writes are now coordinated with per-key locking, so parallel
matrix lanes with identical generation settings reuse one synthesis result
instead of duplicating `yosys mutate -list` work.
Use `run_mutation_matrix.sh --lane-schedule-policy cache-aware` to prioritize
one lane per generated-cache key before same-key followers and reduce cache
lock contention under high `--lane-jobs`.
`metrics.tsv` now reports generated-list cache usage via:
- `generated_mutations_cache_status` (`hit|miss|disabled`)
- `generated_mutations_cache_hit`
- `generated_mutations_cache_miss`
- `generated_mutations_runtime_ns`
- `generated_mutations_cache_saved_runtime_ns`
- `generated_mutations_cache_lock_wait_ns`
- `generated_mutations_cache_lock_contended`

For full option reference, global-filter modes, cache/reuse controls, and the
complete lane TSV schema, see `docs/FormalRegression.md`.

### Submodules

CIRCT contains LLVM as a git submodule.
The LLVM repository here includes staged changes to MLIR which may be necessary to support CIRCT.
It also represents the version of LLVM that has been tested.
MLIR is still changing relatively rapidly, so feel free to use the current version of LLVM, but APIs may have changed.

Whenever you checkout a new CIRCT branch that points to a different version of LLVM, run the following command to update the submodule:
```sh
git submodule update
```

The repository is set up to perform a shallow clone of the submodules, meaning it downloads just enough of the LLVM repository to check out the currently specified commit.
If you wish to work with the full history of the LLVM repository, you can manually "unshallow" the submodule:
```sh
cd llvm
git fetch --unshallow
```

### Building LLVM/MLIR Separately

You can also build LLVM/MLIR in isolation first, and then build CIRCT using that first build.
This allows you to pick different compiler options for the two builds, such as building CIRCT in debug mode but LLVM/MLIR in release mode.

First, build and test *LLVM/MLIR*:
```sh
cd llvm
cmake -G Ninja llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host
ninja -C build
ninja -C build check-mlir
cd ..
```

Then build and test *CIRCT*:
```sh
cmake -G Ninja . -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm
ninja -C build
ninja -C build check-circt
ninja -C build check-circt-integration
```
