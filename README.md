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

Preferred frontend: `circt-mut` (subcommands: `init`, `run`, `report`,
`cover`, `matrix`, `generate`).
Legacy script entrypoints under `utils/` remain supported for compatibility.
`circt-mut generate` now has a native execution path for core generation
options, including native generated-mutation caching via `--cache-dir`.
Unsupported future options still fall back to the script backend during
migration.

Bootstrap a project template (MCY/Certitude-style campaign scaffold):

```sh
circt-mut init --project-dir mut-campaign
```

This writes:
- `mut-campaign/circt-mut.toml` (cover/matrix defaults)
- `mut-campaign/tests.tsv` (test manifest template)
- `mut-campaign/lanes.tsv` (matrix lane template)

Run from project config:

```sh
circt-mut run --project-dir mut-campaign --mode all
```

`circt-mut run` consumes `circt-mut.toml` and dispatches native
preflight-backed `cover` and/or `matrix` flows (`--mode cover|matrix|all`).
`run` now supports both cover mutation sources from config:
- `mutations_file = "..."` (prebuilt list), or
- `generate_mutations = <N>` (+ `mutations_modes`/`mutations_profiles`/
  `mutations_mode_counts`/`mutations_mode_weights`/`mutations_cfg`/`mutations_select`/
  `mutations_yosys`/`mutations_seed`).
These two modes are mutually exclusive and validated natively before dispatch.
`run` also now forwards strict boolean controls from config
(`1|0|true|false|yes|no|on|off`):
- cover/matrix gate toggles:
  `resume`, `skip_baseline`, `fail_on_undetected`, `fail_on_errors`,
  `stop_on_fail`.
- cover formal toggles:
  `formal_global_propagate_assume_known_inputs`,
  `formal_global_propagate_accept_xprop_only`,
  `formal_global_propagate_bmc_run_smtlib`,
  `formal_global_propagate_bmc_assume_known_inputs`.

Aggregate campaign results (cover/matrix/all) into one normalized TSV summary:

```sh
circt-mut report --project-dir mut-campaign --mode all --out reports/campaign.tsv
```

`circt-mut report` automatically reads `cover`/`matrix` output roots from
`circt-mut.toml` (or uses `<project-dir>/out/cover` and
`<project-dir>/out/matrix` defaults), then emits key-value summaries for:
- cover mutation buckets and coverage.
- matrix lane pass/fail/gate status and aggregated per-lane mutant metrics.
- formal-global-filter telemetry aggregation (timeouts, unknown outcomes,
  chain fallback counters, runtime/run counters, and BMC orig-cache counters).

Compare against a baseline campaign report:

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --compare reports/baseline.tsv \
  --out reports/current-with-diff.tsv
```

This appends diff rows (for overlapping numeric keys):
- `diff.<metric>.delta`
- `diff.<metric>.pct_change`
plus summary counters (`diff.overlap_keys`, `diff.numeric_overlap_keys`,
`diff.exact_changed_keys`, `diff.added_keys`, `diff.missing_keys`).

Compare against the latest snapshot in a rolling history file:

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --compare-history-latest reports/history.tsv
```

Append the current report snapshot to history:

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --append-history reports/history.tsv
```

Compute rolling trend summaries from history:

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --trend-history reports/history.tsv \
  --trend-window 10
```

Gate regressions directly in report mode:

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --compare reports/baseline.tsv \
  --fail-if-delta-gt cover.global_filter_timeout_mutants=0 \
  --fail-if-delta-lt cover.detected_mutants=0
```

`--fail-if-delta-gt` / `--fail-if-delta-lt` evaluate numeric
`diff.<metric>.delta` values and set:
- `compare.gate_rules_total`
- `compare.gate_failure_count`
- `compare.gate_status` (`pass`/`fail`)
- `compare.gate_failure_<n>` for each failing rule.
Gate rules require `--compare` or `--compare-history-latest` with numeric
baseline values for the gated keys.
On gate failure, `circt-mut report` exits with code `2`.

Gate against trend deltas (`current - mean(window)`):

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --trend-history reports/history.tsv \
  --trend-window 10 \
  --fail-if-trend-delta-gt cover.global_filter_timeout_mutants=0 \
  --fail-if-trend-delta-lt cover.detected_mutants=0
```

Trend gate rows:
- `trend.gate_rules_total`
- `trend.gate_failure_count`
- `trend.gate_status` (`pass`/`fail`)
- `trend.gate_failure_<n>`
Trend gates require `--trend-history`.

Use built-in policy bundles (recommended for CI):

```sh
circt-mut report \
  --project-dir mut-campaign \
  --mode all \
  --compare-history-latest reports/history.tsv \
  --trend-history reports/history.tsv \
  --trend-window 10 \
  --policy-profile formal-regression-basic \
  --policy-profile formal-regression-trend
```

Built-in profiles:
- `formal-regression-basic`
  - blocks regressions in compare deltas for:
    `cover.detected_mutants`, `cover.global_filter_timeout_mutants`,
    `cover.global_filter_lec_unknown_mutants`,
    `cover.global_filter_bmc_unknown_mutants`.
- `formal-regression-trend`
  - blocks regressions against rolling trend deltas for the same metrics.

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
For Z3 options (`--formal-global-propagate-z3`,
`--formal-global-propagate-bmc-z3`, matrix default variants), use `auto` to
resolve from `PATH`.
`circt-mut cover` now performs this built-in-tool resolution natively and
rewrites bare `--formal-global-propagate-circt-lec` /
`--formal-global-propagate-circt-bmc` to explicit executables before dispatch,
so unresolved toolchains fail fast with a direct CLI diagnostic.
It also runs native global-filter preflight checks:
- validates `--formal-global-propagate-circt-chain` mode values
- auto-injects LEC/BMC built-in tools for chain mode when omitted
- rejects conflicting non-chain global filter mode combinations early.
Cover mutation source consistency is now also validated natively:
- exactly one of `--mutations-file` or `--generate-mutations` must be set
- conflicting or missing source configuration fails fast.
For generated-mutation cover runs, native preflight now also validates:
- `--generate-mutations` as a positive integer
- `--mutations-seed` as a non-negative integer
- `--mutations-modes` names against supported mode/family set
- `--mutations-profiles` names against built-in profile set
- `--mutations-mode-counts` / `--mutations-mode-weights` entry syntax
- `--mutations-mode-counts` / `--mutations-mode-weights` mode names
- count/weight conflict (`mode-counts` vs `mode-weights`)
- `--mutations-mode-counts` total against `--generate-mutations`.
`circt-mut matrix` now applies the same preflight model for default global
filter options (`--default-formal-global-propagate-circt-*`) before dispatch.
It also pre-resolves `--default-mutations-yosys` so generated-mutation lanes
fail fast if the default Yosys executable is unavailable.
Matrix default generated-mutation seed is now configurable via
`--default-mutations-seed` for lanes that leave `mutations_seed` unset.
Generated lanes in `--lanes-tsv` now also get native preflight validation for
lane `mutations_yosys` values before script dispatch.
Generated lanes now also validate:
- `generate_count` as a positive integer
- `mutations_seed` as a non-negative integer (defaults to `1` when unset)
- effective `mutations_modes` names
- effective `mutations_mode_counts` / `mutations_mode_weights` syntax
  (lane override or matrix default)
- effective `mutations_mode_counts` / `mutations_mode_weights` mode names
- count/weight conflict
- `mutations_mode_counts` total against `generate_count`.
Lane mutation source consistency is now also validated natively:
- static lane: `mutations_file` path with no `generate_count`
- generated lane: `mutations_file=-` with positive `generate_count`
- invalid combinations (both source types set, or neither set) now fail fast.
Lane-level formal tool fields in `--lanes-tsv` now also get native preflight
validation (`global_propagate_circt_lec`, `global_propagate_circt_bmc`,
`global_propagate_z3`, `global_propagate_bmc_z3`) with effective default
fallback semantics.
Lane timeout/cache/gate override fields now also get native validation
(`global_propagate_*_timeout_seconds`, `global_propagate_bmc_bound`,
`global_propagate_bmc_ignore_asserts_until`, BMC orig-cache limits/policy, and
`skip_baseline`/`fail_on_undetected`/`fail_on_errors`).
Lane boolean formal fields now also get native validation with explicit
enable/disable forms (`1|0|true|false|yes|no|-`):
`global_propagate_assume_known_inputs`, `global_propagate_accept_xprop_only`,
`global_propagate_bmc_run_smtlib`, `global_propagate_bmc_assume_known_inputs`.
Matrix default numeric/cache options now also fail fast natively:
`--default-formal-global-propagate-*_timeout-seconds`,
`--default-formal-global-propagate-bmc-bound`,
`--default-formal-global-propagate-bmc-ignore-asserts-until`,
`--default-bmc-orig-cache-max-*`, and
`--default-bmc-orig-cache-eviction-policy`.
Matrix default mutation allocation options now also fail fast natively:
`--default-mutations-mode-counts` and `--default-mutations-mode-weights`
(syntax/value checks plus mutual-exclusion conflict checks).
Matrix default mutation mode names are now also validated natively:
`--default-mutations-modes`, `--default-mutations-mode-counts`,
`--default-mutations-mode-weights`.
Matrix generated-lane/default profile names are now also validated natively:
`--default-mutations-profiles` and lane `mutations_profiles`.
When effective matrix global-filter mode + timeout settings are non-zero
(defaults or lane overrides), `circt-mut matrix` now also fail-fast checks
that `timeout` is resolvable from the current `PATH`.
`circt-mut cover` now also fail-fast validates its corresponding formal
numeric/cache controls (`--formal-global-propagate-*_timeout-seconds`,
`--formal-global-propagate-bmc-bound`,
`--formal-global-propagate-bmc-ignore-asserts-until`,
`--bmc-orig-cache-max-*`, `--bmc-orig-cache-eviction-policy`).
For cover mode, non-zero timeout settings with an active global filter now
also require a resolvable `timeout` executable in the current `PATH`.
Both `circt-mut cover` and `circt-mut matrix` now also pre-resolve Z3 options
for built-in filters (`--formal-global-propagate-z3`,
`--formal-global-propagate-bmc-z3`, and default matrix variants). Use explicit
paths or `auto` (PATH lookup); unresolved toolchains fail fast.
`circt-mut cover` also pre-resolves `--mutations-yosys`, and native
`circt-mut generate` now fail-fast resolves `--yosys` before execution and
validates `--mode` / `--mode-count(s)` / `--mode-weight(s)` names against the
supported mode/family set.

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
  --mutations-mode-weights arith=3,control=1 \
  --mutations-seed 1
```

For fault-model-oriented presets, use profiles such as
`--mutations-profiles fault-basic` (or `fault-stuck`, `fault-connect`).

When `--generate-mutations` is not evenly divisible:
- across selected top-level mode groups, and
- across concrete operators inside family aliases
  (`arith/control/balanced/all/stuck/invert/connect`)
extra allocations are distributed with deterministic seed-rotated assignment
(`--mutations-seed`) instead of always favoring the first listed operator.
Native `circt-mut generate` and `utils/generate_mutations_yosys.sh` use the
same policy.
Both now also fail-fast validate mode names for `--mode` / `--modes` and mode
keys in `--mode-count(s)` / `--mode-weight(s)`.

Additional fault-model aliases:
- `stuck` -> `const0,const1`
- `invert` -> `inv`
- `connect` -> `cnot0,cnot1`

Run multiple lanes with `circt-mut matrix`:

```sh
circt-mut matrix \
  --lanes-tsv /path/to/lanes.tsv \
  --out-dir /tmp/mutation-matrix \
  --default-mutations-yosys yosys \
  --default-mutations-seed 1 \
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
`utils/run_mutation_matrix.sh` now also pre-validates default generated-mutation
mode/profile/allocation options, and marks malformed generated lane mutation
config as `CONFIG_ERROR` before launching per-lane cover runs.
Matrix `results.tsv` now includes `config_error_reason`, making lane
configuration failures directly actionable in CI summaries.

Mutation materialization is built in by default via
`utils/create_mutated_yosys.sh`. Override with `--create-mutated-script` only
if you need MCY/external compatibility behavior.

Command mapping by workflow:

0. Bootstrap campaign project

`circt-mut`:

```sh
circt-mut init --project-dir /path/to/mut-campaign
```

Equivalent `mcy` flow:

```sh
cd /path/to/mcy_project
mcy init
```

Equivalent Certitude-style flow (schematic):

```sh
certitude_init -out /path/to/mut-campaign
```

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
  --default-mutations-seed 1 \
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

4. Aggregate/report campaign outcomes

`circt-mut report`:

```sh
circt-mut report \
  --project-dir /path/to/mut-campaign \
  --mode all \
  --compare-history-latest /tmp/mutation-history.tsv \
  --fail-if-delta-gt cover.global_filter_timeout_mutants=0 \
  --fail-if-delta-lt cover.detected_mutants=0 \
  --trend-history /tmp/mutation-history.tsv \
  --trend-window 10 \
  --fail-if-trend-delta-gt cover.global_filter_timeout_mutants=0 \
  --fail-if-trend-delta-lt cover.detected_mutants=0 \
  --policy-profile formal-regression-basic \
  --policy-profile formal-regression-trend \
  --append-history /tmp/mutation-history.tsv \
  --out /tmp/mutation-report.tsv
```

Equivalent `mcy` flow (schematic aggregation from DB/status files):

```sh
cd /path/to/mcy_project
mcy status
# plus site-specific scripting to merge multi-project/lane summaries
```

Equivalent Certitude-style flow (schematic):

```sh
certitude_report \
  -in /tmp/certitude-run \
  -out /tmp/certitude-report
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
- report history file (`--compare-history-latest` / `--append-history`):
  `run_id<TAB>timestamp_utc<TAB>key<TAB>value`.

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
