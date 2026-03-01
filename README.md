<p align="center"><img src="docs/includes/img/circt-logo.svg"/></p>

[![](https://github.com/llvm/circt/actions/workflows/buildAndTest.yml/badge.svg?event=push)](https://github.com/llvm/circt/actions?query=workflow%3A%22Build+and+Test%22)
[![Nightly integration tests](https://github.com/llvm/circt/workflows/Nightly%20integration%20tests/badge.svg)](https://github.com/llvm/circt/actions?query=workflow%3A%22Nightly+integration+tests%22)

# CIRCT — Circuit IR Compilers and Tools

This is a fork of [llvm/circt](https://github.com/llvm/circt) that adds a
**SystemVerilog simulator** (`circt-sim`) capable of running full **UVM
testbenches** end-to-end. It compiles SystemVerilog through
[slang](https://github.com/MikePajworski/slang) into MLIR, lowers it through
CIRCT's Moore/LLHD dialects, and interprets the result with 4-state semantics,
fork/join process control, VPI, and a built-in UVM runtime.

## What works today

| Area | Status |
|------|--------|
| **SystemVerilog parsing** | Full IEEE 1800-2017 via slang (268/268 ImportVerilog tests) |
| **Simulation** | 530+ lit tests, 907 sv-tests (855 pass + 50 expected-fail) |
| **UVM testbenches** | 9 protocol AVIPs run to completion (APB, AHB, AXI4, AXI4-Lite, SPI, I2S, I3C, JTAG, UART) |
| **VPI / cocotb** | 48/50 cocotb tests pass |
| **Coverage** | Covergroups, cross coverage, parametric sample |
| **Formal verification** | BMC + LEC via `circt-bmc` / `circt-lec` |
| **Mutation testing** | `circt-mut` cover/matrix/report with formal pre-qualification |

## Fork highlights

Compared to upstream CIRCT, this fork adds:

- **`circt-sim`** — an MLIR-based simulator with an event-driven process
  scheduler, 4-state value semantics, and a UVM runtime (config_db, factory,
  phase sequencing, objections, sequencer, coverage, and more).
- **Class OOP** — full class hierarchy, virtual methods, vtable dispatch,
  parameterized classes, static properties.
- **Virtual interfaces** — shadow signal propagation, modport support,
  interface-to-ref conversion.
- **Fork/join** — `fork`/`join`, `join_any`, `join_none`, `wait fork`,
  `disable fork`.
- **Constraint randomization** — range, multi-range, soft constraints, inline
  implication; 94% operator coverage.
- **File I/O & system tasks** — `$fopen`/`$fwrite`/`$fclose`,
  `$display`/`$sformatf`, `$finish`/`$fatal`, `$readmemh`.
- **Coverage** — covergroups, coverpoints, cross coverage, `sample()`.
- **VPI runtime** — enough of IEEE 1364 VPI for cocotb integration.
- **Mutation testing** — `circt-mut` for RTL mutation campaigns with formal
  pre-qualification (see [docs/FormalRegression.md](docs/FormalRegression.md)).
- **LSP / IDE** — UVM-aware `circt-lsp-server` with `--uvm-path` support.

For a full changelog, see [docs/CHANGES.md](docs/CHANGES.md).

## Getting started

### Prerequisites

You need git, ninja, python3, cmake, and a C++ compiler. See the
[LLVM requirements](https://llvm.org/docs/GettingStarted.html#requirements) for
details.

### Build

```sh
# Clone with submodules
git clone --recursive git@github.com:thomasnormal/circt.git
cd circt

# Build (add -DCIRCT_SLANG_FRONTEND_ENABLED=ON for circt-verilog)
cmake -G Ninja llvm/llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_TARGETS_TO_BUILD=host \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_EXTERNAL_PROJECTS=circt \
    -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD \
    -DLLVM_ENABLE_LLD=ON

# Build just circt-sim and circt-verilog
ninja -C build circt-sim circt-verilog

# Run tests
ninja -C build check-circt
```

To build a specific tool only (faster):
```sh
ninja -C build bin/circt-sim      # simulator
ninja -C build bin/circt-verilog  # SV frontend
ninja -C build bin/circt-opt      # MLIR optimizer
ninja -C build bin/firtool        # FIRRTL compiler
```

### Running a simulation

```sh
# Compile SystemVerilog to MLIR
circt-verilog input.sv -o design.mlir

# Simulate
circt-sim design.mlir --top top_module
```

For UVM testbenches, compile with `--uvm-path` pointing at a UVM library and
use `--top` for both HVL and HDL top modules:
```sh
circt-sim design.mlir --top hvl_top --top hdl_top --max-time=500000000
```

### Building LLVM separately

You can build LLVM/MLIR separately and point CIRCT at it, which lets you use
different build types (e.g., LLVM in release, CIRCT in debug):

```sh
# Build LLVM/MLIR
cd llvm
cmake -G Ninja llvm -B build \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DLLVM_ENABLE_PROJECTS=mlir \
    -DLLVM_TARGETS_TO_BUILD=host
ninja -C build
cd ..

# Build CIRCT against it
cmake -G Ninja . -B build \
    -DCMAKE_BUILD_TYPE=Debug \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DMLIR_DIR=$PWD/llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/llvm/build/lib/cmake/llvm
ninja -C build check-circt
```

### Submodules

CIRCT pins LLVM as a git submodule. After switching branches:
```sh
git submodule update
```

## Mutation testing

This fork includes `circt-mut` for Certitude/MCY-style RTL mutation campaigns
with formal pre-qualification via `circt-bmc` and `circt-lec`.

```sh
# Build mutation tools
ninja -C build circt-mut circt-bmc circt-lec circt-verilog

# Bootstrap a campaign
circt-mut init --project-dir my-campaign

# Run it
circt-mut run --project-dir my-campaign --mode all

# Report results
circt-mut report --project-dir my-campaign --mode all --out report.tsv
```

For the full CLI reference, lane TSV schema, global-filter modes, and CI
integration guide, see [docs/FormalRegression.md](docs/FormalRegression.md).

## Tool resource guard

Most CIRCT tools include a built-in resource guard to prevent runaway memory or
time usage:

- Disable: `--no-resource-guard`
- Adjust: `--max-rss-mb=...` / `--max-wall-ms=...` (or env vars
  `CIRCT_MAX_RSS_MB`, `CIRCT_MAX_WALL_MS`, etc.)
- Debug: `--resource-guard-verbose`

## Upstream CIRCT

This fork tracks [llvm/circt](https://github.com/llvm/circt). For upstream
documentation, community links, and the project charter:

- [Discourse forum](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/)
- [Discord channel](https://discord.com/channels/636084430946959380/742572728787402763)
- [Weekly meeting notes](https://docs.google.com/document/d/1fOSRdyZR2w75D87yU2Ma9h2-_lEPL4NxvhJGJd-s5pk/edit#)
- [Charter](docs/Charter.md)
- [Getting started (upstream)](docs/GettingStarted.md)
- [Python bindings](docs/PythonBindings.md)
