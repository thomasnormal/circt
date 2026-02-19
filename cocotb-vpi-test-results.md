# cocotb Test Suite Results vs circt-sim VPI

**Date:** 2026-02-18
**cocotb version:** 2.0.1 (installed) vs HEAD (test sources from ~/cocotb)
**circt-sim VPI:** `--vpi` flag with `libcocotbvpi_icarus.vpl`
**Compile flags:** `circt-verilog --ir-llhd --no-uvm-auto-include`

## Summary

| Metric | Count |
|--------|-------|
| Test groups run | 33 |
| Individual test functions | 252 |
| **PASS** | **156 (64.7%)** |
| FAIL | 85 |
| SKIP | 11 |
| Infra failures (not VPI) | 10 groups |

Of the 33 test groups that produced results, **17 pass cleanly** (52%), covering **156 individual test functions**.

The 10 infrastructure failures are Python import/version mismatches (cocotb 2.0.1 installed vs HEAD test sources) and missing PYTHONPATH entries — not VPI issues.

## Passing Test Groups (17 groups, 156 individual tests)

| Test Group | Tests | What's Validated |
|-----------|-------|-----------------|
| test_tests | **25** | Core cocotb test infrastructure, decorators, factories |
| test_timing_triggers | **21** | Timer, ReadOnly, ReadWrite, NextTimeStep triggers (21/25) |
| test_deprecated | **24** | Deprecated API compatibility (24/27) |
| test_async_bridge | **15** | Async coroutines, triggers, timers, events, queues |
| test_clock | **14** | Clock generation, period, duty cycle (14/15, 1 skip) |
| test_first_combine | **10** | First/Combine trigger patterns (10/25) |
| test_discovery | **10** | Signal discovery, access, read/write (10/33, 10 skip) |
| test_synchronization_primitives | **9** | Lock, Event, Barrier, Semaphore |
| test_logging | **5** | Log levels, formatting (5/6) |
| test_async_coroutines | **4** | Async/await patterns |
| test_async_generators | **4** | Async generator patterns |
| issue_348 | **3** | Signal read/write timing |
| issue_1279 | **2** | Coroutine cleanup |
| issue_957 | **2** | Clock edge detection |
| issue_142 | **1** | Timer resolution |
| test_fatal | **1** | `$fatal` handling via VPI |
| test_forked_exception | **1** | Exception propagation in forked coroutines |
| test_long_log_msg | **1** | Long message handling |
| test_one_empty_test | **1** | Empty test discovery |
| test_packed_union | **1** | Packed union signal access |
| test_start_soon | **1** | start_soon coroutine |
| test_sim_time_utils | **1** | Simulation time utility functions |

### Highlights
- **test_tests (25 pass):** All core test infrastructure works
- **test_timing_triggers (21/25 pass):** Timer and scheduling triggers mostly work
- **test_async_bridge (15 pass):** Full async coroutine stack works
- **test_clock (14 pass):** Clock generation fully functional
- **test_synchronization_primitives (9 pass):** All sync primitives work

## Failure Analysis

### Category 1: Missing Hierarchy / Sub-instances (20 failures)

**Root cause:** VPI `buildHierarchy()` only exposes top-level module signals as flat `vpiReg` objects. Does NOT expose:
- Sub-module instances (`vpi_iterate(vpiModule, parent)` for children)
- Generate block scopes (`genblk1`, `cond_scope`, `arr`)
- SV interface instances (`sv_if_i`, `intf_arr`)
- Gate instances (`test_and_gate`)
- Parameters/localparam as VPI objects
- String constants/variables

**Affected tests:**
- `test_discovery` (13/33 fail) — scope/generate/interface/gate discovery
- `test_sv_interface` (5/5 fail) — SV interface instances
- `test_compare` (2/2 fail) — sub-module handle comparison
- `test_defaultless_parameter` (1/1 fail) — sub-module parameter access
- `issue_2255` (1/1 fail) — generate scope discovery

### Category 2: Signal Width Doubling (28 failures)

**Root cause:** Four-state signals use `struct<value: iN, unknown: iN>` (physical width 2N). VPI `buildHierarchy` already halves width for `FourStateStruct`, but cocotb still sees doubled widths — likely `vpi_get(vpiSize, ...)` returns physical width.

**Affected tests:**
- `test_multi_dimension_array` (25/25 fail)
- `test_struct` (3/3 fail)

### Category 3: cocotb Version Mismatch (18 failures)

**Root cause:** Tests from cocotb HEAD use APIs not in installed 2.0.1 (`gather`, `TaskManager`, deprecated warning changes, `_prime()` signature).

**Affected tests:**
- `test_first_combine` (15/25 fail) — `DID NOT WARN` deprecation, `_prime()` args
- `test_deprecated` (3/27 fail) — API changes
- 5 groups completely failed to import (infra)

### Category 4: Array Value Assignment (6 failures)

**Root cause:** VPI doesn't support writing Python lists to array signals or iterating unpacked array elements.

**Affected tests:**
- `test_array_simple` (6/6 fail)

### Category 5: Timing/Scheduling (5 failures)

- `issue_120` (1 fail) — vpi_put_value write not visible next cycle
- `test_timing_triggers` (4 fail) — ReadOnly timing, NextTimeStep, awaitable types
- `test_3270` (2 fail) — simulation too slow (149M steps in 120s)

### Category 6: 4-State X Initialization (1 failure)

- `example_simple_dff` (1 fail) — `q` starts as X per Verilog spec, test expects 0

### Category 7: Package/Scope (2 failures)

- `test_package` (2 fail) — SV package not in VPI hierarchy

### Category 8: Other (5 failures)

- `test_logging` (1 fail) — log format difference
- `test_struct` included in width doubling above
- `test_discovery::access_integer` — integer type not recognized
- `test_discovery::access_internal_register_array` — packed array indexing

## Infrastructure Issues (not VPI failures)

| Issue | Count | Details |
|-------|-------|---------|
| cocotb version mismatch | 5 groups | `gather`, `TaskManager` not in 2.0.1 |
| Missing PYTHONPATH | 3 groups | `adder_model`, test helper modules |
| Missing source file | 1 group | `example_doc_counter` |
| Other import error | 1 group | `test_first_on_coincident_triggers` |

## VPI Fix Priority

### P0 — Hierarchy Discovery (would fix 20+ test functions)
Build proper module instance hierarchy from MLIR `hw.instance` ops. Support:
1. `vpi_iterate(vpiModule, parent)` for sub-instances
2. `vpi_iterate(vpiInternalScope, parent)` for generate scopes
3. `vpi_iterate(vpiParameter, parent)` for parameters
4. Dotted path in `vpi_handle_by_name("a.b.c", scope)`

### P1 — Width Reporting (would fix 28 test functions)
Ensure `vpi_get(vpiSize, handle)` consistently returns logical width N, not physical 2N for four-state signals.

### P2 — Array Support (would fix 6 test functions)
VPI iteration over unpacked array elements and indexed access.

### P3 — Signal Write Scheduling (would fix 2-5 test functions)
Ensure `vpi_put_value` writes are visible after `cbReadWriteSynch` and before next delta.

## How to Reproduce

```bash
# Compile a design
circt-verilog design.sv --ir-llhd --no-uvm-auto-include -o design.mlir

# Set cocotb environment
export COCOTB_TOPLEVEL=<top_module>
export COCOTB_TEST_MODULES=<test_module>
export TOPLEVEL_LANG=verilog
export LIBPYTHON_LOC=/usr/lib64/libpython3.9.so.1.0
export PYGPI_PYTHON_BIN=$(which python3)
export COCOTB_RESULTS_FILE=results.xml
export COCOTB_RANDOM_SEED=12345
export PYTHONPATH=<path_to_test_dir>

# Run with VPI
circt-sim design.mlir --top <top_module> \
  --vpi /path/to/cocotb/libs/libcocotbvpi_icarus.vpl \
  --max-time=100000000000
```
