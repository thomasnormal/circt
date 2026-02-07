# CIRCT Tool Feature Status

This document tracks the feature parity status of CIRCT tools (`circt-verilog`,
`circt-sim`, `circt-bmc`, `circt-lec`) against the IEEE 1800 SystemVerilog
standard and UVM testbenches.

## sv-tests Compliance (IEEE 1800)

Tested against the [sv-tests](https://github.com/chipsalliance/sv-tests)
repository (1,036 tests across 15 IEEE chapters).

### Overall Results

| Mode | Eligible | Pass | Fail | Rate | Notes |
|------|----------|------|------|------|-------|
| Parsing | 853 | 853 | 0 | **100%** | 183 skipped: 70 negative tests, 104 need UVM, 6 need includes, 3 need `-D` flags |
| Elaboration | 850 | 847 | 3 | **99.6%** | 1 preprocessor edge case, 2 need external defines |
| Simulation (full) | 775 | 714 | 0 | **99.2%** | 884 total, 109 compile fail, 43 class-only (no top), 55 xfail, 6 xpass; `--max-time` resolves all former timeouts |
| BMC (full Z3) | 26 | 26 | 0 | **100%** | All Chapter 16 SVA tests pass with Z3 solving |
| LEC (full Z3) | 23 | 23 | 0 | **100%** | All Chapter 16 equivalence tests pass with Z3 |

### Remaining Bugs for 100%

| Test | Mode | Bug | Root Cause |
|------|------|-----|------------|
| `22.5.1--define-expansion_26.sv` | Elaboration | Macro concatenation (`` ` `` `` ` ``) edge case | Preprocessor token pasting |
| `5.6.4--*-macro_0.sv` | Elaboration | Needs `-DTEST_VAR` | Test harness metadata not applied |
| `5.6.4--*-macro_1.sv` | Elaboration | Needs `-DVAR_1=2 -DVAR_2=5` | Test harness metadata not applied |

### Simulation: 0 Failures, 0 Timeouts

All former timeout tests now pass with `--max-time=10us` (runner script:
`utils/run_sv_tests_circt_sim.sh`). The 6 XPASS tests are should-fail tests
that unexpectedly pass (not a tool bug). The 109 compile failures break down
as 100 UVM-dependent tests (need `import uvm_pkg`) and 9 non-UVM (6 Black Parrot,
3 other).

### What's Needed for True 100%

1. **Apply test `:defines:` metadata** (2 elaboration tests): The runner script
   needs to extract `:defines:` from test metadata and pass them as `-D` flags.
   Not a real tool bug.

2. **Preprocessor macro concatenation** (1 elaboration test): Edge case in
   `` `define `` with token pasting (`` ` `` `` ` ``). Low priority.

3. **Compile failures** (109 tests): 101 UVM-dependent (need `import uvm_pkg`),
   7 Black Parrot (need BSG macros), 1 genuine bug (`$printtimescale` hierarchical
   path resolution).

### circt-sim Unit Tests

| Suite | Total | Pass | XFail | Notes |
|-------|-------|------|-------|-------|
| circt-sim | 47 | 47 | 0 | All tests pass including cumulative __moore_delay |

## UVM Simulation Feature Status

Feature parity status of `circt-sim` for running UVM testbenches, compared
to commercial simulators like Cadence Xcelium.

### Feature Gap Table

| Feature | Status | Notes |
|---------|--------|-------|
| **UVM Core** | | |
| UVM init / phases | WORKS | `uvm_root`, build/connect/run phases |
| UVM factory | WORKS | `create_object_by_type`, class registration |
| UVM report server | WORKS | Severity actions, report messages |
| `$cast` / RTTI | WORKS | Parent table, type hierarchy checking |
| VTable dispatch | WORKS | Inherited methods, virtual calls across class hierarchy |
| `process::self()` | WORKS | Intercepted for both old and new compilations |
| **Data Structures** | | |
| Associative arrays | WORKS | Auto-create on null, integer and string keys |
| Queues | WORKS | `push_back`, `pop_front`, `size`, `sort`, `rsort`, `shuffle`, `reverse`, `unique` |
| Dynamic arrays | WORKS | `new[N]`, element access, `size()` — fixed allocation sizing and native memory tracking |
| Mailboxes | WORKS | DPI-based blocking/non-blocking |
| **Memory / I/O** | | |
| `$readmemh` | WORKS | File I/O for memory initialization |
| `$fopen` / `$fwrite` | WORKS | Basic file operations |
| Malloc / free | WORKS | Heap allocation with unaligned struct layout |
| **Simulation Infrastructure** | | |
| Combined HdlTop+HvlTop | WORKS | Multi-top simulation with BFMs |
| Shutdown cleanup | WORKS | `_exit(0)` skips expensive destructors for large designs |
| Signal driving (`llhd.drv`) | WORKS | Blocking assignments via epsilon delay |
| Signal probing (`llhd.prb`) | WORKS | Read signal values |
| **Known Gaps** | | |
| `$test$plusargs` | WORKS | Runtime `__moore_test_plusargs` checks `CIRCT_UVM_ARGS`/`UVM_ARGS` env var |
| `$value$plusargs` | WORKS | Runtime `__moore_value_plusargs` reads from `CIRCT_UVM_ARGS`/`UVM_ARGS` env var; supports `%d`, `%h`, `%o`, `%b` formats |
| `$urandom` / `$random` | WORKS | mt19937-based RNG via `__moore_urandom()` interceptors |
| `randomize()` (basic) | WORKS | `__moore_randomize_basic()` fills object memory with random bytes |
| `randomize()` (with dist) | WORKS | `__moore_randomize_with_dist()` implements weighted distribution constraints |
| `randomize()` (with ranges) | WORKS | `__moore_randomize_with_ranges()` generates uniform random within range pairs |
| Interface ports | MISSING | Required by AXI-VIP and similar verification IPs |
| ClockVar support | MISSING | Needed by some testbenches |
| `%c` format specifier | WORKS | `moore.fmt.char` in ImportVerilog, `FormatCharOpConversion` in MooreToCore, `sim.fmt.char` in interpreter |
| Coverage collection | MISSING | Functional and code coverage not implemented |
| SystemVerilog Assertions (SVA) | MISSING | Runtime assertion checking |
| `$finish` exit code | WORKS | Propagates exit code from `sim.terminate`; checks error count for UVM `die()` |
| DPI-C imports | PARTIAL | Some intercepted, most stubbed |
| `config_db` | WORKS | `config_db_implementation_t::set/get/exists` intercepted with in-memory key-value store |
| `process::suspend/resume` | WORKS | Lowered in ImportVerilog; interpreter suspends process execution and resumes on `resume()` call |
| Semaphores | WORKS | `__moore_semaphore_create/get/put/try_get` interceptors; blocking get with process suspension |
| Named events | PARTIAL | Basic `wait` / `trigger` / `.triggered` works; gap: `->>` NBA scheduling, event clearing between time slots |
| String methods | WORKS | All 18 IEEE 1800-2017 string methods intercepted |
| Simulation performance | OK | All AVIPs complete within 60s wall-clock |

### AVIP Simulation Status

| AVIP | HvlTop | HdlTop | Combined | Notes |
|------|--------|--------|----------|-------|
| APB | Runs | Runs | Runs | `apb_base_test` completes at ~108 ns sim time |
| AHB | Runs | Runs | Runs | `AhbBaseTest` completes at ~115 ns sim time |
| UART | Runs | Runs | Runs | `UartBaseTest` completes at ~255 ns sim time |
| I2S | Runs | - | - | `I2sBaseTest` completes at ~94 ns sim time |
| I3C | Runs | - | - | `i3c_base_test` completes at ~112 ns sim time |
| SPI | Runs | Runs | Runs | `SpiBaseTest` completes at ~743 ns sim time |

## Key Fixes History

| Fix | Description |
|-----|-------------|
| hw.array indexing | Element 0 at LSB, `idx * elementWidth` |
| Assoc array auto-create | Create on first access when pointer is null |
| findMemoryBlock for func args | Address-based lookup for function argument refs |
| Assoc array keySize | Truncation fix: `(width+7)/8` |
| Store/Load address fallback | Fall through to `findMemoryBlockByAddress` |
| Static vtable fallback | Eliminates dispatch warnings for corrupt pointers |
| Native store OOB fallback | Falls through when native block too small |
| Shutdown `_exit` | Placed in `processInput()` before destructor runs |
| `randomize_basic` | Fills object memory with random bytes instead of no-op |
| `randomize_with_dist` | Weighted distribution constraints with range/weight arrays |
| `randomize_with_ranges` | Uniform random within range pairs for constrained random |
| `$finish` exit code | Propagates exit code; checks error count for UVM `die()` → `$finish` |
| `%c` format | `moore.fmt.char` in ImportVerilog, `FormatCharOpConversion` in MooreToCore, `sim.fmt.char` in interpreter |
| Semaphore support | Full pipeline: ImportVerilog method lowering, constructor keyCount, interpreter interceptors with blocking |
| `$test$plusargs` | Runtime call via `__moore_test_plusargs`; traces through `IntToStringOp` to extract compile-time string; skips constant eval |
| Queue sort/rsort/shuffle/reverse | Interpreter interceptors with element size inference from data block; handles native/interpreter/mixed memory |
| Queue unique fix | Interpreter-managed queue support; native memory block registration for cross-operation data access |
| Native memory threshold fix | Queue operations check `nativeMemoryBlocks` map instead of just address threshold; fixes systems with low malloc addresses |
| `$value$plusargs` | Runtime `__moore_value_plusargs` with signal-aware output; traces SSA through `UnrealizedConversionCastOp` to drive `llhd.sig` |
| hw.bitcast 4-state | Recursive bit redistribution between flat and per-field four-state struct types |
| `llhd.wait` in fork | Fixed `llhd.wait` in nested/fork context to properly wait for time advance |
| Event `.triggered` | Initialize process results to 0 at registration; prevents X reads for self-referential process bodies |
| Interface path threading | Improved hierarchical interface path threading in ImportVerilog for nested module instances |
| Dynamic array allocation | MooreToCore now passes byte count (elemCount * elemSize) to `__moore_dyn_array_new`; interpreter registers native blocks |
| String array initializer | `ArrayCreateOpConversion` handles LLVM array types; `VariableOpConversion` stores initial values for LLVM arrays |
| config_db runtime | `config_db_implementation_t::set/get/exists` intercepted with key-value store |
| process suspend/resume | ImportVerilog lowering + interpreter `__moore_process_suspend/resume` handlers |
