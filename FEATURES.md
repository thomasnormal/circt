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
| Elaboration | 850 | 845 | 5 | **99.4%** | 2 real bugs (llhd.wait in function), 3 need external defines |
| Simulation | 256 | 255 | 1 | **99.6%** | 1 compile bug (llhd.wait in function); 91 skipped: 50 negative, 37 UVM, 4 both |
| BMC (full Z3) | 26 | 26 | 0 | **100%** | All Chapter 16 SVA tests pass with Z3 solving |
| LEC (full Z3) | 23 | 23 | 0 | **100%** | All Chapter 16 equivalence tests pass with Z3 |

### Remaining Bugs for 100%

| Test | Mode | Bug | Root Cause |
|------|------|-----|------------|
| `9.3.3--event.sv` | Elaboration | `llhd.wait` op expects parent `llhd.process` | MooreToCore emits `llhd.wait` inside `llvm.func` instead of `llhd.process` |
| `9.4.2.4--event_sequence.sv` | Elaboration + Sim | Same `llhd.wait` issue | Same root cause |
| `22.5.1--define-expansion_26.sv` | Elaboration | Macro concatenation (`` ` `` `` ` ``) edge case | Preprocessor token pasting |
| `5.6.4--*-macro_0.sv` | Elaboration | Needs `-DTEST_VAR` | Test harness metadata not applied |
| `5.6.4--*-macro_1.sv` | Elaboration | Needs `-DVAR_1=2 -DVAR_2=5` | Test harness metadata not applied |

### What's Needed for True 100%

1. **Fix `llhd.wait` in function context** (2 tests): MooreToCore needs to handle
   event control (`@(posedge clk)`) inside functions/tasks that are called from
   non-process contexts. This is a lowering bug, not a simulator bug.

2. **Apply test `:defines:` metadata** (2 tests): The sv-tests runner script
   needs to extract `:defines:` from test metadata and pass them as `-D` flags.
   Not a real tool bug.

3. **Preprocessor macro concatenation** (1 test): Edge case in `` `define ``
   with token pasting (`` ` `` `` ` ``). Low priority.

### circt-sim Unit Tests

| Suite | Total | Pass | XFail | Notes |
|-------|-------|------|-------|-------|
| circt-sim | 144 | 143 | 1 | XFail: `tlul-bfm-user-default.sv` (hw.bitcast for nested struct init) |

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
| Dynamic arrays | WORKS | Basic operations |
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
| `$value$plusargs` | PARTIAL | Still stubbed to 0; UVM uses DPI-based fallback for `+UVM_TESTNAME` |
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
| Semaphores | WORKS | `__moore_semaphore_create/get/put/try_get` interceptors; blocking get with process suspension |
| Named events | PARTIAL | Basic `wait` / `trigger` works |
| String methods | WORKS | All 18 IEEE 1800-2017 string methods intercepted |
| Simulation performance | SLOW | Large UVM designs (APB AVIP) take >300s wall-clock |

### AVIP Simulation Status

| AVIP | HvlTop | HdlTop | Combined | Notes |
|------|--------|--------|----------|-------|
| APB | Runs (slow) | Runs | Runs | >300s wall-clock, simulation logic completes |
| AHB | Runs (slow) | Runs | Runs | Similar to APB |
| UART | Runs | Runs | Runs | Reaches ~559.7 us sim time |
| I2S | Runs | - | - | `I2sBaseTest` starts, slow |
| I3C | Error | - | - | `ase_test` not registered (test config, not sim bug) |

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
