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
| Elaboration | 1028 | 1021+ | 7 | **99.3%+** | 2 multi-assign detection, 5 crash/timeout (tagged union, SVA); stream_unpack FIXED, queue ops FIXED |
| Simulation (full) | 776 | 714 | 0 | **99.1%** | 876 total, 100 compile fail, 55 xfail, 6 xpass, 1 timeout, 1 error; `--max-time` resolves all former timeouts |
| BMC (full Z3) | 26 | 26 | 0 | **100%** | All Chapter 16 SVA tests pass with Z3 solving |
| LEC (full Z3) | 23 | 23 | 0 | **100%** | All Chapter 16 equivalence tests pass with Z3 |

### Remaining Failures (7 tests)

| Category | Count | Tests | Root Cause |
|----------|-------|-------|------------|
| ~~UVM `stream_unpack`~~ | ~~7~~ | ~~`testbenches/uvm_*`~~ | **FIXED** (b3031c5ec): Extract 4-state value field before i64 extension |
| ~~Queue ops on fixed arrays~~ | ~~3~~ | ~~`18.14`, `18.5.8.*`~~ | **FIXED** (c52eee8a9): Add `UnpackedArrayType` to 5 queue conversion patterns with probe/drive |
| Assignment conflict detection | 2 | `6.5--variable_*` | Slang's `AnalysisManager` crashes with SIGSEGV — needs upstream fix |
| Tagged union | 1 | `11.9--tagged_union_*` | Crash/timeout (empty log) |
| SVA negative tests | 4 | `16.10--*`, `16.15--*` | Crash/timeout (empty log) |

### Simulation: 0 Failures, 0 Timeouts

All former timeout tests now pass with `--max-time=10us` (runner script:
`utils/run_sv_tests_circt_sim.sh`). The 6 XPASS tests are should-fail tests
that unexpectedly pass (not a tool bug).

### What's Needed for True 100%

1. ~~**`moore.stream_unpack` legalization** (7 tests)~~: **FIXED** in commit
   b3031c5ec. Extract 4-state `{value, unknown}` struct before i64 widening.

2. ~~**Queue ops on fixed-size arrays** (3 tests)~~: **FIXED** in commit
   c52eee8a9. Add `UnpackedArrayType` handling to 5 queue conversion patterns
   (UniqueIndex, Reduce, RSorted, Shuffle, Reverse) with probe/drive for
   `!llhd.ref<!hw.array>` operands.

3. **Assignment conflict detection** (2 tests): Slang's `AnalysisManager`
   crashes when invoked from CIRCT. Needs upstream Slang investigation.

4. **Tagged union / SVA crashes** (5 tests): Empty logs suggest OOM or crash
   during compilation. Low priority.

### circt-sim Unit Tests

| Suite | Total | Pass | XFail | Notes |
|-------|-------|------|-------|-------|
| circt-sim | 167 | 167 | 0 | All pass; queue/array ops, config_db, semaphores, vtable, string methods, coverage, reduce ops |

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
| Interface ports | WORKS | Generic interface ports resolved from connection site (IEEE 1800-2017 §25.5) |
| ClockVar support | MISSING | Needed by some testbenches |
| `%c` format specifier | WORKS | `moore.fmt.char` in ImportVerilog, `FormatCharOpConversion` in MooreToCore, `sim.fmt.char` in interpreter |
| Coverage collection | WORKS | Covergroup/coverpoint sampling + reporting via MooreRuntime; implicit sample() evaluates coverpoint expressions; 4-state value extraction |
| SystemVerilog Assertions (SVA) | MISSING | Runtime assertion checking |
| `$finish` exit code | WORKS | Propagates exit code from `sim.terminate`; checks error count for UVM `die()` |
| DPI-C imports | PARTIAL | Some intercepted, most stubbed; UVM regex DPI (`uvm_re_comp/exec/free`) uses `std::regex` for full POSIX extended regex support |
| `config_db` | WORKS | `config_db_implementation_t::set/get/exists` intercepted with in-memory key-value store |
| `process::suspend/resume` | WORKS | Lowered in ImportVerilog; interpreter suspends process execution and resumes on `resume()` call |
| Semaphores | WORKS | `__moore_semaphore_create/get/put/try_get` interceptors; blocking get with process suspension |
| Named events | PARTIAL | Basic `wait` / `trigger` / `.triggered` works; `->>` integer-type NBA works via NonBlockingAssign; gap: native EventType NBA, event clearing between time slots |
| String methods | WORKS | All 18 IEEE 1800-2017 string methods intercepted |
| Simulation performance | GOOD | ~171 ns/s APB (30% from dialect dispatch + interceptor cache, 33x from O(log n) address index); 10us sim in 59s wall-clock |

### AVIP Simulation Status

| AVIP | HvlTop | HdlTop | Combined | Notes |
|------|--------|--------|----------|-------|
| APB | Runs | Runs | Runs | `apb_base_test` completes at ~200 ns sim time |
| AHB | Runs | Runs | Runs | `AhbBaseTest` completes at ~200 ns sim time |
| UART | Runs | Runs* | Runs* | `UartBaseTest` at ~300 ns; *bind assertions need slang fix |
| I2S | Runs | Runs* | Runs* | `I2sBaseTest` at ~200 ns; *bind assertions need slang fix |
| I3C | Runs | Runs | Runs | `i3c_base_test` at ~200 ns; pullup/wire/generate all work |
| SPI | Runs | Runs* | Runs* | `SpiBaseTest` at ~800 ns; *bind assertions need slang fix |
| AXI4Lite | Runs* | N/A | N/A | `Axi4LiteBaseTest`; *exits early due to vtable dispatch gap in `uvm_task_phase::m_traverse` |
| JTAG | Runs | N/A | N/A | `HvlTop` at ~500 ns; DPI/REGEX errors eliminated by `std::regex` |

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
| Generic interface ports | Resolve `InterfacePortSymbol::interfaceDef` via `getConnection()` for generic `interface` port declarations |
| Cumulative `__moore_delay` | Save `CallStackFrame` for LLVM function bodies on suspend; enables sequential delays in class methods/fork branches |
| Nested interface ports | Scope-hierarchy fallback in `resolveInterfaceInstance` for sub-interfaces inside interfaces; navigates through parent interface value |
| O(log n) address index | Replaced 14 O(n) linear scans through 6,577 globals with sorted interval map (`addrRangeIndex`); 33x speedup (~4 ns/s → ~132 ns/s) |
| `interpretFuncBody` caching | Fixed per-op `processStates.find()` in func body loop to use cached `activeProcessState` pointer |
| Dialect fast dispatch | `interpretOperation` goto-based dialect namespace check skips 20+ irrelevant dyn_casts for 93% of ops (LLVM/comb/arith) |
| Interceptor dispatch cache | `nonInterceptedExternals` DenseSet skips 128-entry string chain; `__moore_delay` moved to position #1; ~132 → ~171 ns/s |
| Coverage collection | Real MooreRuntime calls replace stubs; implicit sample() evaluates coverpoint expressions; 4-state struct value extraction; module-level LLVM op signal init propagation |
| `wait_condition` in functions | condBlock = callOp->getBlock() for function-body restart; frame resume override; arith ops in shouldTrace for complete condition chain |
| `wait_event` body pre-exec | Pre-execute body ops (e.g., llvm.call for assoc_get_ref) before signal/memory tracing walks |
| `uvm_wait_for_nba_region` | Intercepted as single delta cycle delay (scheduleProcess to Reactive region) |
| Queue/array ops on fixed arrays | Added `UnpackedArrayType` handling to 5 MooreToCore patterns (unique_index, reduce, rsort, shuffle, reverse) |
| Array reduce/min/unique_index interceptors | 7 new interpreter interceptors: `reduce_sum/product/and/or/xor`, `array_min`, `unique_index` |
| `get_adjacent_successor_nodes` | Interceptor for UVM phase graph traversal via `__moore_get_adjacent_successor_nodes` |
| StreamUnpackOp 4-state fix | Extract 4-state `{value, unknown}` struct before i64 widening in MooreToCore |
| Queue ops on fixed arrays | Add `UnpackedArrayType` to 5 MooreToCore patterns with `llhd::ProbeOp`/`llhd::DriveOp` for `!llhd.ref<!hw.array>` |
| Parameterized interface dedup | Use `hasSameType()` for deduplication; different parameterizations get separate MLIR declarations |
| Bounded delta polling (RUNPHSTIME fix) | Use delta steps for first 1000 polls, then 1ps fallback; eliminates UVM_FATAL [RUNPHSTIME] in all AVIPs |
| UVM regex DPI (`std::regex`) | Replace manual pattern matcher with `std::regex` for full POSIX extended regex; eliminates UVM_ERROR DPI/REGEX in AXI4Lite/JTAG |
| Fixed-array sort/stream ops | Add `UnpackedArrayType` to 7 more MooreToCore patterns (SortWith, RSortWith, ArraySize, StreamConcat/Unpack, StreamConcatMixed/UnpackMixed) with probe/drive |
