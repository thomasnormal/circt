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
| Simulation (full) | 912 | 856 | 0 | **99.9%** | 912 total, 0 xfail, 7 xpass; 0 fail, 0 timeout; 1 compile-only (event sequence) |
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

### Simulation: 0 Failures, 7 Unexpected Passes, 0 Timeouts

912 tests found, 856 pass, 0 fail, 0 xfail, 7 xpass.
All tests properly categorized in `utils/sv-tests-sim-expect.txt`:
- 9 `skip` (8 should-fail tests circt-verilog doesn't detect + 1 utility file)
- 1 `compile-only` (event sequence control — needs `wait(condition)` support)
- 0 `xfail` (all UVM testbench tests now pass)
- 7 `xpass` (4 agent/monitor + 3 scoreboard — fixed by resolveSignalId + analysis port interceptor)
- 92 `pass` (Ch18 constraints/random stability/UVM phases + 26 SVA UVM tests now fully simulated)
- ~44 class-only Ch18 tests now fully simulated via auto-generated wrapper modules (no expect entry needed)

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
| circt-sim | 223 | 223 | 0 | All pass; sequencer interface, analysis ports, resolveSignalId cast+probe tracing, VIF shadow signals, resolveDrivers multi-bit fix, randomize(null/var_list), stream unpack, constraint solver, per-object RNG, parametric coverage, VIF clock propagation |
| MooreToCore | 124 | 122 | 2 | All pass; 2 XFAIL (array-locator-func-call, interface-timing-after-inlining) |
| ImportVerilog | 268 | 268 | 0 | All pass; short-circuit &&/\|\|/->, virtual-iface-bind-override, SVA moore.past, covergroup iff-no-parens |

## UVM Simulation Feature Status

Feature parity status of `circt-sim` for running UVM testbenches, compared
to commercial simulators like Cadence Xcelium.

### Feature Gap Table

| Feature | Status | Notes |
|---------|--------|-------|
| **UVM Core** | | |
| UVM init / phases | WORKS | `uvm_root`, build/connect/run phases; per-process phase map, master_phase_process detection, join_any objection polling |
| UVM factory | WORKS | `create_object_by_type`, class registration |
| UVM report server | WORKS | Severity actions, report messages |
| `$cast` / RTTI | WORKS | Parent table, type hierarchy checking |
| VTable dispatch | WORKS | Inherited methods, virtual calls across class hierarchy; runtime vtable override in all 3 call_indirect paths (X-fallback, direct, static) |
| `process::self()` | WORKS | Intercepted for both old and new compilations |
| Sequencer interface | WORKS | `start_item`/`finish_item`/`seq_item_pull_port::get` native interceptors; per-sequencer FIFO with call stack frame override for blocking get retry |
| Analysis ports | WORKS | `connect()`/`write()` interceptors; chain-following BFS dispatch via vtable slot 11 |
| **Data Structures** | | |
| Associative arrays | WORKS | Auto-create on null, integer and string keys; deep-copy on whole-assignment (`aa1 = aa2`) via `__moore_assoc_copy_into` |
| Queues | WORKS | `push_back`, `pop_front`, `size`, `sort`, `rsort`, `shuffle`, `reverse`, `unique` |
| Dynamic arrays | WORKS | `new[N]`, element access, `size()` — fixed allocation sizing and native memory tracking |
| Mailboxes | WORKS | DPI-based blocking/non-blocking |
| **Memory / I/O** | | |
| `$readmemh` | WORKS | File I/O for memory initialization |
| `$fopen` / `$fwrite` | WORKS | Basic file operations |
| Malloc / free | WORKS | Heap allocation with unaligned struct layout |
| **Simulation Infrastructure** | | |
| Combined HdlTop+HvlTop | WORKS | Multi-top simulation; APB AVIP dual-top boots full UVM topology with BFM registration via config_db (no UVM_FATAL); agents, drivers, monitors, sequencers, scoreboards all constructed |
| VIF task/function calls | MISSING | Virtual interface method dispatch to BFM interface tasks (e.g., `bfm_h.wait_for_preset_n()`) — current blocker for AVIP transaction flow |
| seq_item_port connection | MISSING | Driver proxy `seq_item_port` not connected to sequencer during `connect_phase`; prevents driver from obtaining sequences |
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
| `randomize(null)` check-only | WORKS | IEEE 18.11.1: check-only mode returns 1 if constraints satisfiable without modifying state |
| `randomize(var_list)` filtering | WORKS | IEEE 18.11: argument list controls which variables are randomized; unlisted rand variables treated as state |
| Interface ports | WORKS | Generic interface ports resolved from connection site (IEEE 1800-2017 §25.5) |
| Short-circuit evaluation | WORKS | `&&`, `\|\|`, `->` only evaluate RHS when needed (IEEE 1800-2017 §11.4.7); uses `moore.conditional` in procedural contexts |
| ClockVar support | MISSING | Needed by some testbenches |
| `%c` format specifier | WORKS | `moore.fmt.char` in ImportVerilog, `FormatCharOpConversion` in MooreToCore, `sim.fmt.char` in interpreter |
| Coverage collection | WORKS | Covergroup/coverpoint sampling + reporting via MooreRuntime; implicit AND parametric `with function sample()` evaluate per-coverpoint expressions; 4-state value extraction; bin hit tracking with min/max/unique |
| SystemVerilog Assertions (SVA) | MISSING | Runtime assertion checking |
| `$finish` exit code | WORKS | Propagates exit code from `sim.terminate`; checks error count for UVM `die()` |
| DPI-C imports | PARTIAL | Some intercepted, most stubbed; UVM regex DPI (`uvm_re_comp/exec/free`) uses `std::regex` for full POSIX extended regex support |
| `config_db` | WORKS | Dual-path interceptor: both `func.call` and `call_indirect` (VTable dispatch) for `config_db_default_implementation_t::set/get/exists`; in-memory key-value store; fuzzy `_x` suffix matching; `findMemoryBlockByAddress` for process-local write-back |
| `process::suspend/resume` | WORKS | Lowered in ImportVerilog; interpreter suspends process execution and resumes on `resume()` call |
| Semaphores | WORKS | `__moore_semaphore_create/get/put/try_get` interceptors; blocking get with process suspension |
| Wand/wor nets | WORKS | IEEE 1800-2017 §6.7 wired-AND/wired-OR resolution for multi-driver nets; `circt.resolution` attribute on signals |
| Named events | PARTIAL | Basic `wait` / `trigger` / `.triggered` works; `->>` integer-type NBA works via NonBlockingAssign; gap: native EventType NBA, event clearing between time slots |
| String methods | WORKS | All 18 IEEE 1800-2017 string methods intercepted |
| Simulation performance | GOOD | ~171 ns/s APB (30% from dialect dispatch + interceptor cache, 33x from O(log n) address index); 10us sim in 59s wall-clock |

### AVIP Simulation Status

All 9 AVIPs use a proxy-BFM split architecture where `hvl_top` (UVM test/env/agents)
communicates with `hdl_top` (clock/reset/BFM interfaces) via `uvm_config_db` virtual
interface handles. APB has been recompiled for dual-top simulation and **boots fully
without UVM_FATAL** — config_db handles flow correctly from hdl_top to hvl_top.
**No real bus transactions flow yet** because driver proxy's `seq_item_port` is not
connected to the sequencer and VIF task dispatch to BFM interfaces is not implemented.

Xcelium reference: APB `apb_8b_write_test` achieves 21-30% coverage with real SETUP/ACCESS
bus transactions at 130ns sim time, 0 UVM errors. Our output: 0% coverage, 42ps sim time,
no UVM_FATAL but no transactions.

| AVIP | HvlTop | HdlTop | Combined | Status | Notes |
|------|--------|--------|----------|--------|-------|
| APB | Full topology | ✅ Dual-top | ✅ Boots | seq_item_port gap | config_db works; all agents/monitors/scoreboards constructed; 0% coverage |
| AHB | Full phase lifecycle | Needs compile | Not tested | Needs dual-top MLIR | No MLIR file available |
| UART | Full phase lifecycle | Needs compile | Not tested | Needs dual-top MLIR | hvl_top-only exits 0 |
| I2S | UVM_FATAL (stale) | Needs compile | Not tested | Stale MLIR | Pre-compiled MLIR has parse errors |
| I3C | UVM_FATAL (stale) | Needs compile | Not tested | Stale MLIR | Pre-compiled MLIR has parse errors |
| SPI | Full phase lifecycle | Needs compile | Not tested | Needs dual-top MLIR | hvl_top-only exits 0 |
| AXI4 | Full phase lifecycle | Needs compile | Not tested | Needs dual-top MLIR | hvl_top-only exits 0 |
| AXI4Lite | Full phase lifecycle | Needs compile | Not tested | Needs dual-top MLIR | hvl_top-only exits 0 |
| JTAG | Full phase lifecycle | Needs compile | Not tested | Needs dual-top MLIR | hvl_top-only exits 0, 0% coverage |

**Road to full AVIP parity**:
1. ~~Compile both `hdl_top.sv` + `hvl_top.sv` together via `circt-verilog`~~ ✅ APB done
2. ~~Simulate with `circt-sim combined.mlir --top hvl_top --top hdl_top`~~ ✅ APB boots
3. ~~Validate BFM handles flow through config_db (hdl_top sets, hvl_top gets)~~ ✅ APB works
4. **Fix seq_item_port connection** — driver needs sequencer wiring
5. **Implement VIF task dispatch** — BFM interface method calls
6. **Recompile all 9 AVIPs** with dual-top support
7. Compare coverage numbers vs Xcelium reference outputs

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
| config_db runtime | Dual-path interceptor for `config_db_default_implementation_t::set/get/exists` — both `func.call` and `call_indirect` (VTable dispatch); fuzzy `_x` suffix matching; `findMemoryBlockByAddress` for alloca write-back |
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
| `get_adjacent_successor_nodes` | Interceptor for UVM phase graph traversal; layout-safe offsets via `getLLVMStructFieldOffset()` from MLIR struct type (was broken by hardcoded aligned offsets) |
| StreamUnpackOp 4-state fix | Extract 4-state `{value, unknown}` struct before i64 widening in MooreToCore |
| Queue ops on fixed arrays | Add `UnpackedArrayType` to 5 MooreToCore patterns with `llhd::ProbeOp`/`llhd::DriveOp` for `!llhd.ref<!hw.array>` |
| Parameterized interface dedup | Use `hasSameType()` for deduplication; different parameterizations get separate MLIR declarations |
| Bounded delta polling (RUNPHSTIME fix) | Use delta steps for first 1000 polls, then 1ps fallback; eliminates UVM_FATAL [RUNPHSTIME] in all AVIPs |
| UVM regex DPI (`std::regex`) | Replace manual pattern matcher with `std::regex` for full POSIX extended regex; eliminates UVM_ERROR DPI/REGEX in AXI4Lite/JTAG |
| Fixed-array sort/stream ops | Add `UnpackedArrayType` to 7 more MooreToCore patterns (SortWith, RSortWith, ArraySize, StreamConcat/Unpack, StreamConcatMixed/UnpackMixed) with probe/drive |
| Short-circuit evaluation | `&&`/`\|\|`/`->` use `moore.conditional` for lazy RHS evaluation in procedural contexts (IEEE 1800-2017 §11.4.7) |
| Virtual iface bind override | Wire `allowVirtualIfaceWithOverride` to slang `CompilationFlags`; enables virtual iface assignment with defparam/bind targets |
| Wand/wor net support | `NetOpConversion` handles WAnd/WOr/TriAnd/TriOr with `circt.resolution` attribute; ProcessScheduler AND/OR multi-driver resolution |
| MooreToCore compilation perf | Skip pre-patterns when no target ops; direct walk cleanup replaces greedy rewriter; disable region simplification |
| Vtable internal failure absorption | `call_indirect` absorbs internal failures from virtual methods; prevents cascading failures in UVM phase traversal |
| Slang covergroup iff-no-parens | Parser extension allows `iff valid` without parentheses (Xcelium/VCS compat) |
| Slang sequence decl semicolon | Parser extension allows missing semicolon before `endsequence` |
| MooreToCore early-exit | Skip entire pass when no Moore ops remain; avoids expensive full-module scans in second pipeline invocation |
| pre/post_randomize callbacks | User-defined `pre_randomize()`/`post_randomize()` methods called during `randomize()`; Public visibility prevents SymbolDCE removal |
| Class property initializers (no ctor) | `moore.class.new` emits property initializer assignments for classes without explicit constructors (IEEE 1800-2017 section 8.8) |
| Constraint implication/if-else/set-membership | Extract `->`, `if/else`, `inside` constraints from `ConstraintExprOp` in MooreToCore; lowers to conditional range application |
| rand_mode receiver resolution | Fix `rand_mode()` to use implicit class receiver; correct return value semantics (IEEE 1800-2017 §18.8) |
| Soft range constraints from ConstraintExprOp | Extract soft constraints from `ConstraintExprOp` with `isSoft` flag; apply as default ranges when no hard constraint overrides |
| Constraint guard null checks | Support `if (next == null) b1 == 5;` guards; pre-save pointer predicates before `randomize_basic`; `LLVM::ICmpOp` for pointer comparison (IEEE 1800-2017 §18.5.13) |
| Function lookup cache pre-population | Pre-populate interpreter function lookup cache during initialization; eliminates repeated O(n) searches during simulation |
| Constraint extraction zext/sext | `getPropertyName()` looks through `ZExtOp`/`SExtOp` for narrow-type properties; compound `and(uge, ule)` decomposed to min/max ranges |
| Constraint inheritance | Walk parent class hierarchy via `classDecl.getBaseAttr()` chain to collect inherited constraints |
| Distribution constraint traceToPropertyName | Replace BlockArgument-based property lookup with `traceToPropertyName()` through ReadOp/ClassPropertyRefOp/VariableOp |
| Per-object RNG state | `__moore_class_srandom(objPtr, seed)` in ImportVerilog; `std::mt19937` per object address in interpreter (IEEE 1800-2017 §18.13) |
| Virtual interface clock propagation | `ContinuousAssignOp` at module level creates `llhd.process` to watch source signals and continuously update interface memory |
| Parametric covergroup sample() | Fix 0% coverage: evaluate per-coverpoint expressions with `visitSymbolReferences()` to collect actual FormalArgument symbol pointers for name-based binding |
| Runtime vtable override (direct path) | All 3 `call_indirect` paths (X-fallback, direct, static) now check self object's runtime vtable pointer at byte offset 4; resolves derived class method from correct vtable slot |
| UVM phase sequencing | Per-process `executePhaseBlockingPhaseMap`, `master_phase_process` fork detection by name, join_any objection polling with `masterPhaseProcessChild` alive tracking; `wait_for_self_and_siblings_to_drop` native implementation |
| Debug output cleanup | Removed 29 debug `llvm::errs()` traces ([EXEC-PHASE], [OBJECTION-DBG], [CI-DISPATCH], [VTABLE-DBG], [SIM-FORK], [DYN-CAST], etc.) |
| Assoc array deep copy | `aa1 = aa2` emits `__moore_assoc_copy_into(dst, src)` instead of shallow pointer copy; fixes UVM phase graph corruption where `uvm_phase::add()` copies then deletes predecessor maps |
| Per-process RNG for random stability | Replaced global `std::rand/urandom` with per-process `std::mt19937` (IEEE 1800-2017 §18.13); `__moore_class_get/set_randstate` for save/restore |
| Coverpoint iff guard lowering | Added `iff_conditions` to `CovergroupSampleOp` with `AttrSizedOperandSegments`; conditional branch in MooreToCore to skip sample when iff evaluates false |
| Call stack restoration fix | Three bugs: innermost-first frame processing, `waitConditionSavedBlock` derivation from outermost frame's callOp, `outermostCallOp` fallback for non-wait_condition suspensions |
| `randomize(null)` check-only mode | IEEE 18.11.1: returns 1 if constraints are satisfiable without modifying object state; skip randomization and post_randomize |
| `randomize(var_list)` argument filtering | IEEE 18.11: only randomize specified variables; unlisted rand variables hold current state during constraint solving |
| Stream unpack interpreter handler | Handle `moore.stream_unpack` in interpreter for runtime streaming unpack operations |
| Foreach iterative constraints (18.5.8.1) | Constraint solver handles `foreach` over arrays with per-element constraints |
| Array reduction constraints (18.5.8.2) | Constraint solver supports `sum`, `product`, `and`, `or`, `xor` array reduction in constraints |
| Inline constraint range extraction (18.7) | Extract range constraints from `randomize() with { ... }` inline constraint blocks |
| Static rand_mode across instances (18.8) | `rand_mode(0/1)` correctly persists across all instances of a class |
| Inline variable control (18.11) | Both inline constraint checker and variable control tests now passing |
| UVM resource_db read_by_name | `uvm_resource_db::read_by_name` intercepted for UVM resource lookup |
| resolveDrivers() multi-bit fix | Fixed signal resolution for FourStateStruct `hw.struct<value, unknown>`: group drivers by full APInt value instead of just LSB; fixes clock signals never toggling in multi-driver scenarios |
| VIF shadow signals | Interface field shadow signals: `createInterfaceFieldShadowSignals()` creates per-field signals, store interception drives shadows when memory written, sensitivity expansion adds field signals to process wait lists |
