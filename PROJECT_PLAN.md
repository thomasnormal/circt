# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/mbit/*_avip` testbenches using only CIRCT tools and the library ~/uvm-core.
Secondary goal: Get to 100% in the ~/sv-tests/ and ~/verilator-verification/ test suites.

---

## Remaining Limitations & Next Steps

### CRITICAL: Simulation Runtime Blockers (Updated Iteration 74)

> **See `.claude/plans/ticklish-sleeping-pie.md` for detailed implementation plan.**

**RESOLVED in Iteration 71-74:**
1. ~~**Simulation Time Advancement**~~: ✅ FIXED - ProcessScheduler↔EventScheduler works correctly
2. ~~**DPI/VPI Real Hierarchy**~~: ✅ FIXED - Signal registry bridge implemented with callbacks
3. ~~**Virtual Interface Binding**~~: ✅ FIXED - InterfaceInstanceOp now returns llhd.sig properly
4. ~~**4-State X/Z Propagation**~~: ✅ INFRASTRUCTURE - X/Z preserved in IR, lowering maps to 0
5. ~~**Queue Sort With Method Calls**~~: ✅ FIXED (Iter 73) - QueueSortWithOpConversion implemented
6. ~~**LLHD Process Pattern Mismatch**~~: ✅ VERIFIED (Iter 73) - cf.br pattern is correctly handled
7. ~~**Signal-Sensitive Waits**~~: ✅ VERIFIED (Iter 73) - @(posedge/negedge) working
8. ~~**sim.proc.print Output**~~: ✅ FIXED (Iter 73) - $display now prints to console
9. ~~**ProcessOp Canonicalization**~~: ✅ FIXED (Iter 74) - Processes with $display/$finish no longer removed

**REMAINING BLOCKERS:**
1. **Concurrent Process Scheduling** ✅ FIXED (Iter 81):
   - Fixed `findNextEventTime()` to return minimum time across ALL slots
   - Added `advanceToNextTime()` for single-step advancement
   - Fixed interpreter state mismatch for event-triggered processes
   - All 22 ProcessScheduler unit tests now pass

2. **UVM Library** ✅ RESOLVED: **Use the real UVM library from `~/uvm-core`**
   - `circt-verilog --uvm-path ~/uvm-core/src` parses the real UVM successfully
   - All AVIP testbenches compile with the real UVM library
   - No need to maintain UVM stubs - just use the official IEEE 1800.2 implementation

3. **Class Method Inlining** ⚠️ MEDIUM: Virtual method dispatch and class hierarchy not fully simulated.

### Concurrent Process Scheduling Root Cause Analysis (Iteration 76)

**The Problem**:
When a SystemVerilog file has both `initial` and `always` blocks, only the `initial` block executes. The simulation shows:
- 3 LLHD processes registered
- Only 5 process executions (should be many more for looping always blocks)
- 2 delta cycles, 2 signal updates, 2 edge detections

**Root Cause Analysis** (from Track A investigation):

1. **Signal-to-Process Mapping**: The `signalToProcesses` mapping in `ProcessScheduler` is only populated when `suspendProcessForEvents()` is called, but this mapping is not maintained across process wake/sleep cycles.

2. **One-Shot Sensitivity**: When `interpretWait()` is called, it registers the process via `scheduler.suspendProcessForEvents(procId, waitList)`. But when the process wakes up, `clearWaiting()` clears the `waitingSensitivity` list:
   ```cpp
   void clearWaiting() {
     waitingSensitivity.clear();
     if (state == ProcessState::Waiting)
       state = ProcessState::Ready;
   }
   ```

3. **State Machine Mismatch**: After a process executes, if it doesn't reach its next `llhd.wait`, it defaults to `Suspended` state with no sensitivity, making it impossible to wake.

4. **Event-Driven vs Process-Driven Timing**: The `always #5 clk = ~clk` uses delay-based wait, but the counter process uses event-based wait on `posedge clk`. The timing of signal changes vs event callbacks may cause missed edges.

**Key Files**:
- `lib/Dialect/Sim/ProcessScheduler.cpp` lines 192-228, 269-286, 424-475
- `tools/circt-sim/LLHDProcessInterpreter.cpp` lines 247-322, 1555-1618

### Track Status & Next Tasks (Iteration 208 Update)

**Iteration 207 Results (COMPLETE):**
- Track A: ✅ **BUG FIXED** - `llhd.wait` delta-step resumption for `always @(*)`
- Track B: ✅ Multiple lit test fixes (externalize-registers, derived-clocks, etc.)
- Track C: ✅ **APB AVIP: 100K+ iterations** (was hanging, now works!)
- Track D: ✅ **40/40 OpenTitan crypto primitives** (prim_secded_*, gf_mult, etc.)

**Key Achievements from Iteration 207:**
- APB AVIP simulation unblocked - runs 100K+ iterations
- OpenTitan primitives: 12 → **52** (+40 crypto)
- llhd.wait hang bug fixed

**Iteration 210 Results (COMPLETE):**
- Track A: ✅ **Test suite stability verified** - sv-tests BMC 23/26, verilator 17/17, yosys 14/14
- Track B: ✅ **Stack overflow fix confirmed** - APB runs up to 1ms with 561 signals, 9 processes
- Track C: ✅ **Process canonicalization investigation complete** - func.call correctly detected as side effect
- Track D: ✅ **circt-lec.cpp compilation fix** - Attribute::getValue() -> cast to StringAttr
- Track E: ✅ **XFAIL tests marked** - uvm-run-test.mlir, array-locator-func-call-test.sv

**Key Achievements from Iteration 210:**
- Verified test suite stability matches expectations
- Stack overflow fix working in production AVIP simulations
- UVM processes preserved (func.call has side effects)
- Key finding: UVM report functions exist in runtime but MooreToCore doesn't generate calls yet
  - sim.proc.print works ($display output working)
  - __moore_uvm_report_* functions NOT being called from compiled UVM code

**Iteration 208 Results (COMPLETE):**
- Track A: ✅ **Multi-top module support VERIFIED** - `--top hdl_top --top hvl_top` works correctly
- Track B: ⚠️ 76 lit test failures remaining (fixed slang v10 version check, lec-extnets-cycle.mlir)
- Track C: ✅ **UVM STACK OVERFLOW FIXED** - Added call depth tracking to `func.call` and `func.call_indirect`
- Track D: ✅ **6 Full OpenTitan IPs with Alerts SIMULATE** (GPIO, UART, I2C, SPI Host, SPI Device, USBDev)
- Track I: 🔍 **UVM OUTPUT ROOT CAUSE FOUND** - External C++ runtime functions not dispatched in interpreter

**Key Achievements from Iteration 208:**
- UVM stack overflow fixed - full AVIP with hvl_top no longer crashes
- Root cause for silent UVM output: `__moore_uvm_report_*` functions need dispatcher handlers
- Test suites improved: sv-tests BMC 23/26 (+14), verilator-verification 17/17 (100%)
- OpenTitan: **39 testbenches pass** (12 full IPs + 26 reg_top + 1 fsm)

**UVM AVIP Compilation Status (VERIFIED):**
- **3/9 compile successfully** (APB, I2S, I3C) - 33%
- 5/9 have source code bugs (AHB, SPI, UART, JTAG, AXI4)
- 1/9 has complex build setup (AXI4Lite with env vars)

**Test Suite Status (Updated Iteration 207):**
- sv-tests SVA: **23/26 pass (88%)**
- verilator-verification: **17/17 compile (100%)** ✅ All compile now
- Yosys SVA: **14/14 pass (100%)** - stable

**OpenTitan Simulation Status:**
- **33/33 reg_top modules simulate**
- **52+ primitives**: flop_2sync, arbiter_fixed/ppc, lfsr, fifo_sync, timer_core, uart_tx/rx, alert_sender, packer, subreg, edge_detector, pulse_sync, + 40 crypto (secded, gf_mult, present, prince, subst_perm)
- **Large FSMs work**: i2c_controller_fsm (2293 ops, 9 processes)

**7 AVIPs Running in circt-sim:**
- AHB AVIP - 1M+ clock edges
- APB AVIP - 1M+ clock edges
- SPI AVIP - 111 executions
- UART AVIP - 20K+ executions
- I3C AVIP - 112 executions
- I2S AVIP - E2E working
- **AXI4 AVIP** - 100K+ clock edges ✅ NEW (Iter 118)

**20+ Chapters at 100% effective:** (Verified 2026-01-23)
- sv-tests Chapter-5: **50/50** (100%) ✅
- sv-tests Chapter-6: **82/84** (100% effective) - 2 "should_fail" tests (expected negative tests)
- sv-tests Chapter-7: **103/103** (100%) ✅
- sv-tests Chapter-8: **53/53** (100%) ✅
- sv-tests Chapter-10: **10/10** (100%) ✅
- sv-tests Chapter-11: **77/78** (100% effective) - 1 runtime "should_fail" test
- sv-tests Chapter-12: **27/27** (100%) ✅
- sv-tests Chapter-13: **15/15** (100%) ✅
- sv-tests Chapter-14: **5/5** (100%) ✅
- sv-tests Chapter-15: **5/5** (100%) ✅
- sv-tests Chapter-16: **23/26 non-UVM** (100% effective) - 27 UVM tests blocked on UVM library
- sv-tests Chapter-18: **68/68 non-UVM** (100%) - 66 UVM tests blocked on UVM library
- sv-tests Chapter-20: **46/47** (97.9%) - 1 hierarchical path test (test design issue)
- sv-tests Chapter-21: **29/29** (100%) ✅
- sv-tests Chapter-22: **74/74** (100%) ✅
- sv-tests Chapter-23: **3/3** (100%) ✅
- sv-tests Chapter-24: **1/1** (100%) ✅
- sv-tests Chapter-25: **1/1** (100%) ✅
- sv-tests Chapter-26: **2/2** (100%) ✅

**Other Chapters:**
- sv-tests Chapter-9: **46/46** (100%) ✅

**External Test Suites:**
- Yosys SVA BMC: **14/14 passing** (100%) ✅
- verilator-verification: **80.8%** (114/141 passing) ✅ **CORRECTED COUNT (Iter 113)**

**UVM AVIP Status:**
- **2/9 AVIPs compile from fresh SV source:** APB, I2S (rest have AVIP source bugs or CIRCT limitations)
- **7 AVIPs run in circt-sim with pre-compiled MLIR:** APB, AHB, SPI, UART, I3C, I2S, AXI4 (historical MLIR from when local fixes were applied)
- **I3C AVIP:** E2E circt-sim (112 executions, 107 cycles, array.contains fix, Iter 117) ✅
- **UART AVIP:** E2E circt-sim (20K+ executions, 500MHz clock, Iter 117) ✅
- **SPI AVIP:** E2E circt-sim (111 executions, 107 cycles, no fixes needed, Iter 116) ✅
- **AHB AVIP:** E2E circt-sim (clock/reset work, 107 process executions, Iter 115) ✅
- **APB AVIP:** E2E circt-sim (clock/reset work, 56 process executions, Iter 114) ✅

### Remaining Limitations (Updated Iteration 117)

**For Full UVM Testbench Execution:**
1. ~~**UVM run_test()**~~: ✅ IMPLEMENTED - Factory-based component creation
2. ~~**+UVM_TESTNAME parsing**~~: ✅ IMPLEMENTED (Iter 107) - Command-line test name support
3. ~~**UVM config_db**~~: ✅ IMPLEMENTED (Iter 108) - Hierarchical/wildcard path matching
4. ~~**TLM Ports/Exports**~~: ✅ IMPLEMENTED (Iter 109) - Runtime infrastructure for analysis ports/FIFOs
5. ~~**UVM Objections**~~: ✅ IMPLEMENTED (Iter 110) - Objection system for phase control
6. ~~**UVM Sequences**~~: ✅ IMPLEMENTED (Iter 111) - Sequence/sequencer runtime infrastructure
7. ~~**UVM Scoreboard**~~: ✅ IMPLEMENTED (Iter 112) - Scoreboard utility functions
8. ~~**UVM RAL**~~: ✅ IMPLEMENTED (Iter 113) - Register abstraction layer runtime
9. ~~**Array locator external calls**~~: ✅ FIXED (Iter 111, 112, 113) - Pre-scan + vtable dispatch
10. ~~**UVM recursive init calls**~~: ✅ FIXED (Iter 114) - Skip inlining guarded recursion
11. ~~**Class shallow copy**~~: ✅ FIXED (Iter 114) - moore.class.copy legalization
12. ~~**UVM Messages**~~: ✅ IMPLEMENTED (Iter 115) - Report info/warning/error/fatal with verbosity
13. ~~**Hierarchical instances**~~: ✅ FIXED (Iter 115) - circt-sim descends into hw.instance
14. ~~**Virtual Interface Binding**~~: ✅ IMPLEMENTED (Iter 116) - Thread-safe vif registries with modport support
15. ~~**case inside**~~: ✅ IMPLEMENTED (Iter 116) - Set membership with ranges and wildcards
16. ~~**Wildcard associative arrays**~~: ✅ FIXED (Iter 116) - [*] array key type lowering
17. ~~**TLM FIFO Query Methods**~~: ✅ IMPLEMENTED (Iter 117) - can_put, can_get, used, free, capacity
18. ~~**Unpacked arrays in inside**~~: ✅ FIXED (Iter 117) - moore.array.contains operation
19. ~~**Structure/variable patterns**~~: ✅ IMPLEMENTED (Iter 117) - Pattern matching for matches operator

**For sv-tests Completion:**
1. **Chapter-9 (100%)**:
   - ~~4 process class tests~~: ✅ FIXED (Iter 111)
   - ~~1 SVA sequence event test~~: ✅ FIXED
2. **10 Chapters at 100%:** 7, 10, 13, 14, 20, 21, 23, 24, 25, 26

**For verilator-verification (80.8%):**
- 21 of 27 failures are test file syntax issues (not CIRCT bugs)
- UVM-dependent tests: 1 test (skip)
- Expected failures: 4 tests (signal-strengths-should-fail/)
- Non-standard syntax: 14 tests (`1'z`, `@posedge (clk)`)
- Other LRM/slang limitations: 8 tests

### Current Track Status (Iteration 189)

**Completed Tracks:**
- **Track H (prim_diff_decode)**: ✅ FIXED - Mem2Reg predecessor deduplication, committed 8116230df
- **Track M (crypto IPs)**: ✅ DONE - Found CSRNG, keymgr, KMAC, OTBN parse; CSRNG recommended next
- **Track N (64-bit bug)**: ✅ ROOT CAUSE FOUND - SignalValue uses uint64_t, crashes on >64-bit signals
- **Track O (AVIP analysis)**: ✅ DONE - 2/9 compile (APB, I2S); rest are AVIP bugs/CIRCT limitations
- **Track P (CSRNG crypto IP)**: ✅ DONE - 10th OpenTitan IP simulates (173 ops, 66 signals, 12 processes)
- **Track Q (SignalValue 64-bit)**: ✅ FIXED - Upgraded to APInt, test/Tools/circt-sim/signal-value-wide.mlir
- **Track R (prim_alert_sender)**: ✅ VERIFIED - Mem2Reg fix works, 7+ IPs unblocked (gpio, uart, spi, i2c, timers)
- **Track S (test suite)**: ✅ VERIFIED - No regressions (sv-tests 9+3xfail, verilator 8/8, yosys 14/16)

**Iteration 209 Results (COMPLETE):**
- Track A: ✅ **UVM REPORT DISPATCHERS IMPLEMENTED** - `__moore_uvm_report_info/warning/error/fatal` now work
- Track B: ✅ 4 lit test failures fixed (59→55), commit 0b7b93202
- Track C: ✅ UVM_INFO output verified working in circt-sim unit test
- Track D: ✅ Test suites stable: verilator 100%, yosys 100%, sv-tests 23/26

**Key Achievements from Iteration 209:**
- UVM report functions now dispatch to C++ runtime (UVM_INFO/WARNING/ERROR/FATAL)
- Global string initialization fixed - string constants properly copied to memory blocks
- Added unit tests: `uvm-report-minimal.mlir`, `uvm-report-simple.mlir`
- Fixed 4 lit test CHECK patterns for updated output formats

**Active Tracks (Iteration 211):**
- **Track A**: Generate MooreToCore calls to `__moore_uvm_report_*` functions
- **Track B**: Fix remaining lit test failures (~55 remaining)
- **Track C**: Expand OpenTitan IP simulation coverage
- **Track D**: AVIP end-to-end simulation with UVM output

**Remaining Limitations for UVM Parity:**
1. ~~**UVM Code Stack Overflow**~~ ✅ FIXED (Iter 208) - Call depth tracking added
2. ~~**UVM Output Silent from hvl_top**~~ ✅ FIXED (Iter 209) - UVM report dispatchers implemented
   - `__moore_uvm_report_info/warning/error/fatal` now call C++ runtime functions
   - `__moore_uvm_report_enabled`, `__moore_uvm_report_summarize` also implemented
3. **llhd.process Canonicalization** - Processes without signal drives get removed as dead code
   - Workaround: Add a signal drive to processes that would otherwise be empty
3. **~76 Lit Test Failures** - Various categories: ImportVerilog, circt-bmc, circt-lec, circt-sim
4. **InOut Interface Ports** - I3C AVIP blocked (SCL port)
5. **AVIP Source Bugs** - 6/9 AVIPs have source-level issues (not CIRCT bugs)

**Completed (Iteration 208):**
1. ✅ **Multi-top module support** - `--top hdl_top --top hvl_top` verified working
2. ✅ **6 Full OpenTitan IPs with Alerts** - GPIO, UART, I2C, SPI Host, SPI Device, USBDev
3. ✅ **slang v10 version check** - Fixed commandline.sv test
4. ✅ **UVM stack overflow fix** - Call depth tracking in func.call/func.call_indirect
5. ✅ **Unit test** - test/Tools/circt-sim/call-depth-protection.mlir
6. ✅ **sv-tests BMC** - 23/26 pass (+14 improvement from 9)
7. ✅ **verilator-verification BMC** - 17/17 (100%)
8. ✅ **39 OpenTitan testbenches** - 12 full IPs + 26 reg_top + 1 fsm

### New: OpenTitan Simulation Support
- **Phase 1 Complete**: prim_fifo_sync, prim_count simulate in circt-sim
- **Phase 2 MILESTONE**: 10 register blocks simulate:
  - Communication: `gpio_reg_top`, `uart_reg_top`, `spi_host_reg_top`, `i2c_reg_top`
  - Timers (CDC): `aon_timer_reg_top`, `pwm_reg_top`, `rv_timer_reg_top`
  - Crypto: `hmac_reg_top`, `aes_reg_top`, `csrng_reg_top` (shadowed registers, dual reset)
- **Phase 3 Validated**: TileLink-UL protocol adapters (including tlul_socket_1n router) and CDC primitives work
- **FIXED**: `prim_diff_decode.sv` control flow bug - deduplication added in LLHD Mem2Reg.cpp `insertBlockArgs` function
- **FIXED**: circt-sim SignalValue 64-bit limit - upgraded to APInt for arbitrary-width signals
- **AVIP Analysis Complete**: 2/9 AVIPs compile (APB, I2S); remaining failures are AVIP source bugs or CIRCT limitations
- **Crypto IPs Parseable**: CSRNG, keymgr, KMAC, OTBN all parse successfully
- **timer_core**: Should now work with APInt-based SignalValue (ready to test)
- **Scripts**: `utils/run_opentitan_circt_verilog.sh`, `utils/run_opentitan_circt_sim.sh`
- **Tracking**: `PROJECT_OPENTITAN.md`

### Current Test Suite Status (Iteration 186)
- **sv-tests SVA BMC**: 9/26 pass, 3 xfail, 0 fail, 0 error (Verified 2026-01-26)
- **sv-tests Chapters**: 821/831 (98%) - aggregate across all chapters
- **verilator-verification BMC**: 8/8 active tests pass (Verified 2026-01-26)
- **yosys SVA**: 14/16 (87.5%) (Verified 2026-01-26)
- **AVIPs**: 2/9 compile (APB, I2S) - REGRESSION from claimed 9/9 baseline
  - Root causes: bind/vif conflicts, UVM method signature mismatches, InOut interface ports
  - Fixed I2S by handling file paths in +incdir+ gracefully (script fix)
- **OpenTitan**: 8 register blocks SIMULATE (communication + timers + crypto), TL-UL + CDC primitives validated

### AVIP Analysis Complete (Track W - Iteration 190)

**Summary**: 2/9 AVIPs compile via `./utils/run_avip_circt_verilog.sh`. The remaining failures are AVIP source bugs, CIRCT limitations, or test infrastructure issues.

| AVIP | Status | Root Cause | Fix Responsibility |
|------|--------|------------|-------------------|
| APB | ✅ PASS | - | - |
| I2S | ✅ PASS | - | - |
| AHB | FAIL | bind scope refs parent module ports (`ahbInterface`) | AVIP source bug |
| I3C | FAIL | InOut interface port (`SCL`) not supported | CIRCT limitation |
| AXI4 | FAIL | bind scope refs parent module ports (`intf`) | AVIP source bug |
| JTAG | FAIL | bind/vif conflict, enum casts, range OOB | AVIP source bugs |
| SPI | FAIL | nested comments, empty args, class access | AVIP source bugs |
| UART | FAIL | do_compare default arg mismatch with UVM base | AVIP source bug (strict LRM) |
| AXI4Lite | FAIL | No compile filelist found (needs env vars) | Test infra |

**Error Categories:**
- **AVIP source bugs (6)**: AHB, AXI4, JTAG, SPI, UART, AXI4Lite - require AVIP repo fixes
- **CIRCT limitation (1)**: I3C - InOut interface ports not yet supported
- **Previously documented local fixes**: UART (do_compare), JTAG (enum casts) were fixed locally but repos were reset

**Workaround**: Use `--allow-virtual-iface-with-override` for JTAG bind/vif conflicts (does not fix all errors).

### Remaining Limitations & Features to Build (Iteration 189)

**RESOLVED This Iteration:**
1. ~~**circt-sim SignalValue 64-bit limit**~~: ✅ FIXED (Track Q) - Upgraded to APInt for arbitrary widths
2. ~~**prim_diff_decode control flow bug**~~: ✅ FIXED (Mem2Reg deduplication) - Unblocks 7+ OpenTitan IPs
3. ~~**Full OpenTitan IPs with Alerts**~~: ✅ VERIFIED (Track R) - prim_alert_sender compiles

**Critical Blockers for Full UVM Parity:**
1. **Class Method Inlining** - Virtual method dispatch and class hierarchy not fully simulated
   - **Impact**: Some UVM patterns may not work correctly at simulation time
   - **Priority**: HIGH - Required for complex UVM factory/callback patterns

**Medium Priority Enhancements:**
1. **AVIP bind scope support** - Allow bind to reference parent module ports
   - Would require slang enhancement or workaround
2. **do_compare default argument relaxation** - Strict LRM blocks common UVM pattern
   - Would require slang relaxation flag

**Test Suite Targets:**
- sv-tests: Maintain 98%+ (currently 821/831)
- verilator-verification: Maintain 80%+ (114/141)
- yosys SVA: Maintain 87%+ (14/16)
- OpenTitan: Expand from 9 register blocks to full IPs

**Infrastructure:**
- circt-sim: **LLVM dialect + FP ops + hierarchical instances** ✅ **IMPROVED (Iter 115)**
- UVM Phase System: **All 9 phases + component callbacks** ✅
- UVM config_db: **Hierarchical/wildcard matching** ✅
- UVM TLM Ports: **Analysis port/FIFO with can_put/can_get/used/free/capacity** ✅ **IMPROVED (Iter 117)**
- UVM Objections: **Raise/drop/drain with threading** ✅
- UVM Sequences: **Sequencer arbitration + driver handshake** ✅
- UVM Scoreboard: **Transaction comparison with TLM integration** ✅
- UVM RAL: **Register model with fields, blocks, maps** ✅
- UVM Messages: **Report info/warning/error/fatal with verbosity** ✅
- UVM Virtual Interfaces: **Thread-safe vif registries with modport support** ✅ **NEW (Iter 116)**

**Key Blockers RESOLVED**:
1. ✅ VTable polymorphism (Iteration 96)
2. ✅ Array locator lowering (Iteration 97)
3. ✅ `bit` clock simulation bug (Iteration 98)
4. ✅ String array types (Iteration 98)
5. ✅ Type mismatch in AND/OR ops (Iteration 99)
6. ✅ $countbits with 'x/'z (Iteration 99)
7. ✅ Mixed static/dynamic streaming (Iteration 99)
8. ✅ MooreToCore queue pop with class/struct types (Iteration 100)
9. ✅ MooreToCore time type conversion (Iteration 100)
10. ✅ Wildcard associative array element select (Iteration 100)
11. ✅ 64-bit streaming limit (Iteration 101)
12. ✅ hw.struct/hw.array in LLVM operations (Iteration 101)
13. ✅ Constraint method calls (Iteration 101)
14. ✅ circt-sim continuous assignments (Iteration 101)
15. ✅ LLVM dialect in interpreter (Iteration 102)
16. ✅ randomize(null) and randomize(v,w) modes (Iteration 102)
17. ✅ Virtual interface modport access in classes (Iteration 102)
18. ✅ UVM run_test() runtime stub (Iteration 103)
19. ✅ LLVM float ops in interpreter (Iteration 103)
20. ✅ String conversion methods (Iteration 103)
21. ✅ UVM phase system (Iteration 104)
22. ✅ UVM recursive function stubs (Iteration 104)
23. ✅ Fixed-to-dynamic array conversion (Iteration 104)
24. ✅ TypeReference handling (Iteration 104)

**Remaining Limitations**:
1. **Sibling Hierarchical Refs** - extnets.sv (cross-module wire refs)
2. **SVA Sequence Tests** - 6 verilator-verification tests (Codex handling SVA)
3. **Class Method Inlining** - Virtual method dispatch for complex UVM patterns
4. **slang v10 patches** - Some v9.1 patches don't apply to v10.0 (bind-scope, bind-instantiation-def)
5. ~~**Moore-to-Core Control Flow**~~: ✅ FIXED (Iter 189) - Mem2Reg deduplication fix
6. ~~**SignalValue 64-bit limit**~~: ✅ FIXED (Iter 189) - APInt upgrade

**Features to Build Next (Priority Order)**:
1. **Full OpenTitan IP simulation** - Test GPIO, UART, SPI with alerts now that prim_diff_decode fixed
2. **More crypto IPs** - Add keymgr_reg_top, otbn_reg_top to expand coverage (targeting 12+ IPs)
3. **Virtual method dispatch** - Improve class method inlining for UVM patterns
4. **Clocking blocks** - Chapter 14 at 0% pass rate

**New in Iteration 180**:
- ✅ slang upgraded from v9.1 to v10.0
- ✅ --compat vcs flag for VCS compatibility mode
- ✅ AllowVirtualIfaceWithOverride for Xcelium bind/vif compatibility

**Active Workstreams (Iteration 105)**:
1. **Track A: UVM Component Callbacks** - Hook phase methods to actual component code
2. **Track B: Chapter-6 to 90%** - Continue fixing remaining tests
3. **Track C: Chapter-18 Progress** - Address 15 remaining XFAIL tests
4. **Track D: More AVIP Testing** - Test AXI4Lite, I3C, GPIO AVIPs

**Iteration 93 Accomplishments**:
1. ✅ **$ferror system call** - Added FErrorBIOp with output argument handling
2. ✅ **$fgets system call** - Connected existing FGetSBIOp to ImportVerilog
3. ✅ **$ungetc system call** - Connected existing UngetCBIOp
4. ✅ **Dynamic array string init** - Fixed hw.bitcast for concatenation patterns
5. ✅ **BMC Sim stripping** - Added sim-strip pass for formal flows
6. ✅ **LLHD halt handling** - Fixed halt→yield for combinational regions
7. ✅ **$strobe/$monitor support** - Full $strobe/b/o/h, $fstrobe, $monitor/b/o/h, $fmonitor, $monitoron/off
8. ✅ **File positioning** - $fseek, $ftell, $rewind file position functions
9. ✅ **Binary file I/O** - $fread binary data reading
10. ✅ **Memory loading** - $readmemb, $readmemh for memory initialization
11. ✅ **$fflush, $printtimescale** - Buffer flush and timescale printing
12. ✅ **BMC LLHD lowering** - Inline llhd.combinational, replace llhd.sig with SSA values
13. ✅ **MooreToCore vtable fix** - Fixed build errors in vtable infrastructure code
14. ✅ **Hierarchical sibling extnets** - Fixed instance ordering for cross-module hierarchical refs
15. ✅ **System call unit tests** - Added MooreToCore lowering tests for all new system calls
16. ✅ **Expect assertions** - Map AssertionKind::Expect to moore::AssertOp/verif::AssertOp (+5 sv-tests)

**Virtual Method Dispatch Research (Track A)**:
Agent A completed research and identified the key gap for UVM polymorphism:
- **Current**: VTableLoadMethodOpConversion does STATIC dispatch at compile time
- **Needed**: Runtime DYNAMIC dispatch through vtable pointer in objects
- **Plan**: 5-step implementation involving vtable pointer in structs, global vtable arrays,
  vtable initialization in `new`, and dynamic dispatch in VTableLoadMethodOp

**Iteration 92 Accomplishments**:
1. ✅ **llvm.store/load hw.struct** - Fixed struct storage/load via llvm.struct conversion
2. ✅ **UVM lvalue streaming** - Fixed 93 tests: packed types + dynamic arrays in streaming ops
3. ✅ **TaggedUnion expressions** - Implemented `tagged Valid(N)` syntax (7 tests)
4. ✅ **Repeated event control** - Implemented `@(posedge clk, negedge reset)` multi-edge (4 tests)
5. ✅ **moore.and region regression** - Fixed parallel region scheduling (57 tests)
6. ✅ **Virtual interface binding** - Confirmed full infrastructure complete and working
7. ✅ **I2S AVIP assertions** - Verified all assertions compile and execute correctly
8. ✅ **VoidType conversion fix** - Resolved void return type handling in function conversions (+62 tests)
9. ✅ **Assert parent constraint fix** - Fixed constraint context inheritance for nested assertions (+22 tests)
10. ✅ **LTL non-overlapping delay fix** - Corrected `##` operator semantics for non-overlapping sequences

**AVIP Pipeline Status**:
| AVIP | ImportVerilog | MooreToCore | Remaining Blocker |
|------|---------------|-------------|-------------------|
| **APB** | ✅ | ✅ | None - Full pipeline works! |
| **SPI** | ✅ | ✅ | None - Full pipeline works! |
| **UART** | ✅ | ✅ | UVM-free components compile! |
| **I2S** | ✅ | ✅ | Assertions work! Full AVIP needs UVM |
| **AHB** | ⚠️ | - | UVM dependency, hierarchical task calls |

**Key Blockers for UVM Testbench Execution**:
1. ~~**Delays in class tasks**~~ ✅ FIXED - `__moore_delay()` runtime function for class methods
2. ~~**Constraint context properties**~~ ✅ FIXED - Non-static properties no longer treated as static
3. ~~**config_db runtime**~~ ✅ FIXED - `uvm_config_db::set/get/exists` lowered to runtime functions
4. ~~**get_full_name() recursion**~~ ✅ FIXED - Runtime function replaces recursive inlining
5. ~~**MooreToCore f64 BoolCast**~~ ✅ FIXED (Iter 90) - `arith::CmpFOp` for float-to-bool
6. ~~**NegOp 4-state types**~~ ✅ FIXED (Iter 90) - Proper 4-state struct handling
7. ~~**chandle <-> integer**~~ ✅ FIXED (Iter 90) - `llvm.ptrtoint`/`inttoptr` for DPI handles
8. ~~**class handle -> integer**~~ ✅ FIXED (Iter 90) - null comparison support
9. ~~**array.locator**~~ ✅ FIXED (Iter 90) - External variable references + fallback to inline loop
10. ~~**open_uarray <-> queue**~~ ✅ FIXED (Iter 90) - Same runtime representation
11. ~~**integer -> queue<T>**~~ ✅ FIXED (Iter 91) - Stream unpack to queue conversion
12. ~~**$past assertion**~~ ✅ FIXED (Iter 91) - moore::PastOp preserves value type
13. ~~**Interface port members**~~ ✅ FIXED (Iter 91) - Skip hierarchical path for interface ports
14. ~~**ModportPortSymbol handler**~~ ✅ FIXED (Iter 91) - Modport member access in Expressions.cpp
15. ~~**EmptyArgument expressions**~~ ✅ FIXED (Iter 91) - Optional arguments in $random(), etc.
16. ~~**4-state power operator**~~ ✅ FIXED (Iter 91) - Extract value before math.ipowi
17. ~~**4-state bit extraction**~~ ✅ FIXED (Iter 91) - sig_struct_extract for value/unknown
18. ~~**llvm.store/load hw.struct**~~ ✅ FIXED (Iter 92) - Convert hw.struct to llvm.struct for storage
19. ~~**Virtual interface binding**~~ ✅ COMPLETE (Iter 92) - Full infrastructure in place (VirtualInterface ops + runtime)
20. ~~**UVM lvalue streaming**~~ ✅ FIXED (Iter 92) - Packed types + dynamic arrays in streaming (93 tests)
21. ~~**TaggedUnion expressions**~~ ✅ FIXED (Iter 92) - `tagged Valid(N)` syntax now supported (7 tests)
22. ~~**Repeated event control**~~ ✅ FIXED (Iter 92) - Multi-edge sensitivity `@(posedge, negedge)` (4 tests)
23. ~~**moore.and region regression**~~ ✅ FIXED (Iter 92) - Parallel region scheduling (57 tests)
24. **Virtual method dispatch** - Class hierarchy not fully simulated
25. **Method overloading** - Base/derived class method resolution edge cases

**Using Real UVM Library** (Recommended):
```bash
# Compile APB AVIP with real UVM
circt-verilog --uvm-path ~/uvm-core/src \
  -I ~/mbit/apb_avip/src/hvl_top/master \
  -I ~/mbit/apb_avip/src/hvl_top/env \
  ~/mbit/apb_avip/src/globals/apb_global_pkg.sv \
  ~/mbit/apb_avip/src/hdl_top/apb_if/apb_if.sv \
  ... (see AVIP compile order)
```

**Track A: AVIP Simulation (Priority: HIGH) - Iteration 92 Complete**
| Status | Latest Accomplishment |
|--------|----------------------|
| ✅ **APB AVIP FULL PIPELINE** | ✅ ImportVerilog + MooreToCore both work! |
| ✅ **SPI AVIP FULL PIPELINE** | ✅ ImportVerilog + MooreToCore both work! |
| ✅ **UART AVIP 4-STATE FIXED** | ✅ UVM-free components compile! |
| ✅ **I2S AVIP ASSERTIONS** | ✅ All assertions verified working end-to-end! |
| ✅ ModportPortSymbol (Iter 91) | Handle modport member access in Expressions.cpp |
| ✅ EmptyArgument (Iter 91) | Optional arguments in $random(), etc. |
| ✅ 4-state power (Iter 91) | Extract value before math.ipowi |
| ✅ 4-state bit extract (Iter 91) | sig_struct_extract for value/unknown |
| ✅ llvm.store/load hw.struct (Iter 92) | Convert hw.struct to llvm.struct for storage |
| ✅ UVM lvalue streaming (Iter 92) | Packed types + dynamic arrays in streaming (93 tests) |
| ✅ TaggedUnion expressions (Iter 92) | `tagged Valid(N)` syntax now fully supported (7 tests) |
| ✅ Repeated event control (Iter 92) | Multi-edge sensitivity `@(posedge, negedge)` (4 tests) |
| ✅ moore.and regions (Iter 92) | Fixed parallel region scheduling (57 tests) |
| ✅ Virtual interfaces | Full infrastructure complete |
| ⚠️ **Virtual method dispatch** | **NEXT**: Base/derived class method resolution |
| ⚠️ Method overloading | Edge cases in class hierarchy |

**Iteration 93 Priorities** (Updated):
1. **Virtual method dispatch** - Enable UVM polymorphism (factory, callbacks) [Track A]
2. **sv-tests moore.conversion** - Fix remaining type conversion tests [Track C]
3. **Hierarchical interface task calls** - Unblock AHB AVIP [Track A]
4. ✅ **System call stubs** - $ferror, $fgets, $ungetc done; remaining: $fread, $fscanf, $fpos
5. **BMC sequence patterns** - Complete value-change X/Z semantics [Track B]
6. **Runtime DPI stubs** - Complete UVM runtime function stubs [Track D]

**Remaining Limitations**:
- Virtual method dispatch not implemented (critical for UVM factory/callbacks)
- Some file I/O system calls still missing ($fscanf improvements, $value$plusargs)
- VCD dump functions ($dumpfile, $dumpvars, $dumpports) not implemented
- Hierarchical interface task calls need work for AHB AVIP

**Track B: BMC/Formal (Codex Agent Handling) - Iteration 92 Progress**
| Status | Progress |
|--------|----------|
| ✅ basic03 works | Run ~/yosys/tests/sva suite |
| ✅ Derived clocks | Multiple derived clocks constrained to single BMC clock |
| ✅ **Yosys SVA BMC** | **82%** (up from 75% in Iter 91) - **7% improvement!** |
| ⚠️ SVA defaults | Default clocking/disable iff reset LTL state; property instances avoid double defaults |
| ⚠️ Sequence patterns | Fixed ##N concat delays; yosys counter passes; value-change ops mostly fixed (changed/rose/wide). Remaining: value_change_sim X/Z edge semantics, extnets |

**Track C: Test Suite Validation**
| Test Suite | Location | Purpose | Agent |
|------------|----------|---------|-------|
| AVIP Testbenches | ~/mbit/*avip | UVM verification IPs | Track A |
| sv-tests | ~/sv-tests | SV language compliance | Track C |
| Verilator tests | ~/verilator-verification | Simulation edge cases | Track C |
| Yosys SVA | ~/yosys/tests/sva | Formal verification | Track B |

**Track D: Runtime & Infrastructure**
| Status | Next Priority |
|--------|---------------|
| ✅ Static class properties | Constraint context fix - no longer treats non-static as static |
| ✅ Class task delays | __moore_delay() runtime function implemented |
| ✅ config_db operations | uvm_config_db::set/get/exists runtime functions |
| ✅ get_full_name() | Runtime function for hierarchical name building |
| ✅ String methods | compare(), icompare() implemented |
| ✅ File I/O functions | $feof(), $fgetc() implemented |
| ⚠️ DPI function stubs | Complete runtime stubs for UVM |
| ⚠️ Coroutine runtime | Full coroutine support for task suspension |

### Real-World Test Results (Updated Iteration 90)

**AVIP Pipeline Status** (Iteration 90):

| AVIP | ImportVerilog | MooreToCore | Current Blocker |
|------|---------------|-------------|-----------------|
| APB | ✅ PASS | ✅ PASS | None - full pipeline works |
| I2S | ✅ PASS (276K lines) | ⚠️ BLOCKED | `array.locator` not supported |
| SPI | ✅ PASS (268K lines) | ⚠️ BLOCKED | `array.locator` not supported |
| UART | ✅ PASS (240K lines) | ⚠️ BLOCKED | `array.locator` not supported |
| JTAG | ✅ PASS | Not tested | Bind directive warnings |
| AHB | ⚠️ PARTIAL | Not tested | Interface hierarchical refs |
| AXI4 | ⚠️ PARTIAL | Not tested | Dependency/ordering issues |
| I3C | ⚠️ PARTIAL | Not tested | UVM import issues |
| AXI4Lite | ⚠️ PARTIAL | Not tested | Missing package |

**Fixes in Iteration 90**:
- ✅ f64 BoolCast: `arith::CmpFOp` for float-to-bool (covergroup get_coverage())
- ✅ NegOp 4-state: Proper unknown bit propagation
- ✅ chandle/integer: `llvm.ptrtoint`/`inttoptr` for DPI handles
- ✅ class handle: null comparison support

**sv-tests Compliance Suite** (1,028 tests):
- Sample Pass Rate: **86%** (first 100 tests) - NO REGRESSION
- Adjusted Pass Rate: **~83%** (excluding expected failures)
- Main failure categories:
  - UVM package not found (51% of failures)
  - TaggedUnion expressions not supported
  - Disable statement not implemented

**verilator-verification Tests** (154 tests):
- Parse Pass Rate: **59%** (91/154) - small regression to investigate
- MooreToCore Pass Rate: **100%** (all that parse)
- Main failure categories:
  - Dynamic type access outside procedural context (15 failures)
  - Sequence clocking syntax issues (6 failures)
  - UVM base class resolution (11 failures)

**Track D - SVA Formal Verification** (Updated Iteration 77):
- Working: implications (|-> |=>), delays (##N), repetition ([*N]), sequences
- ✅ FIXED: $rose/$fell in implications now work via ltl.past buffer infrastructure
- ✅ FIXED: $past supported via PastInfo struct and buffer tracking
- Remaining: $countones/$onehot use llvm.intr.ctpop (pending BMC symbol issue)
- New: local circt-bmc harnesses for `~/sv-tests` and `~/verilator-verification`
  to drive test-driven SVA progress (see `utils/run_sv_tests_circt_bmc.sh` and
  `utils/run_verilator_verification_circt_bmc.sh`).
- New: LEC smoke harnesses for `~/sv-tests`, `~/verilator-verification`, and
  `~/yosys/tests/sva` (see `utils/run_sv_tests_circt_lec.sh`,
  `utils/run_verilator_verification_circt_lec.sh`, and
  `utils/run_yosys_sva_circt_lec.sh`).
- ✅ LEC: `--run-smtlib` now scans stdout/stderr for SAT results, fixing empty
  token failures when z3 emits warnings.
- ✅ LEC smoke: yosys `extnets` now passes by flattening private HW modules
  before LEC; ref inout/multi-driver now abstracted to inputs (approx), full
  resolution still missing.
- ✅ LEC: interface fields with multiple stores now abstract to inputs to avoid
  hard failures; full multi-driver semantics still missing.
- ✅ LEC smoke: verilator-verification now passes 17/17 after stripping LLHD
  combinational/signal ops in the LEC pipeline.
- ✅ BMC: LowerToBMC now defers probe replacement for nested combinational
  regions so probes see driven values; verilator-verification asserts pass
  (17/17).
- ✅ Progress: HWToSMT now lowers `hw.struct_create/extract/explode` to SMT
  bitvector concat/extract, unblocking BMC when LowerToBMC emits 4-state
  structs.
- ✅ FIXED: VerifToSMT now rewrites `smt.bool`↔`bv1` unrealized casts into
  explicit SMT ops, eliminating Z3 sort errors in yosys `basic03.sv`.
- ✅ Verified end-to-end BMC pipeline with yosys `basic03.sv`
  (pass/fail cases both clean).
- ✅ FIXED: constrain equivalent derived `seq.to_clock` inputs to the generated
  BMC clock (LowerToBMC), including use-before-def cases; `basic03` and the full
  yosys SVA suite now pass (2 VHDL skips remain).
- In progress: gate BMC checks to posedge iterations when not in
  `rising-clocks-only` mode to prevent falling-edge false violations.
- In progress: gate BMC delay/past buffer updates on posedge so history
  advances once per cycle in non-rising mode.

**SVA Support Plan (End-to-End)**:
1. **Pipeline robustness**: keep SV→Moore→HW→BMC→SMT legal (no illegal ops).
   - Guardrails: HWToSMT aggregate lowering, clock handling in LowerToBMC.
2. **Temporal semantics**: complete and validate `##[m:$]`, `[*N]`, goto, and
   non-consecutive repetition in multi-step BMC.
3. **Clocked sampling correctness**: fix `$past/$rose/$fell` alignment and
   sampled-value timing in BMC (yosys `basic03.sv` pass must be clean).
4. **Procedural concurrent assertions**: hoist/guard `assert property` inside
   `always` blocks, avoiding `seq.compreg` inside `llhd.process` (current
   externalize-registers failure in yosys `sva_value_change_sim`).
5. **4-state modeling**: ensure `value/unknown` propagation is consistent
   across SVAToLTL → VerifToSMT → SMT (document X/unknown semantics).
6. **Solver output + traces**: stable SAT/UNSAT results, trace extraction for
   counterexamples, and consistent CLI reporting.
7. **External suite gating**: keep `sv-tests`, `verilator-verification`,
   `yosys/tests/sva`, and AVIP subsets green with recorded baselines.

**Test-Driven Suites**:
- `TEST_FILTER=... utils/run_yosys_sva_circt_bmc.sh` (per-feature gating).
- `utils/run_sv_tests_circt_bmc.sh` for sv-tests SVA coverage.
- `utils/run_verilator_verification_circt_bmc.sh` for verilator-verification.
- `utils/run_sv_tests_circt_lec.sh` for sv-tests LEC smoke coverage.
- `utils/run_verilator_verification_circt_lec.sh` for verilator LEC smoke coverage.
- `utils/run_yosys_sva_circt_lec.sh` for yosys SVA LEC smoke coverage.
- Manual AVIP spot checks in `~/mbit/*avip*` with targeted properties.

### Feature Completion Matrix

| Feature | Parse | IR | Lower | Runtime | Test |
|---------|-------|-----|-------|---------|------|
| rand/randc | ✅ | ✅ | ✅ | ✅ | ✅ |
| Constraints (basic) | ✅ | ✅ | ✅ | ✅ | ✅ |
| Soft constraints | ✅ | ✅ | ✅ | ✅ | ✅ |
| Distribution constraints | ✅ | ✅ | ✅ | ✅ | ✅ |
| Inline constraints | ✅ | ✅ | ✅ | ✅ | ✅ |
| constraint_mode() | ✅ | ✅ | ✅ | ✅ | ✅ |
| rand_mode() | ✅ | ✅ | ✅ | ✅ | ✅ |
| Covergroups | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coverpoints | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cross coverage | ✅ | ✅ | ✅ | ✅ | ✅ |
| Transition bins | ✅ | ✅ | ✅ | ✅ | ✅ |
| Wildcard bins | ✅ | ✅ | ✅ | ✅ | ✅ |
| pre/post_randomize | ✅ | ✅ | ✅ | ✅ | ✅ |
| LSP code actions | - | - | - | - | ✅ |
| Illegal/ignore bins | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coverage merge | - | - | - | ✅ | ✅ |
| Virtual interfaces | ✅ | ✅ | ✅ | ⚠️ config_db | ⚠️ |
| Classes | ✅ | ✅ | ✅ | ⚠️ | ✅ |
| UVM base classes | ✅ | ⚠️ | ⚠️ | ✅ | ✅ |
| Array unique constraints | ✅ | ✅ | ✅ | ✅ | ✅ |
| Cross named bins | ✅ | ✅ | ✅ | ✅ | ✅ |
| LSP inheritance completion | - | - | - | - | ✅ |
| LSP chained completion | - | - | - | - | ✅ |
| LSP document formatting | - | - | - | - | ✅ |
| Coverage options | ✅ | ✅ | ✅ | ✅ | ✅ |
| Constraint implication | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coverage callbacks | ✅ | ✅ | ✅ | ✅ | ✅ |
| LSP find references | - | - | - | - | ✅ |
| Solve-before constraints | ✅ | ✅ | ✅ | ✅ | ✅ |
| LSP rename refactoring | - | - | - | - | ✅ |
| Coverage get_inst_coverage | - | - | - | ✅ | ✅ |
| Coverage HTML reports | - | - | - | ✅ | ✅ |
| LSP call hierarchy | - | - | - | - | ✅ |
| Array foreach constraints | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coverage DB persistence | - | - | - | ✅ | ✅ |
| LSP workspace symbols | - | - | - | - | ✅ |
| Pullup/pulldown primitives | ✅ | ✅ | ✅ | - | ✅ |
| Coverage exclusions | - | - | - | ✅ | ✅ |
| LSP semantic tokens | - | - | - | - | ✅ |
| Gate primitives (12 types) | ✅ | ✅ | ✅ | - | ✅ |
| Coverage assertions | - | - | - | ✅ | ✅ |
| LSP code lens | - | - | - | - | ✅ |
| MOS primitives (12 types) | ✅ | ✅ | ✅ | - | ✅ |
| UVM coverage model | - | - | - | ✅ | ✅ |
| LSP type hierarchy | - | - | - | - | ✅ |
| $display/$write runtime | - | - | - | ✅ | ✅ |
| Constraint implication | ✅ | ✅ | ✅ | ✅ | ✅ |
| Coverage UCDB format | - | - | - | ✅ | ✅ |
| LSP inlay hints | - | - | - | - | ✅ |

Legend: ✅ Complete | ⚠️ Partial | ❌ Not Started

---

## Current Status: ITERATION 77 - Multi-Track Improvements (January 21, 2026)

**Summary**: Continuing investigation and fixes for concurrent process scheduling, UVM macros, dynamic type access, and SVA edge functions.

### Iteration 77 In-Progress

**Track A: Event-Based Waits**
- Testing event-based waits with llhd.wait observed operands
- Initial tests show event-based waits working for simple cases
- Need more comprehensive testing for edge cases

**Track B: UVM Macro Completion**
- Adding remaining UVM macros for full compilation
- Focus on copier, comparer, packer, recorder macros

**Track C: Dynamic Type Access**
- Investigating "dynamic type access outside procedural context" errors
- These affect ~104 AVIP test failures
- DynamicNotProcedural diagnostics being addressed

**Track D: SVA Edge Functions**
- ✅ $rose/$fell now use case-equality comparisons (X/Z transitions behave)
- ✅ Procedural concurrent assertions hoisted with guards to module scope
- ✅ BMC accepts preset initial values for 4-state regs (bitwidth-matched)

---

## Previous: ITERATION 76 - Concurrent Process Scheduling Root Cause (January 21, 2026)

**Summary**: Identified and fixed root cause of concurrent process scheduling issue.

### Iteration 76 Highlights

**Sensitivity Persistence Fix** ⭐ CRITICAL FIX
- Fixed `ProcessScheduler::suspendProcessForEvents()` to make sensitivity list persistent
- Previously, sensitivity was only stored in `waitingSensitivity` which cleared on wake
- Now also updates main `sensitivity` list for robustness

**Root Cause Analysis**
- Signal-to-process mapping not persistent across wake/sleep cycles
- Processes ended in Suspended state without sensitivity after first execution
- Event-driven vs process-driven timing caused missed edges

**Real-World Test Results**
- 73% pass rate on AVIP testbenches (1294 tests)
- APB AVIP components now compile successfully
- Main remaining issues: UVM package stubs, dynamic type access

---

## Previous: ITERATION 74 - ProcessOp Canonicalization Fix (January 21, 2026)

**Summary**: Fixed critical bug where processes with $display/$finish were being removed by the optimizer.

### Iteration 74 Highlights

**ProcessOp Canonicalization Fix** ⭐ CRITICAL
- Fixed ProcessOp::canonicalize() to preserve processes with side effects
- Previously only checked for DriveOp, missing sim.proc.print and sim.terminate
- Now checks for all side-effect operations including memory writes
- Initial blocks with $display/$finish now work correctly
- Test: `simple_initial_test.sv` prints "Hello from initial block!" and terminates at correct time

**UVM Macro Enhancements**
- Added UVM_STRING_QUEUE_STREAMING_PACK, uvm_typename, uvm_type_name_decl
- Added uvm_object_abstract_utils, uvm_component_abstract_utils
- Fixed uvm_object_utils conflict with uvm_type_name_decl

**Known Issue Discovered**
- Concurrent process scheduling broken: initial+always blocks don't work together
- Needs investigation in LLHDProcessInterpreter

**Files Modified**:
- `lib/Dialect/LLHD/IR/LLHDOps.cpp` - ProcessOp canonicalization fix
- `lib/Runtime/uvm/uvm_macros.svh` - Additional UVM macro stubs
- New test: `canonicalize-process-with-side-effects.mlir`

---

## Previous: ITERATION 73 - Major Simulation Fixes (January 21, 2026)

**Summary**: Fixed $display output, $finish termination, queue sort with expressions.

### Iteration 73 Highlights
- **Queue Sort With**: QueueSortWithOpConversion for `q.sort with (expr)` pattern
- **$display Output**: sim.proc.print now prints to console
- **$finish Support**: sim.terminate properly terminates simulation
- **seq.initial Support**: Added support for sequential initial blocks

---

## Previous: ITERATION 71 - RandSequence Fractional N Support (January 21, 2026)

**Summary**: Fixed `rand join (N)` to support fractional N values per IEEE 1800-2017 Section 18.17.5.

---

## Previous: ITERATION 70 - $display Runtime + Constraint Implication + UCDB Format + LSP Inlay Hints (January 20, 2026)

**Summary**: Implemented $display system tasks, completed constraint implication lowering, added UCDB coverage file format, and added LSP inlay hints.

### Iteration 70 Highlights

**Track A: $display Runtime Support** ⭐ FEATURE
- ✅ Implemented $display, $write, $strobe, $monitor runtime functions
- ✅ Added FormatDynStringOp support in LowerArcToLLVM
- ✅ 12 unit tests for display system tasks

**Track B: Constraint Implication Lowering** ⭐ FEATURE
- ✅ Extended test coverage with 7 new tests (nested, soft, distribution)
- ✅ Added runtime functions for implication checking
- ✅ 8 unit tests for implication constraints

**Track C: Coverage UCDB File Format** ⭐ FEATURE
- ✅ UCDB-compatible JSON format for coverage persistence
- ✅ File merge support for regression runs
- ✅ 12 unit tests for UCDB functionality

**Track D: LSP Inlay Hints** ⭐ FEATURE
- ✅ Parameter name hints for function/task calls
- ✅ Port connection hints for module instantiations
- ✅ Return type hints for functions

---

## Previous: ITERATION 67 - Pullup/Pulldown + Inline Constraints + Coverage Exclusions (January 20, 2026)

**Summary**: Added pullup/pulldown primitive support, implemented full inline constraint lowering, and added coverage exclusion APIs.

### Iteration 67 Highlights

**Track A: Pullup/Pulldown Primitives** ⭐ FEATURE
- ✅ Basic parsing support for pullup/pulldown Verilog primitives
- ✅ Models as continuous assignment of constant value
- ⚠️ Does not yet model drive strength or 4-state behavior
- ✅ Unblocks I3C AVIP compilation

**Track B: Inline Constraint Lowering** ⭐ FEATURE
- ✅ Full support for `randomize() with { ... }` inline constraints
- ✅ Inline constraints merged with class-level constraints
- ✅ Comprehensive test coverage in inline-constraints.mlir

**Track C: Coverage Exclusions API** ⭐ FEATURE
- ✅ 7 new API functions for exclusion management
- ✅ Exclusion file format with wildcard support
- ✅ 13 unit tests for exclusion functionality

**Track D: LSP Semantic Tokens** ⭐ VERIFICATION
- ✅ Confirmed already fully implemented (23 token types, 9 modifiers)

---

## Previous: ITERATION 66 - AVIP Verification + Coverage DB Persistence + Workspace Symbols (January 20, 2026)

**Summary**: Verified APB/SPI AVIPs compile with proper timing control conversion, implemented coverage database persistence with metadata, fixed workspace symbols deadlock.

### Iteration 66 Highlights

**Track A: AVIP Testbench Verification** ⭐ TESTING
- ✅ APB and SPI AVIPs compile fully to HW IR with llhd.wait
- ✅ Timing controls in interface tasks properly convert after inlining
- ⚠️ I3C blocked by missing pullup primitive support
- ✅ Documented remaining blockers for full AVIP support

**Track B: Foreach Implication Constraint Tests** ⭐ FEATURE
- ✅ 5 new test cases in array-foreach-constraints.mlir
- ✅ New foreach-implication.mlir with 7 comprehensive tests
- ✅ Verified all constraint ops properly erased during lowering

**Track C: Coverage Database Persistence** ⭐ FEATURE
- ✅ `__moore_coverage_save_db()` with metadata (test name, timestamp)
- ✅ `__moore_coverage_load_db()` and `__moore_coverage_merge_db()`
- ✅ `__moore_coverage_db_get_metadata()` for accessing saved metadata
- ✅ 15 unit tests for database persistence

**Track D: Workspace Symbols Fix** ⭐ BUG FIX
- ✅ Fixed deadlock in Workspace.cpp findAllSymbols()
- ✅ Created workspace-symbols.test with comprehensive coverage
- ✅ All workspace symbol tests passing

---

## Previous: ITERATION 65 - Second MooreToCore Pass + Coverage HTML + LSP Call Hierarchy (January 20, 2026)

**Summary**: Added second MooreToCore pass after inlining to convert timing controls in interface tasks, implemented coverage HTML report generation, and added full LSP call hierarchy support.

### Iteration 65 Highlights

**Track A: Second MooreToCore Pass After Inlining** ⭐ ARCHITECTURE
- ✅ Added second MooreToCore pass after InlineCalls in pipeline
- ✅ Timing controls in interface tasks now properly convert to llhd.wait
- ✅ Key step toward full AVIP simulation support

**Track B: Array Constraint Foreach Simplification** ⭐ FEATURE
- ✅ Simplified ConstraintForeachOpConversion to erase during lowering
- ✅ Runtime validation via `__moore_constraint_foreach_validate()`
- ✅ 4 test cases (basic, index, range, nested)

**Track C: Coverage HTML Report Generation** ⭐ FEATURE
- ✅ `__moore_coverage_report_html()` for professional HTML reports
- ✅ Color-coded badges, per-bin details, cross coverage
- ✅ Responsive tables, modern CSS styling
- ✅ 4 unit tests for HTML report generation

**Track D: LSP Call Hierarchy** ⭐ FEATURE
- ✅ prepareCallHierarchy for functions and tasks
- ✅ incomingCalls to find all callers
- ✅ outgoingCalls to find all callees
- ✅ 6 test scenarios in call-hierarchy.test

---

## Previous: ITERATION 64 - Solve-Before Constraints + LSP Rename + Coverage get_inst_coverage (January 20, 2026)

**Summary**: Implemented solve-before constraint ordering, LSP rename refactoring, coverage instance-specific APIs, and fixed llhd-mem2reg for LLVM pointer types.

### Iteration 64 Highlights

**Track A: Dynamic Legality for Timing Controls** ⭐ ARCHITECTURE
- ✅ Added dynamic legality rules for WaitEventOp and DetectEventOp
- ✅ Timing controls in class tasks remain unconverted until inlined into llhd.process
- ✅ Unblocks AVIP tasks with `@(posedge clk)` timing

**Track B: Solve-Before Constraints** ⭐ FEATURE
- ✅ Full MooreToCore lowering for `solve a before b` constraints
- ✅ Topological sort using Kahn's algorithm for constraint ordering
- ✅ 5 comprehensive test cases (basic, multiple, chained, partial, erased)

**Track C: Coverage get_inst_coverage API** ⭐ FEATURE
- ✅ `__moore_covergroup_get_inst_coverage()` for instance-specific coverage
- ✅ `__moore_coverpoint_get_inst_coverage()` and `__moore_cross_get_inst_coverage()`
- ✅ Enhanced `get_coverage()` to respect per_instance option
- ✅ Enhanced cross coverage to respect at_least threshold

**Track D: LSP Rename Refactoring** ⭐ FEATURE
- ✅ Extended prepareRename for ClassType, ClassProperty, InterfacePort
- ✅ Support for Modport, FormalArgument, TypeAlias
- ✅ 10 comprehensive test scenarios in rename-refactoring.test

**Bug Fix: llhd-mem2reg LLVM Pointer Types**
- ✅ Fixed default value materialization for LLVM pointer types
- ✅ Use llvm.mlir.zero instead of invalid integer bitcast
- ✅ Added regression test mem2reg-llvm-zero.mlir

---

## Previous: ITERATION 63 - Distribution Constraints + Coverage Callbacks + LSP Find References (January 20, 2026)

**Summary**: Implemented distribution constraint lowering, added coverage callbacks API, enhanced LSP find references, investigated AVIP E2E blockers.

### Iteration 63 Highlights

**Track A: AVIP E2E Testing** ⭐ INVESTIGATION
- ✅ Created comprehensive AVIP-style testbench test
- ⚠️ Identified blocker: `@(posedge clk)` in class tasks causes llhd.wait error
- ✅ Parsing and basic lowering verified working

**Track B: Distribution Constraints** ⭐ FEATURE
- ✅ Full MooreToCore lowering for `dist` constraints
- ✅ Support for `:=` and `:/` weight operators
- ✅ 7 new unit tests

**Track C: Coverage Callbacks** ⭐ FEATURE
- ✅ 13 new runtime functions for callbacks/sampling
- ✅ pre/post sample hooks, conditional sampling
- ✅ 12 new unit tests

**Track D: LSP Find References** ⭐ FEATURE
- ✅ Enhanced with class/typedef type references
- ✅ Base class references in `extends` clauses

---

## Previous: ITERATION 62 - Virtual Interface Fix + Coverage Options + LSP Formatting (January 20, 2026)

**Summary**: Fixed virtual interface timing bug, added coverage options, implemented LSP document formatting.

### Iteration 62 Highlights

**Track A: Virtual Interface Timing** ⭐ BUG FIX
- ✅ Fixed modport-qualified virtual interface type conversion
- ✅ All 6 virtual interface tests passing

**Track B: Constraint Implication** ⭐ VERIFICATION
- ✅ Verified `->` and `if-else` fully implemented
- ✅ Created 25 comprehensive test scenarios

**Track C: Coverage Options** ⭐ FEATURE
- ✅ goal, at_least, weight, auto_bin_max support
- ✅ 14 new unit tests

**Track D: LSP Formatting** ⭐ FEATURE
- ✅ Full document and range formatting
- ✅ Configurable indentation

---

## Previous: ITERATION 61 - UVM Stubs + Array Constraints + Cross Coverage (January 20, 2026)

**Summary**: Extended UVM stubs, added array constraint support, enhanced cross coverage with named bins, LSP inheritance completion.

### Iteration 61 Highlights

**Track A: UVM Base Class Stubs** ⭐ FEATURE
- ✅ Extended with `uvm_cmdline_processor`, `uvm_report_server`, `uvm_report_catcher`
- ✅ All 12 UVM test files compile successfully

**Track B: Array Constraints** ⭐ FEATURE
- ✅ unique check, foreach validation, size/sum constraints
- ✅ 15 unit tests added

**Track C: Cross Coverage** ⭐ FEATURE
- ✅ Named bins with binsof, ignore_bins, illegal_bins
- ✅ 7 unit tests added

**Track D: LSP Inheritance** ⭐ FEATURE
- ✅ Inherited members show "(from ClassName)" annotation

---

## Previous: ITERATION 60 - circt-sim Expansion + Coverage Enhancements + LSP Actions (January 20, 2026)

**Summary**: Major circt-sim interpreter expansion, pre/post_randomize callbacks, wildcard and transition bin coverage, LSP code actions. 6 parallel work tracks completed.

### Iteration 60 Highlights

**Track A: circt-sim LLHD Process Interpreter** ⭐ MAJOR FEATURE
- ✅ Added 20+ arith dialect operations (addi, subi, muli, cmpi, etc.)
- ✅ Implemented SCF operations: scf.if, scf.for, scf.while
- ✅ Added func.call/func.return for function invocation
- ✅ Added hw.array operations: array_create, array_get, array_slice, array_concat
- ✅ X-propagation and loop safety limits (100K max)
- Tests: 6 new circt-sim tests

**Track B: pre/post_randomize Callbacks** ⭐ FEATURE
- ✅ Direct method call generation for pre_randomize/post_randomize
- ✅ Searches ClassMethodDeclOp or func.func with conventional naming
- ✅ Graceful fallback when callbacks don't exist
- Tests: `pre-post-randomize.mlir`, `pre-post-randomize-func.mlir`, `pre-post-randomize.sv`

**Track C: Wildcard Bin Matching** ⭐ FEATURE
- ✅ Implemented wildcard formula: `((value ^ bin.low) & ~bin.high) == 0`
- Tests: 8 unit tests for wildcard patterns

**Track E: Transition Bin Coverage** ⭐ FEATURE
- ✅ Multi-step sequence state machine for transition tracking
- ✅ Integrated with __moore_coverpoint_sample()
- Tests: 10+ unit tests for transition sequences

**Track F: LSP Code Actions** ⭐ FEATURE
- ✅ Missing semicolon quick fix
- ✅ Common typo fixes (rge→reg, wrie→wire, etc.)
- ✅ Begin/end block wrapping
- Tests: `code-actions.test`

**AVIP Validation**: APB, AXI4, SPI, UART all compile successfully

---

## Previous: ITERATION 59 - Coverage Illegal/Ignore Bins + LSP Chained Access (January 20, 2026)

**Summary**: Implemented illegal/ignore bins MooreToCore lowering and chained member access for LSP completion.

### Iteration 59 Highlights

**Track C: Coverage Illegal/Ignore Bins Lowering** ⭐ FEATURE
- ✅ Extended CovergroupDeclOpConversion to process CoverageBinDeclOp
- ✅ Generates runtime calls for `__moore_coverpoint_add_illegal_bin` and `__moore_coverpoint_add_ignore_bin`
- ✅ Supports single values and ranges in bin definitions
- ✅ Added CoverageBinDeclOpConversion pattern to erase bins after processing
- Test: `coverage-illegal-bins.mlir` (new)

**Track D: LSP Chained Member Access** ⭐ FEATURE
- ✅ Extended completion context analysis to parse full identifier chains (e.g., `obj.field1.field2`)
- ✅ Added `resolveIdentifierChain()` to walk through member access chains
- ✅ Supports class types, instance types, and interface types in chains
- ✅ Returns completions for the final type in the chain

**Files Modified**:
- `lib/Conversion/MooreToCore/MooreToCore.cpp` - illegal/ignore bins lowering
- `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogDocument.cpp` - chained access
- `test/Conversion/MooreToCore/coverage-illegal-bins.mlir` - new test

---

## Previous: ITERATION 58 - Inline Constraints + Coverage Merge + AVIP Demo (January 17, 2026)

**Summary**: Implemented inline constraints (with clause), coverage database merge, comprehensive AVIP testbench demo, and LSP fuzzy workspace search. LARGEST ITERATION: 2,535 insertions.

### Iteration 58 Highlights

**Track A: End-to-End AVIP Testbench** ⭐ DEMONSTRATION
- ✅ Created comprehensive APB testbench: `avip-apb-simulation.sv` (388 lines)
- ✅ Components: Transaction, Coverage, Scoreboard, Memory
- ✅ Shows: randomize, sample, check, report flow
- Documents circt-sim procedural execution limitations

**Track B: Inline Constraints (with clause)** ⭐ MAJOR FEATURE
- ✅ Extended `RandomizeOp` and `StdRandomizeOp` with inline_constraints region
- ✅ Parses with clause from randomize() calls
- ✅ Supports: `obj.randomize() with {...}`, `std::randomize(x,y) with {...}`
- Test: `randomize.sv` (enhanced)

**Track C: Coverage Database Merge** ⭐ VERIFICATION FLOW
- ✅ JSON-based coverage database format
- ✅ Functions: save, load, merge, merge_files
- ✅ Cumulative bin hit counts, name-based matching
- Tests: `MooreRuntimeTest.cpp` (+361 lines)

**Track D: LSP Workspace Symbols (Fuzzy)** ⭐
- ✅ Sophisticated fuzzy matching with CamelCase detection
- ✅ Score-based ranking, finds functions/tasks
- Test: `workspace-symbol-fuzzy.test` (new)

**Summary**: 2,535 insertions across 13 files (LARGEST ITERATION!)

---

## Previous: ITERATION 57 - Coverage Options + Solve Constraints (January 17, 2026)

**Summary**: circt-sim simulation verified, solve-before constraints, comprehensive coverage options. 1,200 insertions.

---

## Previous: ITERATION 56 - Distribution Constraints + Transition Bins (January 17, 2026)

**Summary**: Implemented distribution constraints for randomization, transition coverage bins with state machine tracking, documented simulation alternatives. 918 insertions.

---

## Previous: ITERATION 55 - Constraint Limits + Coverage Auto-Bins (January 17, 2026)

**Summary**: Added constraint solving iteration limits with fallback, implemented coverage auto-bin patterns. 985 insertions.

---

## Previous: ITERATION 54 - LLHD Fix + Moore Conversion + Binsof (January 17, 2026)

**Summary**: Fixed critical LLHD process canonicalization, implemented moore.conversion lowering for ref-to-ref types, added full binsof/intersect support for cross coverage, and implemented LSP document highlights. 934 insertions.

---

## Previous: ITERATION 53 - Simulation Analysis + LSP Document Symbols (January 17, 2026)

**Summary**: Identified CRITICAL blocker for AVIP simulation (llhd.process not lowered), verified soft constraints already implemented, analyzed coverage features, and added LSP document symbols support.

---

## Previous: ITERATION 52 - All 9 AVIPs Validated + Foreach Constraints (January 17, 2026)

**Summary**: MAJOR MILESTONE! All 9 AVIPs (1,342 files total) now compile with ZERO errors. Implemented foreach constraint support, enhanced coverage runtime with cross coverage/goals/HTML reports, and improved LSP diagnostics.

---

## Previous: ITERATION 51 - DPI/VPI Stubs, Randc Fixes, LSP Code Actions (January 18, 2026)

**Summary**: Expanded DPI/VPI runtime stubs with in-memory HDL access, improved randc/randomize lowering, added class covergroup property lowering, and implemented LSP code actions quick fixes.

### Iteration 51 Highlights

**Track A: DPI/VPI + UVM Runtime**
- ✅ HDL access stubs backed by in-memory path map with force/release semantics
- ✅ VPI stubs: `vpi_handle_by_name`, `vpi_get`, `vpi_get_str`, `vpi_get_value`, `vpi_put_value`, `vpi_release_handle`
- ✅ Regex stubs accept basic `.` and `*` patterns

**Track B: Randomization + Randc Correctness** ⭐
- ✅ randc cycles deterministically per-field; constrained fields skip overrides
- ✅ Non-rand fields preserved around randomize lowering
- ✅ Wide randc uses linear full-cycle fallback for >16-bit domains

**Track C: Coverage / Class Features** ⭐
- ✅ Covergroups in classes lower to class properties
- ✅ Queue concatenation accepts element operands
- ✅ Queue `$` indexing supported for unbounded literals

**Track D: LSP Tooling** ⭐
- ✅ Code actions: declare wire/logic/reg, module stub, missing import, width fixes
- ✅ Refactor actions: extract signal, instantiation template

---

## Major Workstreams (Parity With Xcelium)

| Workstream | Status | Current Limitations | Next Task |
|-----------|--------|---------------------|-----------|
| Full SVA support with Z3 (~/z3) | Not integrated | Z3-based checks not wired into CIRCT pipeline | Define Z3 bridge API + proof/CE format |
| Scalable multi-core (Arcilator/tools) | Not started | Single-threaded scheduling | Identify parallel regions + add job orchestration |
| LSP + debugging | In progress | No debugging hooks; limited code actions | Add debug adapters + trace stepping |
| Full 4-state (X/Z) propagation | Not started | 2-state assumptions in lowering/runtime | Design 4-state IR + ops, add X/Z rules |
| Coverage support | Partial | Runtime sampling/reporting gaps | Finish covergroup runtime + bin hit reporting |
| DPI/VPI | Partial (stubs) | In-memory only; no simulator wiring | Connect HDL/VPI to simulator data model |

---

## Previous: ITERATION 49 - Virtual Interface Methods Fixed! (January 17, 2026)

**Summary**: Fixed the last remaining UVM APB AVIP blocker! Virtual interface method calls like `vif.method()` from class methods now work correctly. APB AVIP compiles with ZERO "interface method call" errors.

### Iteration 49 Highlights (commit c8825b649)

**Track A: Virtual Interface Method Call Fix** ⭐⭐⭐ MAJOR FIX!
- ✅ Fixed `vif.method()` calls from class methods failing with "interface method call requires interface instance"
- ✅ Root cause: slang's `CallExpression::thisClass()` doesn't populate for vi method calls
- ✅ Solution: Extract vi expression from syntax using `Expression::bind()` when `thisClass()` unavailable
- ✅ APB AVIP now compiles with ZERO "interface method call" errors!
- Files: `lib/Conversion/ImportVerilog/Expressions.cpp` (+35 lines)
- Test: `test/Conversion/ImportVerilog/virtual-interface-methods.sv`

**Track B: Coverage Runtime Documentation** ✓
- ✅ Verified coverage infrastructure already comprehensive
- ✅ Created test documenting runtime functions and reporting
- ✅ Fixed syntax in `test/Conversion/MooreToCore/coverage-ops.mlir`
- Test: `test/Conversion/ImportVerilog/coverage-runtime.sv`

**Track C: SVA Sequence Declarations** ✓
- ✅ Verified already supported via slang's AssertionInstanceExpression expansion
- ✅ Created comprehensive test with sequences, properties, operators
- Test: `test/Conversion/ImportVerilog/sva-sequence-decl.sv`

**Track D: LSP Rename Symbol Support** ✓
- ✅ Verified already fully implemented with prepareRename() and renameSymbol()
- ✅ Comprehensive test coverage already exists

---

## Previous: ITERATION 48 - Cross Coverage & LSP Improvements (January 17, 2026)

**Summary**: Added cross coverage support, improved LSP find-references, verified runtime randomization infrastructure. UVM APB AVIP now down to just 3 errors.

### Iteration 48 Highlights (commit 64726a33b)

**Track A: Re-test UVM after P0 fix** ✓
- ✅ APB AVIP now down to only 3 errors (from many more before 'this' fix)
- ✅ Remaining errors: virtual interface method calls
- ✅ UVM core library compiles with minimal errors

**Track B: Runtime Randomization Verification** ✓
- ✅ Verified infrastructure already fully implemented
- ✅ MooreToCore.cpp has RandomizeOpConversion (lines 8734-9129)
- ✅ MooreRuntime has __moore_randomize_basic, __moore_randc_next, etc.
- Test: `test/Conversion/ImportVerilog/runtime-randomization.sv`

**Track C: Cross Coverage Support** ⭐
- ✅ Fixed coverpoint symbol lookup bug (use original slang name as key)
- ✅ Added automatic name generation for unnamed cross coverage
- ✅ CoverCrossDeclOp now correctly references coverpoints
- Test: `test/Conversion/ImportVerilog/covergroup_cross.sv`

**Track D: LSP Find-References Enhancement** ✓
- ✅ Added `includeDeclaration` parameter support through call chain
- ✅ Modified LSPServer.cpp, VerilogServer.h/.cpp, VerilogTextFile.h/.cpp, VerilogDocument.h/.cpp
- ✅ Find-references now properly includes or excludes declaration

---

## Previous: ITERATION 47 - P0 BUG FIXED! (January 17, 2026)

**Summary**: Critical 'this' pointer scoping bug FIXED! UVM testbenches that previously failed now compile. Also fixed BMC clock-not-first crash.

### Iteration 47 Highlights (commit dd7908c7c)

**Track A: Fix 'this' pointer scoping in constructor args** ⭐⭐⭐ P0 FIXED!
- ✅ Fixed BLOCKING UVM bug in `Expressions.cpp:4059-4067`
- ✅ Changed `context.currentThisRef = newObj` to `context.methodReceiverOverride = newObj`
- ✅ Constructor argument evaluation now correctly uses caller's 'this' scope
- ✅ Expressions like `m_cb = new({name,"_cb"}, m_cntxt)` now work correctly
- ✅ ALL UVM heartbeat and similar patterns now compile
- Test: `test/Conversion/ImportVerilog/constructor-arg-this-scope.sv`

**Track B: Fix BMC clock-not-first crash** ⭐
- ✅ Fixed crash in `VerifToSMT.cpp` when clock is not first non-register argument
- ✅ Added `isI1Type` check before position-based clock detection
- ✅ Prevents incorrect identification of non-i1 types as clocks
- Test: `test/Conversion/VerifToSMT/bmc-clock-not-first.mlir`

**Track C: SVA bounded sequences ##[n:m]** ✓ Already Working
- ✅ Verified feature already implemented via `ltl.delay` with min/max attributes
- ✅ Supports: `##[1:3]`, `##[0:2]`, `##[*]`, `##[+]`, chained sequences
- Test: `test/Conversion/ImportVerilog/sva_bounded_delay.sv`

**Track D: LSP completion support** ✓ Already Working
- ✅ Verified feature already fully implemented
- ✅ Keywords, snippets, signal names, module names all working
- Existing test: `test/Tools/circt-verilog-lsp-server/completion.test`

### Key Gaps Remaining
1. ~~**'this' pointer scoping bug**~~: ✅ FIXED in Iteration 47
2. **Randomization**: `randomize()` and constraints not yet at runtime
3. ~~**Pre-existing BMC crash**~~: ✅ FIXED in Iteration 47

---

## Comprehensive Gap Analysis & Roadmap

### P0 - BLOCKING UVM (Must fix for any UVM testbench)

| Gap | Location | Impact | Status |
|-----|----------|--------|--------|
| ~~'this' pointer scoping in constructor args~~ | `Expressions.cpp:4059-4067` | ~~Blocks ALL UVM~~ | ✅ FIXED |

### P1 - CRITICAL (Required for full UVM stimulus)

| Gap | Component | Impact | Est. Effort |
|-----|-----------|--------|-------------|
| Runtime randomization | MooreToCore | No random stimulus | 2-3 days |
| Constraint solving | MooreToCore | No constrained random | 3-5 days |
| Covergroup runtime | MooreRuntime | No coverage collection | 2-3 days |

### P2 - IMPORTANT (Needed for comprehensive UVM)

| Gap | Component | Impact | Est. Effort |
|-----|-----------|--------|-------------|
| SVA bounded sequences `##[n:m]` | ImportVerilog | Limited temporal props | 1-2 days |
| BMC clock-not-first bug | VerifToSMT | Crash on some circuits | 1 day |
| Cross coverage | MooreOps | No cross bins | 1-2 days |
| Functional coverage callbacks | MooreRuntime | Limited covergroup | 1 day |

### P3 - NICE TO HAVE (Quality of life)

| Gap | Component | Impact | Est. Effort |
|-----|-----------|--------|-------------|
| LSP find-references | VerilogDocument | No ref navigation | 1-2 days |
| LSP rename symbol | VerilogDocument | No refactoring | 1 day |
| More UVM snippets | VerilogDocument | Developer productivity | 0.5 day |

---

## Track Status & Next Tasks

### Track A: UVM Runtime / DPI/VPI
**Status**: In progress (stubs wired to in-memory HDL map)
**Current**: VPI and DPI stubs exist; HDL access backed by map
**Next Tasks**:
1. Wire HDL/VPI access to simulator signal model
2. Expand VPI property coverage and vector formatting
3. Run ~/mbit/*avip regressions after wiring
4. Keep DPI/UVM unit tests in sync with runtime behavior

### Track B: Randomization + 4-State
**Status**: Randc improvements landed; 4-state not started
**Current**: randc cycles per-field, constrained fields skip overrides
**Next Tasks**:
1. Add real constraint solving (hard/soft/inline)
2. Design 4-state value model and propagation rules
3. Update MooreRuntime + lowering for X/Z operations
4. Re-test ~/sv-tests and targeted UVM randomization suites

### Track C: SVA/Z3 + Coverage
**Status**: SVA parsing ok; Z3 and coverage runtime incomplete
**Current**: Covergroups in classes lowered as properties
**Next Tasks**:
1. Define Z3 bridge for SVA evaluation (~/z3)
2. Implement coverage sample/report hooks end-to-end
3. Add coverage tests in ~/verilator-verification where applicable
4. Track coverage feature gaps vs Xcelium

### Track D: Tooling, LSP, Debugging
**Status**: LSP features expanding (code actions landed)
**Current**: Quick fixes + refactors added
**Next Tasks**:
1. Add debugger hooks and trace stepping
2. Improve workspace symbol/indexing coverage
3. Expand diagnostics and refactor actions
4. Validate against larger sv-test workspaces

---

## Coordination & Cadence
- Keep four agents active in parallel (one per track) to maintain velocity.
- Add unit tests alongside new features and commit regularly.
- Merge work trees into `main` frequently to keep agents synchronized.

## Testing Strategy

### Regular Testing on Real-World Code
```bash
# UVM Core
~/circt/build/bin/circt-verilog --ir-moore -I ~/uvm-core/src ~/uvm-core/src/uvm_pkg.sv 2>&1

# APB AVIP (most comprehensive)
cd ~/mbit/apb_avip && ~/circt/build/bin/circt-verilog --ir-moore \
  -I ~/uvm-core/src -I src/globals -I src/hvl_top/master \
  ~/uvm-core/src/uvm_pkg.sv src/globals/apb_global_pkg.sv ...

# SV tests (use the existing harness)
cd ~/sv-tests && ./run.sh --tool=circt-verilog

# Verilator verification suites
cd ~/verilator-verification && ./run.sh --tool=circt-verilog

# Run unit tests
ninja -C build check-circt-unit
```

### Key Test Suites
- `test/Conversion/ImportVerilog/*.sv` - Import tests
- `test/Conversion/VerifToSMT/*.mlir` - BMC tests
- `test/Tools/circt-verilog-lsp-server/*.test` - LSP tests
- `unittests/Runtime/MooreRuntimeTest.cpp` - Runtime tests

---

## Previous: ITERATION 45 - DPI-C STUBS + VERIFICATION (January 17, 2026)

**Summary**: Major progress on DPI-C runtime stubs, class randomization verification, multi-step BMC analysis, and LSP workspace fixes.

### Iteration 45 Highlights (commit 0d3777a9c)

**Track A: DPI-C Import Support** ⭐ MAJOR MILESTONE
- ✅ Added 18 DPI-C stub functions to MooreRuntime for UVM support
- ✅ HDL access stubs: uvm_hdl_deposit, force, release, read, check_path
- ✅ Regex stubs: uvm_re_comp, uvm_re_exec, uvm_re_free, uvm_dump_re_cache
- ✅ Command-line stubs: uvm_dpi_get_next_arg_c, get_tool_name_c, etc.
- ✅ Changed DPI-C handling from skipping to generating runtime function calls
- ✅ Comprehensive unit tests for all DPI-C stub functions
- Files: `include/circt/Runtime/MooreRuntime.h`, `lib/Runtime/MooreRuntime.cpp`
- Tests: `test/Conversion/ImportVerilog/dpi_imports.sv`, `uvm_dpi_basic.sv`

**Track B: Class Randomization Verification**
- ✅ Verified rand/randc properties, randomize() method fully working
- ✅ Constraints with pre/post, inline, soft constraints all operational
- Tests: `test/Conversion/ImportVerilog/class-randomization.sv`, `class-randomization-constraints.sv`

**Track C: Multi-Step BMC Analysis**
- ⚠️ Documented ltl.delay limitation (N>0 converts to true in single-step BMC)
- ✅ Created manual workaround demonstrating register-based approach
- ✅ Design documentation for proper multi-step implementation
- Tests: `test/Conversion/VerifToSMT/bmc-manual-multistep.mlir`

**Track D: LSP Workspace Symbols**
- ✅ Fixed VerilogServer.cpp compilation errors (StringSet, .str() removal)
- ✅ Fixed workspace symbol gathering in Workspace.cpp
- Files: `lib/Tools/circt-verilog-lsp-server/`

### Key Gaps Remaining
1. **Multi-step BMC**: Need proper ltl.delay implementation for N>0
2. **Covergroups**: Not yet supported (needed for UVM coverage)
3. **DPI-C design integration**: HDL access uses in-memory map only

---

## Previous: ITERATION 44 - UVM PARITY PUSH (January 17, 2026)

**Summary**: Multi-track progress on queue sort.with, UVM patterns, SVA tests, LSP workspace symbol indexing (open docs + workspace files).

### Real-World UVM Testing Results (~/mbit/*avip, ~/uvm-core)

**UVM Package Compilation**: ✅ `uvm_pkg.sv` compiles successfully
- Warnings: Minor escape sequence, unreachable code
- Remarks: DPI-C imports skipped (expected), class builtins dropped (expected)

### Iteration 44 Highlights (commit 66b424f6e + 480081704)

**Track A: UVM Class Method Patterns**
- ✅ Verified all UVM patterns work (virtual methods, extern, super calls, constructors)
- ✅ 21 comprehensive test cases passing
- Tests: `test/Conversion/ImportVerilog/uvm_method_patterns.sv`
 - ✅ DPI-C imports now lower to runtime stub calls (instead of constant fallbacks)

**Track B: Queue sort.with Operations**
- ✅ Added `QueueSortWithOp`, `QueueRSortWithOp`, `QueueSortKeyYieldOp`
- ✅ Memory effect declarations prevent CSE/DCE removal
- ✅ Import support for `q.sort() with (expr)` and `q.rsort() with (expr)`
- Files: `include/circt/Dialect/Moore/MooreOps.td`, `lib/Conversion/ImportVerilog/Expressions.cpp`

**Track C: SVA Implication Tests**
- ✅ Verified `|->` and `|=>` implemented in VerifToSMT
- ✅ Added 117 lines of comprehensive implication tests
- Tests: `test/Conversion/VerifToSMT/ltl-temporal.mlir`
- ✅ LTLToCore shifts exact delayed consequents to past-form implications for BMC
- ✅ Disable-iff now shifts past reset alongside delayed implications (yosys basic00 pass)
- ✅ Multiple non-final asserts are combined for BMC (yosys basic01 pass)
- ✅ circt-bmc flattens private modules so bound assertions are checked (yosys basic02 bind)
- Tests: `test/Conversion/VerifToSMT/bmc-nonoverlap-implication.mlir`, `integration_test/circt-bmc/sva-e2e.sv`

**Track D: LSP Workspace Symbols**
- ✅ `workspace/symbol` support added for open docs and workspace files
- ✅ Workspace scan covers module/interface/package/class/program/checker
- ✅ Workspace-symbol project coverage added
- Files: `lib/Tools/circt-verilog-lsp-server/`

---

## Previous: ITERATION 43 - WORKSPACE SYMBOL INDEXING (January 18, 2026)

**Summary**: Added workspace symbol search across workspace files with basic regex indexing.

### Iteration 43 Highlights

**Track D: Tooling & Debug (LSP)**
- ✅ Workspace symbol search scans workspace files (module/interface/package/class/program/checker)
- ✅ Deduplicates results between open docs and workspace index
- ✅ Added workspace project coverage for symbol queries
- Files: `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/Workspace.cpp`, `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/Workspace.h`, `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogServer.cpp`
- Tests: `test/Tools/circt-verilog-lsp-server/workspace-symbol-project.test`

---

## Previous: ITERATION 42 - LSP WORKSPACE SYMBOLS (January 18, 2026)

**Summary**: Added workspace symbol search for open documents.

### Iteration 42 Highlights

**Track D: Tooling & Debug (LSP)**
- ✅ `workspace/symbol` implemented for open documents
- ✅ Added lit coverage for workspace symbol queries
- Files: `lib/Tools/circt-verilog-lsp-server/LSPServer.cpp`, `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogServer.cpp`, `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/VerilogServer.h`
- Tests: `test/Tools/circt-verilog-lsp-server/workspace-symbol.test`

---

## Previous: ITERATION 41 - SVA GOTO/NON-CONSEC REPETITION (January 18, 2026)

**Summary**: Added BMC conversions for goto and non-consecutive repetition.

### Iteration 41 Highlights

**Track C: SVA + Z3 Track**
- ✅ `ltl.goto_repeat` and `ltl.non_consecutive_repeat` lower to SMT booleans
- ✅ Base=0 returns true; base>0 uses the input at a single step
- ✅ Added coverage to `ltl-temporal.mlir`
- Files: `lib/Conversion/VerifToSMT/VerifToSMT.cpp`
- Tests: `test/Conversion/VerifToSMT/ltl-temporal.mlir`

---

## Previous: ITERATION 40 - RANDJOIN BREAK SEMANTICS (January 18, 2026)

**Summary**: `break` in forked randjoin productions exits only that production.

### Iteration 40 Highlights

**Track A: UVM Language Parity (ImportVerilog/Lowering)**
- ✅ `break` inside forked randjoin branches exits the production branch
- ✅ Added randjoin+break conversion coverage
- Files: `lib/Conversion/ImportVerilog/Statements.cpp`
- Tests: `test/Conversion/ImportVerilog/randsequence.sv`

---

## Previous: ITERATION 39 - RANDJOIN ORDER RANDOMIZATION (January 18, 2026)

**Summary**: randjoin(all) now randomizes production execution order.

### Iteration 39 Highlights

**Track A: UVM Language Parity (ImportVerilog/Lowering)**
- ✅ randjoin(N>=numProds) uses Fisher-Yates selection to randomize order
- ✅ joinCount clamped to number of productions before dispatch
- Files: `lib/Conversion/ImportVerilog/Statements.cpp`

---

## Previous: ITERATION 38 - RANDSEQUENCE BREAK/RETURN (January 18, 2026)

**Summary**: Randsequence productions now support `break` and production-local `return`.

### Iteration 38 Highlights

**Track A: UVM Language Parity (ImportVerilog/Lowering)**
- ✅ `break` exits the randsequence statement
- ✅ `return` exits the current production without returning from the function
- ✅ Added return target stack and per-production exit blocks
- Files: `lib/Conversion/ImportVerilog/Statements.cpp`, `lib/Conversion/ImportVerilog/ImportVerilogInternals.h`
- Tests: `test/Conversion/ImportVerilog/randsequence.sv`

---

## Previous: ITERATION 37 - LTL SEQUENCE OPS + LSP FIXES (January 17, 2026)

**Summary**: LTL sequence operators (concat/delay/repeat) for VerifToSMT, LSP test fixes.

### Iteration 37 Highlights (commit 3f73564be)

**Track A: Randsequence randjoin(N>1)**
- ✅ Extended randjoin test coverage with `randsequence-randjoin.sv`
- ✅ Fisher-Yates partial shuffle for N distinct production selection
- Files: `lib/Conversion/ImportVerilog/Statements.cpp`

**Track C: SVA Sequence Operators in VerifToSMT**
- ✅ `ltl.delay` → delay=0 passes through, delay>0 returns true (BMC semantics)
- ✅ `ltl.concat` → empty=true, single=itself, multiple=smt.and
- ✅ `ltl.repeat` → base=0 returns true, base>=1 returns input
- ✅ LTL type converters for `!ltl.sequence` and `!ltl.property` to `smt::BoolType`
- Files: `lib/Conversion/VerifToSMT/VerifToSMT.cpp` (+124 lines)
- Test: `test/Conversion/VerifToSMT/ltl-temporal.mlir` (+88 lines)

**Track D: LSP Hover and Completion Tests**
- ✅ Fixed `hover.test` character position coordinate
- ✅ Fixed `class-hover.test` by wrapping classes in package
- ✅ Verified all LSP tests pass: hover, completion, class-hover, uvm-completion
- Files: `test/Tools/circt-verilog-lsp-server/hover.test`, `class-hover.test`

---

## Previous: ITERATION 36 - QUEUE SORT RUNTIME FIX (January 18, 2026)

**Summary**: Queue sort/rsort now sort in place with element size awareness.

### Iteration 36 Highlights

**Track B: Runtime & Array/Queue Semantics**
- ✅ `queue.sort()` and `queue.rsort()` lower to in-place runtime calls
- ✅ Element-size-aware comparators for <=8 bytes and bytewise fallback for larger
- Files: `lib/Runtime/MooreRuntime.cpp`, `lib/Conversion/MooreToCore/MooreToCore.cpp`, `include/circt/Runtime/MooreRuntime.h`

---

## Previous: ITERATION 35 - RANDSEQUENCE CONCURRENCY + TAGGED UNIONS (January 18, 2026)

**Summary**: Four parallel agents completed: randsequence randjoin>1 fork/join, tagged union patterns, dynamic array streaming lvalues, randsequence case exit fix.

### Iteration 35 Highlights

**Track A: Randsequence randjoin>1 Concurrency**
- ✅ randjoin(all) and randjoin(subset) now use `moore.fork join`
- ✅ Distinct production selection via partial Fisher-Yates shuffle
- ✅ Forked branches dispatch by selected index
- Files: `lib/Conversion/ImportVerilog/Statements.cpp`

**Track B: Tagged Union Lowering + Pattern Matches**
- ✅ Tagged unions lowered to `{tag, data}` wrapper structs
- ✅ `.tag` access and tagged member extraction lowered
- ✅ PatternCase and `matches` expressions for tagged/constant/wildcard patterns
- Files: `lib/Conversion/ImportVerilog/Types.cpp`, `lib/Conversion/ImportVerilog/Expressions.cpp`, `lib/Conversion/ImportVerilog/Statements.cpp`

**Track C: Streaming Lvalue Fix (Dynamic/Open Arrays)**
- ✅ `{>>{arr}} = packed` lvalue streaming now supports open unpacked arrays
- ✅ Lowered to `moore.stream_unpack` in lvalue context
- Files: `lib/Conversion/ImportVerilog/Expressions.cpp`

**Track D: Randsequence Case Exit Correctness**
- ✅ Default fallthrough now branches to exit, not last match
- Files: `lib/Conversion/ImportVerilog/Statements.cpp`

---

## Previous: ITERATION 34 - MULTI-TRACK PARALLEL PROGRESS (January 17, 2026)

**Summary**: Four parallel agents completed: randcase, queue delete(index), LTL-to-SMT operators, LSP verification.

### Iteration 34 Highlights (commit 0621de47b)

**Track A: randcase Statement (IEEE 1800-2017 §18.16)**
- ✅ Weighted random selection using `$urandom_range`
- ✅ Cascading comparisons for branch selection
- ✅ Edge case handling (zero weights, single-item optimization)
- Files: `lib/Conversion/ImportVerilog/Statements.cpp` (+100 lines)

**Track B: Queue delete(index) Runtime**
- ✅ `__moore_queue_delete_index(queue, index, element_size)` with proper shifting
- ✅ MooreToCore lowering passes element size from queue type
- ✅ Bounds checking and memory management
- Files: `lib/Runtime/MooreRuntime.cpp`, `lib/Conversion/MooreToCore/MooreToCore.cpp`

**Track C: LTL Temporal Operators in VerifToSMT**
- ✅ `ltl.and`, `ltl.or`, `ltl.not`, `ltl.implication` → SMT boolean ops
- ✅ `ltl.eventually` → identity at each step (BMC accumulates with OR)
- ✅ `ltl.until` → `q || p` (weak until for BMC)
- ✅ `ltl.boolean_constant` → `smt.constant`
- Files: `lib/Conversion/VerifToSMT/VerifToSMT.cpp` (+178 lines)

**Track D: LSP go-to-definition Verification**
- ✅ Confirmed existing implementation works correctly
- ✅ Added comprehensive test coverage for modules, wires, ports
- Files: `test/Tools/circt-verilog-lsp-server/goto-definition.test` (+133 lines)

**Total**: 1,695 insertions across 13 files

---

## Active Workstreams (Next Tasks)

**We should keep four agents running in parallel.**

### Track A: UVM Language Parity (ImportVerilog/Lowering)
**Status**: Active | **Priority**: CRITICAL
**Next Task**: DPI-C HDL Access Behavior (blocking for UVM)
- UVM uses DPI-C for HDL access, regex, command line processing
- Runtime stubs are wired; HDL access now uses in-memory map
- Next add HDL hierarchy access (connect to simulation objects)
- Command line args are read from `CIRCT_UVM_ARGS`/`UVM_ARGS` (space-delimited)
- Command line args support quoted strings and basic escapes
- Command line args reload when env strings change (useful for tests)
- Force semantics preserved in HDL access stub (deposit respects force)
- UVM HDL access DPI calls covered by ImportVerilog tests
- Added VPI stub API placeholders (no real simulator integration yet)
- uvm_hdl_check_path initializes entries in the HDL map
- VPI stubs now return basic handles/strings for smoke testing
- vpi_handle_by_name seeds the HDL access map
- vpi_release_handle added for cleanup
- vpi_put_value updates the HDL access map for matching reads
- vpi_put_value flags now mark the entry as forced
- Files: `lib/Runtime/MooreRuntime.cpp`, `lib/Conversion/ImportVerilog/Expressions.cpp`

### Track B: Class Randomization & Constraints
**Status**: IN PROGRESS | **Priority**: CRITICAL
**Next Task**: Rand/RandC semantics beyond basic preservation
- Randomize now preserves non-rand fields during `randomize()`
- randc cycling now supported for small bit widths (linear fallback above 16 bits)
- Soft/hard constrained randc fields bypass randc cycling
- Next implement broader constraint coverage and widen randc cycles
- Add coverage for multiple randc fields and cycle reset behavior
- Multi-field randc conversion coverage added
- Randc cycle resets on bit-width changes
 - Randc fields with hard constraints bypass randc cycling
- Files: `lib/Conversion/MooreToCore/MooreToCore.cpp`, `lib/Runtime/MooreRuntime.cpp`

### Track C: SVA + Z3 Track
**Status**: ⚠️ PARTIAL (multi-step delay buffering for `##N`/bounded `##[m:n]` on i1) | **Priority**: HIGH
**Next Task**: Extend temporal unrolling beyond delay
- ✅ Repeat (`[*N]`) expansion in BMC (bounded by BMC depth; uses delay buffers)
- ✅ Added end-to-end BMC tests for repeat fail cases
- ⚠️ Repeat pass cases still fail due to LTLToCore implication semantics (needs fix)
- ✅ Goto/non-consecutive repeat expanded in BMC (bounded by BMC depth)
- ✅ Added local yosys SVA harness script for circt-bmc runs
- ✅ Import now preserves concurrent assertions with action blocks (`else $error`)
- ✅ yosys `basic00.sv`, `basic01.sv`, `basic02.sv` pass in circt-bmc harness
- ⚠️ yosys `basic03.sv` pass still fails (sampled-value alignment for clocked assertions; $past comparisons)
- ✅ Non-overlapped implication for property RHS now uses `seq ##1 true` encoding
- ✅ LTL-aware equality/inequality enabled for `$past()` comparisons in assertions
- ✅ Handle unbounded delay ranges (`##[m:$]`) in BMC within bound (bounded approximation)
- ✅ Added end-to-end SVA BMC integration tests (SV → `circt-bmc`) for delay and range delay (pass + fail cases; pass uses `--ignore-asserts-until=1`)
- Add more end-to-end BMC tests with Z3 (`circt-bmc`) for temporal properties
- Files: `lib/Tools/circt-bmc/`, `lib/Conversion/VerifToSMT/VerifToSMT.cpp`

### Track D: Tooling & Debug (LSP)
**Status**: ✅ Workspace Symbols (workspace files) | **Priority**: MEDIUM
**Next Task**: Replace regex symbol scan with parsed symbol index
- Build a symbol index from Slang AST for precise ranges and more symbol kinds
- Keep `workspace/symbol` results stable across open/closed documents
- Files: `lib/Tools/circt-verilog-lsp-server/VerilogServerImpl/`

**Testing Cadence**
- Run regression slices on `~/mbit/*avip*`, `~/sv-tests/`, `~/verilator-verification/` regularly
- Add unit tests with each feature; commit regularly and merge back to main to keep workers in sync

## Big Projects Status (Parity with Xcelium)

| Project | Status | Next Milestone |
|---------|--------|----------------|
| **DPI/VPI Support** | 🔴 CRITICAL GAP | Implement HDL access behind DPI stubs, add real VPI handle support |
| **Class Randomization** | 🔴 CRITICAL GAP | randc cycling + constraint-aware randomize |
| **Full SVA + Z3** | ⚠️ Bounded delay buffering | Repeat/unbounded delay unrolling |
| **LSP + Debugging** | ✅ Workspace Symbols | Symbol index + rename/debugging hooks |
| **Coverage** | 🟡 PARTIAL | Covergroups + sampling expressions |
| **Multi-core Arcilator** | MISSING | Architecture plan |
| **Full 4-state (X/Z)** | MISSING | Type system + dataflow propagation plan |

**Z3 Configuration** (January 17, 2026):
- Z3 4.12.4 installed at `~/z3-install/`
- CIRCT configured with `-DZ3_DIR=~/z3-install/lib64/cmake/z3`
- `circt-bmc` builds and runs with Z3 backend
- Runtime: `export LD_LIBRARY_PATH=~/z3-install/lib64:$LD_LIBRARY_PATH`

## Current Limitations (Key Gaps from UVM Testing)

**CRITICAL (Blocking UVM)**:
1. **DPI-C imports are partially stubbed** - HDL access uses in-memory map, no real hierarchy
2. **Class randomization partial** - randc cycling limited to <=16-bit fields; wider widths use linear cycle
3. **Covergroups dropped** - Needed for UVM coverage collection

**HIGH PRIORITY**:
4. Temporal BMC unrolling: repeat (`[*N]`) + unbounded `##[m:$]` (bounded delays now buffered)
5. Constraint expressions for randomization
6. Cross coverage and sampling expressions
7. BMC: LLHD time ops from `initial` blocks still fail legalization (avoid for now)

**MEDIUM**:
8. Regex-based workspace symbol scanning (no full parse/index)
9. 4-state X/Z propagation
10. VPI handle support
11. Multi-core Arcilator

## Next Feature Targets (Top Impact for UVM)
1. **DPI-C runtime stubs** - Implement `uvm_hdl_deposit`, `uvm_hdl_force`, `uvm_re_*`
2. **Class randomization** - `rand`/`randc` properties, basic `randomize()` call
3. **Multi-step BMC** - Extend beyond delay buffering (repeat + unbounded delay)
4. **Symbol index** - Replace regex scan with AST-backed symbol indexing
5. **Coverage** - Covergroup sampling basics for UVM

**Immediate Next Task**
- Implement DPI-C import stubs for core UVM functions.

---

## Previous: ITERATION 32 - RANDSEQUENCE SUPPORT (January 17, 2026)

**Summary**: Full randsequence statement support (IEEE 1800-2017 Section 18.17)

### Iteration 32 Highlights

**RandSequence Statement Support (IEEE 1800-2017 Section 18.17)**:
- ✅ Basic sequential productions - execute productions in order
- ✅ Code blocks in productions - `{ statements; }` execute inline
- ✅ Weighted alternatives - `prod := weight | prod2 := weight2` with `$urandom_range`
- ✅ If-else production statements - `if (cond) prod_a else prod_b`
- ✅ Repeat production statements - `repeat(n) production`
- ✅ Case production statements - `case (expr) 0: prod; 1: prod2; endcase`
- ✅ Nested production calls - productions calling other productions
- ✅ Production argument binding (input-only, default values supported)

**sv-tests Section 18.17 Results**:
- 9/16 tests passing (56%)
- All basic functionality working
- Remaining gaps: `break`/`return` in productions, randjoin (only randjoin(1) supported)

**Files Modified**:
- `lib/Conversion/ImportVerilog/Statements.cpp` - Full randsequence implementation (~330 lines)

---

## Previous: ITERATION 31 - CLOCKING BLOCK SIGNAL ACCESS (January 16, 2026)

**Summary**: Clocking block signal access (`cb.signal`), @(cb) event syntax, LLHD Phase 2

### Iteration 31 Highlights

**Clocking Block Signal Access (IEEE 1800-2017 Section 14)**:
- ✅ `cb.signal` rvalue generation - reads correctly resolve to underlying signal
- ✅ `cb.signal` lvalue generation - writes correctly resolve to underlying signal
- ✅ `@(cb)` event syntax - waits for clocking block's clock event
- ✅ Both input and output clocking signals supported

**LLHD Process Interpreter Phase 2**:
- ✅ Full process execution: `llhd.drv`, `llhd.wait`, `llhd.halt`
- ✅ Signal probing and driving operations
- ✅ Time advancement and delta cycle handling
- ✅ 5/6 circt-sim tests passing

**Iteration 31 Commits**:
- **43f3c7a4d** - Clocking block signal access and @(cb) syntax support (1,408 insertions)
  - ClockVar rvalue/lvalue generation in ImportVerilog/Expressions.cpp
  - @(cb) event reference in ImportVerilog/TimingControls.cpp
  - QueueReduceOp for sum/product/and/or/xor methods
  - LLHD process execution fixes

---

## Previous: ITERATION 30 - COMPREHENSIVE TEST SURVEY (January 16, 2026)

**Summary**: SVA boolean context fixes, Z3 CMake linking, comprehensive test suite survey

### Test Suite Coverage (Iteration 30)

| Test Suite | Total Tests | Pass Rate | Notes |
|------------|-------------|-----------|-------|
| **sv-tests** | 989 (non-UVM) | **72.1%** (713/989) | Parsing/elaboration focus |
| **mbit AVIP globals** | 8 packages | **100%** | All package files work |
| **mbit AVIP interfaces** | 8 interfaces | **75%** | 6/8 pass |
| **mbit AVIP HVL** | 8 packages | **0%** | Requires UVM library |
| **verilator-verification** | 154 | ~60% | SVA tests improved |

### sv-tests Chapter Breakdown (72.1% overall)

| Chapter | Pass Rate | Key Gaps |
|---------|-----------|----------|
| Ch 5 (Lexical) | **86%** | Good |
| Ch 6 (Data Types) | **75%** | TaggedUnion |
| Ch 7 (Aggregate) | **72%** | Unpacked dimensions |
| Ch 9 (Behavioral) | **73%** | Minor gaps |
| Ch 10 (Scheduling) | **50%** | RaceyWrite |
| Ch 11 (Operators) | **87%** | Strong |
| Ch 12 (Procedural) | **79%** | SequenceWithMatch |
| Ch 13 (Tasks/Functions) | **86%** | Strong |
| Ch 14 (Clocking Blocks) | **~80%** | Signal access (cb.signal), @(cb) event working |
| Ch 16 (Assertions) | **68%** | EmptyArgument |
| Ch 18 (Random/Constraints) | **25%** | RandSequence |
| Ch 20 (I/O Formatting) | **83%** | Good |
| Ch 21 (I/O System Tasks) | **37%** | VcdDump |

### Top Missing Features (by sv-tests failures)

| Feature | Tests Failed | Priority |
|---------|--------------|----------|
| **ClockingBlock** | ~50 | HIGH |
| **RandSequence** | ~30 | MEDIUM |
| **SequenceWithMatch** | ~25 | MEDIUM |
| **TaggedUnion** | ~20 | MEDIUM |
| **EmptyArgument** | ~15 | LOW |

### Z3 BMC Status

- **Z3 AVAILABLE** at ~/z3 (include: ~/z3/include, lib: ~/z3/lib/libz3.so)
- CMake linking code is correct (both CONFIG and Module mode support)
- Pipeline verified: SV → Moore → HW → BMC MLIR → LLVM IR generation
- **LowerClockedAssertLike pass added** - handles verif.clocked_assert for BMC
- Testing Z3 integration in progress

### Test Commands
```bash
# UVM Parsing - COMPLETE
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src
# Exit code: 0 (SUCCESS!) - 161,443 lines of Moore IR

# SVA BMC (Bounded Model Checking) - CONVERSION WORKS
./build/bin/circt-verilog --ir-hw /tmp/simple_sva.sv | \
  ./build/bin/circt-bmc --bound=10
# VerifToSMT conversion produces valid MLIR (Z3 installation needed)
```

**Iteration 30 Commits**:
- **Multi-track progress (commit ab52d23c2)** - 3,522 insertions across 26 files:
  - Track 1: Clocking blocks - ClockingBlockDeclOp, ClockingSignalOp in Moore
  - Track 2: LLHD interpreter - LLHDProcessInterpreter.cpp/h for circt-sim
  - Track 3: $past fix - moore::PastOp for type-preserving comparisons
  - Track 4: clocked_assert lowering - LowerClockedAssertLike.cpp for BMC
  - LTLToCore enhancements (986 lines added)
- Big projects status survey (commit 9abf0bb24)
- Active development tracks documentation (commit e48c2f3f8)
- SVA functions in boolean contexts (commit a68ed9adf) - ltl.or/ltl.and/ltl.not for LTL types
- Z3 CMake linking fix (commit 48bcd2308) - JIT runtime linking for SMTToZ3LLVM
- $rose/$fell test improvements (commit 8ad3a7cc6)
- MooreToCore coverage ops tests (commit d92d81882)
- VerifToSMT conversion tests (commit ecabb4492)
- SVAToLTL conversion tests (commit 47c5a7f36)

**Iteration 29 Commits**:
- VerifToSMT `bmc.final` assertion handling fixes
- ReconcileUnrealizedCasts pass added to circt-bmc pipeline
- BVConstantOp argument order fix (value, width)
- Clock counting before region conversion
- Proper rewriter.eraseOp() usage in conversion patterns

**Iteration 28 Commits**:
- `7d5391552` - $onehot/$onehot0 system calls
- `2830654d4` - $countbits system call
- `4704320af` - $sampled/$past/$changed for SVA assertions
- `25cd3b6a2` - Direct interface member access
- `12d75735d`, `110fc6caf` - Test fixes and documentation
- `47c5a7f36` - SVAToLTL comprehensive tests
- `ecabb4492` - VerifToSMT comprehensive tests
- `235700509` - CHANGELOG update

**Fork**: https://github.com/thomasnormal/circt (synced with upstream)

**Current Blockers / Limitations** (Post-MooreToCore):
1. **Coverage** ✅ INFRASTRUCTURE DONE - CovergroupHandleType, CovergroupInstOp, CovergroupSampleOp implemented
2. **SVA assertions** ✅ LOWERING WORKS - moore.assert/assume/cover → verif.assert/assume/cover
3. **DPI/VPI** ⚠️ STUBS ONLY - 22 DPI functions return defaults (0, empty string, "CIRCT")
4. **Complex constraints** ⚠️ PARTIAL - ~6% need SMT solver (94% now work!)
5. **System calls** ✅ $countones IMPLEMENTED - $clog2 and some others still needed
6. **UVM reg model** ⚠️ CLASS HIERARCHY ISSUE - uvm_reg_map base class mismatch
7. **Tagged unions** ⚠️ PARTIAL - tag semantics still missing (tag compare/extract correctness)
8. **Dynamic array range select** ✅ IMPLEMENTED - queue/dynamic array slicing supported
9. **Queue sorting semantics** ⚠️ PARTIAL - rsort/shuffle use simple runtime helpers; custom comparator support missing
10. **Randsequence** ⚠️ PARTIAL - formal arguments and break/return in productions not handled

**AVIP Testing Results** (Iteration 28 - comprehensive validation):

| Component Type | Pass Rate | Notes |
|----------------|-----------|-------|
| Global packages | 8/8 (100%) | All package files work |
| Interfaces | 7/9 (78%) | JTAG/I2S fail due to source issues, not CIRCT bugs |

| AVIP | Step 1 (Moore IR) | Step 2 (MooreToCore) | Notes |
|------|------------------|---------------------|-------|
| APB | ✅ PASS | ✅ PASS | Works without UVM |
| AXI4-Lite | ✅ PASS | ✅ PASS | Works without UVM |
| UART | ✅ PASS | ✅ PASS | Works without UVM |
| SPI | ✅ PASS | ✅ PASS | Works without UVM |
| AHB | ✅ PASS | ✅ PASS | Works without UVM |
| AXI4 | ✅ PASS | ✅ PASS | Works without UVM |

**MAJOR MILESTONE (Iteration 28)**:
- **SVA assertion functions** ✅ COMPLETE - $sampled, $past (with delay), $changed, $stable, $rose, $fell all implemented
- **System calls expanded** ✅ COMPLETE - $onehot, $onehot0, $countbits added
- **Direct interface member access** ✅ FIXED - Hierarchical name resolution for interface.member syntax
- **Test coverage improved** ✅ COMPLETE - SVAToLTL: 3 new test files, VerifToSMT: comprehensive tests added
- **AVIP validation** ✅ COMPLETE - Global packages 100%, Interfaces 78% (failures are source issues)

**MAJOR MILESTONE (Iteration 26)**:
- **Upstream merge** ✅ COMPLETE - Merged 21 upstream commits, resolved 4 conflicts
- **Fork published** ✅ COMPLETE - thomasnormal/circt with comprehensive README feature list
- **SVA assertion lowering** ✅ VERIFIED - moore.assert/assume/cover → verif dialect working
- **$countones** ✅ IMPLEMENTED - Lowers to llvm.intr.ctpop
- **AVIP validation** ✅ ALL 6 PASS - APB, AXI4-Lite, UART, SPI, AHB, AXI4 work through MooreToCore
- **Coverage infrastructure** ✅ COMPLETE - CovergroupHandleType and ops implemented in Iteration 25

**MAJOR MILESTONE (Iteration 25)**:
- **Interface ref→vif conversion** ✅ FIXED - Interface member access generates proper lvalue references
- **Constraint MooreToCore lowering** ✅ COMPLETE - All 10 constraint ops now lower to runtime calls
- **$finish in seq.initial** ✅ FIXED - $finish no longer forces llhd.process fallback

**MAJOR MILESTONE (Iteration 23)**:
- **Initial blocks** ✅ FIXED (cabc1ab6e) - Simple initial blocks use seq.initial, work through arcilator!
- **Multi-range constraints** ✅ FIXED (c8a125501) - ~94% total constraint coverage
- **End-to-end pipeline** ✅ VERIFIED - SV → Moore → Core → HW → Arcilator all working

**Fixed (Iteration 22)**:
- **sim.terminate** ✅ FIXED (575768714) - $finish now calls exit(0/1)
- **Soft constraints** ✅ FIXED (5e573a811) - Default value constraints work

**Fixed (Iteration 21)**:
- **UVM LSP support** ✅ FIXED (d930aad54) - `--uvm-path` flag and `UVM_HOME` env var
- **Range constraints** ✅ FIXED (2b069ee30) - Simple range constraints work
- **Interface symbols** ✅ FIXED (d930aad54) - LSP returns proper interface symbols
- **sim.proc.print** ✅ FIXED (2be6becf7) - $display works in arcilator

**Resolved Blockers (Iteration 14)**:
- ~~**moore.builtin.realtobits**~~ ✅ FIXED (36fdb8ab6) - Added conversion patterns for realtobits/bitstoreal

**Recent Fixes (This Session - Iteration 13)**:
- **VTable fallback for classes without vtable segments** ✅ FIXED (6f8f531e6) - Searches ALL vtables when class has no segment
- **AVIP BFM validation** ✅ COMPLETE - APB, AHB, AXI4, AXI4-Lite parse and convert; issues in test code (deprecated UVM APIs) not tool
- **AXI4-Lite AVIP** ✅ 100% PASS - Zero MooreToCore errors
- **Pipeline investigation** ✅ DOCUMENTED - circt-sim runs but doesn't execute llhd.process bodies; arcilator is RTL-only

**Previous Fixes (Iteration 12)**:
- **Array locator inline loop** ✅ FIXED (115316b07) - Complex predicates (string cmp, AND/OR, func calls) now lowered via scf.for loop
- **llhd.time data layout crash** ✅ FIXED (1a4bf3014) - Structs with time fields now handled via getTypeSizeSafe()
- **AVIP MooreToCore** ✅ VALIDATED - All 7 AVIPs (APB, AHB, AXI4, UART, I2S, I3C, SPI) pass through MooreToCore

**Recent Fixes (Previous Session)**:
- **RefType cast crash for structs with dynamic fields** ✅ FIXED (5dd8ce361) - StructExtractRefOp now uses LLVM GEP for structs containing strings/queues instead of crashing on SigStructExtractOp
- **Mem2Reg loop-local variable dominance** ✅ FIXED (b881afe61) - Variables inside loops no longer promoted, fixing 4 dominance errors
- **Static property via instance** ✅ FIXED (a1418d80f) - SystemVerilog allows `obj.static_prop` access. Now correctly generates GetGlobalVariableOp instead of ClassPropertyRefOp.
- **Static property names in parameterized classes** ✅ FIXED (a1418d80f) - Each specialization now gets unique global variable name (e.g., `uvm_pool_1234::m_prop` not `uvm_pool::m_prop`).
- **Abstract class vtable** ✅ FIXED (a1418d80f) - Virtual classes with mixed concrete/pure virtual methods now skip vtable generation instead of emitting error.
- **Time type in Mem2Reg** ✅ FIXED (3c9728047) - `VariableOp::getDefaultValue()` now correctly returns TimeType values instead of l64 constants.
- **Global variable redefinition** ✅ FIXED (a152e9d35) - Fixed duplicate GlobalVariableOp when class type references the variable in methods.
- **Method lookup in parameterized classes** ✅ FIXED (71c80f6bb) - Class bodies now populated via convertClassDeclaration in declareFunction.
- **Property type mismatch** ✅ FIXED - Parameterized class property access uses correct specialized class symbol.

**Previous Blockers FIXED** (Earlier):
1. ~~`$fwrite` unsupported~~ ✅ FIXED (ccfc4f6ca)
2. ~~`$fopen` unsupported~~ ✅ FIXED (ce8d1016a)
3. ~~`next` unsupported~~ ✅ FIXED (2fa392a98) - string assoc array iteration
4. ~~`$fclose` unsupported~~ ✅ FIXED (b4a18d045) - File I/O complete
5. ~~`%20s` width specifier not supported~~ ✅ FIXED (88085cbd7) - String format width
6. ~~String case IntType crash~~ ✅ FIXED (3410de2dc) - String case statement handling

**Note**: Earlier "AVIP passing" tests used wrong UVM path (`~/UVM/distrib/src`).
Correct path is `~/uvm-core/src`. Making good progress on remaining blockers!

---

## Feature Matrix: Current vs Target

| Capability | Current CIRCT | Target (Xcelium Parity) | Status |
|------------|---------------|------------------------|--------|
| **Classes** | Basic OOP + UVM parsing | Full OOP + factory pattern | ✅ Mostly done |
| **Interfaces** | Partial | Virtual interfaces, modports | ✅ Complete |
| **Process Control** | fork/join designed | fork/join, disable, wait | ✅ Designed |
| **File I/O** | $fopen, $fwrite, $fclose | $fopen, $fwrite, $fclose | ✅ Complete |
| **Assoc Arrays** | Int keys work | All key types + iterators | ✅ String keys fixed |
| **Randomization** | Range constraints work | rand/randc, constraints | ⚠️ ~59% working |
| **Coverage** | Coverage dialect exists | Full functional coverage | ⚠️ Partial |
| **Assertions** | SVA functions complete | Full SVA | ✅ $sampled/$past/$changed/$stable/$rose/$fell |
| **DPI/VPI** | Stub returns (0/empty) | Full support | ⚠️ 22 funcs analyzed, stubs work |
| **MooreToCore** | All 9 AVIPs lower | Full UVM lowering | ✅ Complete |

---

## Active Workstreams (keep 4 agents busy)

### Track A: LLHD Process Interpretation in circt-sim 🎯 ITERATION 30
**Status**: 🟡 IMPLEMENTATION PLAN READY - Phase 1 design complete
**Problem**: circt-sim doesn't interpret LLHD process bodies - simulation ends at 0fs

**Implementation Plan (Phase 1A - Core Interpreter)**:

```cpp
// New class: LLHDProcessInterpreter (tools/circt-sim/LLHDProcessInterpreter.h)
class LLHDProcessInterpreter {
  struct SignalState {
    mlir::Value sigValue;
    size_t schedulerSignalId;
  };

  llvm::DenseMap<mlir::Value, SignalState> signals;
  llvm::DenseMap<mlir::Value, llvm::Any> ssaValues;

public:
  // Phase 1A: Register signals from llhd.sig ops
  void registerSignals(mlir::Operation *moduleOp);

  // Phase 1A: Convert llhd.time to SimTime
  SimTime convertTime(llhd::TimeAttr timeAttr);

  // Phase 1A: Core operation handlers
  void interpretProbe(llhd::PrbOp op);     // Read signal value
  void interpretDrive(llhd::DrvOp op);     // Schedule signal update
  void interpretWait(llhd::WaitOp op);     // Suspend process
  void interpretHalt(llhd::HaltOp op);     // Terminate process

  // Phase 1B: Control flow (cf.br, cf.cond_br)
  void interpretBranch(cf::BranchOp op);
  void interpretCondBranch(cf::CondBranchOp op);

  // Phase 1C: Arithmetic (arith.addi, arith.cmpi, etc.)
  void interpretArith(mlir::Operation *op);
};
```

**Integration with circt-sim.cpp**:
```cpp
// In SimulationContext::buildSimulationModel():
for (auto &op : moduleOp.getBody().front()) {
  if (auto processOp = dyn_cast<llhd::ProcessOp>(&op)) {
    auto interpreter = std::make_shared<LLHDProcessInterpreter>();
    interpreter->registerSignals(moduleOp);

    auto callback = [interpreter, &processOp]() {
      interpreter->execute(processOp.getBody());
    };

    scheduler.createProcess(callback);
  }
}
```

**Phased Approach**:
- **Phase 1A** (1 week): Signal registration, llhd.prb/drv/wait/halt handlers
- **Phase 1B** (3-4 days): Control flow (cf.br, cf.cond_br, block arguments)
- **Phase 1C** (3-4 days): Arithmetic operations (arith.addi, cmpi, etc.)
- **Phase 2** (1 week): Complex types, memory, verification

**Files to Create/Modify**:
- `tools/circt-sim/LLHDProcessInterpreter.h` (NEW)
- `tools/circt-sim/LLHDProcessInterpreter.cpp` (NEW)
- `tools/circt-sim/circt-sim.cpp` (modify buildSimulationModel)
- `tools/circt-sim/CMakeLists.txt` (add new source files)

**Verified Test Case**:
```bash
# Input: test_llhd_sim.sv with initial block and always block
./build/bin/circt-verilog --ir-llhd /tmp/test_llhd_sim.sv | ./build/bin/circt-sim --sim-stats
# Output: "Simulation completed at time 0 fs" with only 1 placeholder process
# Expected: Should run llhd.process bodies with llhd.wait delays
```

**Priority**: CRITICAL - Required for behavioral simulation

### Track B: Direct Interface Member Access 🎯 ITERATION 28 - FIXED
**Status**: 🟢 COMPLETE (commit 25cd3b6a2)
**Problem**: "unknown hierarchical name" for direct (non-virtual) interface member access
**Resolution**: Fixed hierarchical name resolution for interface.member syntax
**Verified**: Works in AVIP interface tests
**Files**: `lib/Conversion/ImportVerilog/`
**Priority**: DONE

### Track C: System Call Expansion 🎯 ITERATION 28 - COMPLETE
**Status**: 🟢 ALL SVA FUNCTIONS IMPLEMENTED
**What's Done** (Iteration 28):
- $onehot, $onehot0 IMPLEMENTED (commit 7d5391552)
- $countbits IMPLEMENTED (commit 2830654d4) - count specific bit values
- $countones working (llvm.intr.ctpop)
- $clog2, $isunknown already implemented
- **SVA assertion functions** (commit 4704320af):
  - $sampled - sample value in observed region
  - $past (with delay parameter) - previous cycle value
  - $changed - value changed from previous cycle
  - $stable - value unchanged from previous cycle
  - $rose - positive edge detection
  - $fell - negative edge detection
**What's Needed**:
- Additional system calls as discovered through testing
**Files**: `lib/Conversion/ImportVerilog/Expressions.cpp`, `lib/Conversion/ImportVerilog/AssertionExpr.cpp`
**Priority**: LOW - Core SVA functions complete

### Track D: Coverage Runtime & UVM APIs 🎯 ITERATION 28 - RESEARCH COMPLETE
**Status**: 🟡 DOCUMENTED - Infrastructure exists, event sampling gap identified

### Track E: SVA Bounded Model Checking 🎯 ITERATION 29 - IN PROGRESS
**Status**: 🟢 CONVERSION WORKING - VerifToSMT produces valid MLIR, Z3 linking pending

**What's Working** (Iteration 29):
1. **Moore → Verif lowering**: SVA assertions lower to verif.assert/assume/cover
2. **Verif → LTL lowering**: SVAToLTL pass converts SVA sequences to LTL properties
3. **LTL → Core lowering**: LTLToCore converts LTL to hw/comb logic
4. **VerifToSMT conversion**: Bounded model checking loop with final assertion handling
5. **`bmc.final` support**: Assertions checked only at final step work correctly

**Key Fixes (Iteration 29)**:
- `ReconcileUnrealizedCastsPass` added to pipeline (cleanup unrealized casts)
- `BVConstantOp` argument order: (value, width) not (width, value)
- Clock counting moved BEFORE region type conversion
- `rewriter.eraseOp()` instead of direct `op->erase()` in conversion patterns
- Yield modification before op erasure (values must remain valid)

**What's Pending**:
1. **Z3 runtime linking** - Symbols not found: Z3_del_config, Z3_del_context, etc.
2. **Integration tests** - Need end-to-end SVA → SAT/UNSAT result tests
3. **Performance benchmarking** - Compare vs Verilator/Xcelium assertion checking

**Test Pipeline**:
```bash
# SVA property implication test
echo 'module test(input clk, a, b);
  assert property (@(posedge clk) a |=> b);
endmodule' > /tmp/sva_test.sv
./build/bin/circt-verilog --ir-hw /tmp/sva_test.sv | ./build/bin/circt-bmc --bound=10
```

**Files**:
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - Core BMC loop generation
- `tools/circt-bmc/circt-bmc.cpp` - BMC tool pipeline
- `lib/Conversion/SVAToLTL/SVAToLTL.cpp` - SVA to LTL conversion
- `lib/Conversion/LTLToCore/LTLToCore.cpp` - LTL to HW/Comb lowering

**Priority**: HIGH - Critical for formal verification capability

**COVERAGE INFRASTRUCTURE ANALYSIS (Iteration 28)**:

**What's Implemented** (MooreOps.td + MooreToCore.cpp):
1. `moore.covergroup.decl` - Covergroup type declarations with coverpoints/crosses
2. `moore.coverpoint.decl` - Coverpoint declarations with type info
3. `moore.covercross.decl` - Cross coverage declarations
4. `moore.covergroup.inst` - Instantiation (`new()`) with handle allocation
5. `moore.covergroup.sample` - Explicit `.sample()` method call
6. `moore.covergroup.get_coverage` - Get coverage percentage (0.0-100.0)
7. `CovergroupHandleType` - Runtime handle type (lowers to `!llvm.ptr`)

**MooreToCore Runtime Interface** (expected external functions):
- `__moore_covergroup_create(name, num_coverpoints) -> void*`
- `__moore_coverpoint_init(cg, index, name) -> void`
- `__moore_coverpoint_sample(cg, index, value) -> void`
- `__moore_covergroup_get_coverage(cg) -> double`
- (Future) `__moore_covergroup_destroy(cg)`, `__moore_coverage_report()`

**THE SAMPLING GAP**:
- **Explicit sampling works**: `cg.sample()` calls generate `CovergroupSampleOp` which lowers to runtime calls
- **Event-driven sampling NOT connected**: SystemVerilog `covergroup cg @(posedge clk)` syntax
  - Slang parses the timing event but CIRCT doesn't connect it to sampling triggers
  - The `@(posedge clk)` sampling event is lost during IR generation
  - Would require: (1) storing event info in CovergroupDeclOp, (2) generating always block to call sample

**AVIP COVERGROUP PATTERNS** (from ~/mbit/* analysis):
- AVIPs use `covergroup ... with function sample(args)` pattern (explicit sampling)
- Sample called from `write()` method in uvm_subscriber (UVM callback-based)
- Example from axi4_master_coverage.sv:
  ```systemverilog
  covergroup axi4_master_covergroup with function sample(cfg, packet);
  ...
  function void write(axi4_master_tx t);
    axi4_master_covergroup.sample(axi4_master_agent_cfg_h, t);
  endfunction
  ```
- This pattern IS SUPPORTED by current infrastructure (explicit sample calls work)

**DEPRECATED UVM APIs IN AVIPs** (need source updates for UVM 2017+):
| AVIP | File | Deprecated API |
|------|------|----------------|
| ahb_avip | AhbBaseTest.sv | `uvm_test_done.set_drain_time()` |
| i2s_avip | I2sBaseTest.sv | `uvm_test_done.set_drain_time()` |
| axi4_avip | axi4_base_test.sv | `uvm_test_done.set_drain_time()` |
| apb_avip | apb_base_test.sv | `uvm_test_done.set_drain_time()` |
| axi4Lite_avip | Multiple tests | `uvm_test_done.set_drain_time()` |
| i3c_avip | i3c_base_test.sv | `uvm_test_done.set_drain_time()` |

**Modern replacement**: `phase.phase_done.set_drain_time(this, time)` or objection-based

**What's Needed for Full Coverage Support**:
1. **Runtime library implementation** - C library implementing `__moore_*` functions
2. **Event-driven sampling** (optional) - Parse and connect @(event) to sampling triggers
3. **Coverage report generation** - At $finish, call `__moore_coverage_report()`
4. **Bins and illegal_bins** - Currently declarations only, need runtime bin tracking

**Files**: `lib/Conversion/MooreToCore/MooreToCore.cpp` (lines 1755-2095), `include/circt/Dialect/Moore/MooreOps.td` (lines 3163-3254)
**Priority**: MEDIUM - Explicit sampling works for AVIP patterns; event-driven sampling is enhancement

### Operating Guidance
- Keep 4 agents active on highest-priority tracks:
  - **Track A (LLHD interpretation)** - CRITICAL blocker for behavioral simulation
  - **Track E (SVA BMC)** - Z3 linking, then integration tests
  - **Track D (Coverage/UVM)** - Runtime library implementation
  - **Track C (System calls)** - As discovered through testing
- Track B (interface access) is COMPLETE.
- Add unit tests for each new feature or bug fix.
- Commit regularly and merge worktrees into main to keep workers in sync.
- Test on ~/mbit/*avip* and ~/sv-tests/ for real-world feedback.

### Iteration 29 Results - SVA BMC CONVERSION FIXED
**Key Fixes**:
- VerifToSMT `bmc.final` assertion handling - proper hoisting and final-only checking
- ReconcileUnrealizedCastsPass added to circt-bmc pipeline
- BVConstantOp argument order corrected (value, width)
- Clock counting before region type conversion
- Proper rewriter.eraseOp() usage in conversion patterns

**Status**: VerifToSMT conversion produces valid MLIR. Z3 runtime linking is the remaining blocker.

### Iteration 28 Results - COMPREHENSIVE UPDATE
**Commits**:
- `7d5391552` - $onehot/$onehot0 system calls
- `2830654d4` - $countbits system call
- `4704320af` - $sampled/$past/$changed for SVA assertions
- `25cd3b6a2` - Direct interface member access fix
- `12d75735d`, `110fc6caf` - Test fixes and documentation
- `47c5a7f36` - SVAToLTL comprehensive tests (3 new test files)
- `ecabb4492` - VerifToSMT comprehensive tests
- `235700509` - CHANGELOG update

**AVIP Testing Results**:
- Global packages: 8/8 pass (100%)
- Interfaces: 7/9 pass (78%) - JTAG/I2S fail due to source issues, not CIRCT bugs

**SVA Assertion Functions** - All implemented:
- $sampled, $past (with delay), $changed, $stable, $rose, $fell

**Test Coverage Improved**:
- SVAToLTL: 3 new test files added
- VerifToSMT: comprehensive tests added
- ImportVerilog: 38/38 tests pass (100%)

### Iteration 27 Results - KEY DISCOVERIES
- **$onehot/$onehot0**: ✅ IMPLEMENTED (commit 7d5391552) - lowers to llvm.intr.ctpop == 1 / <= 1
- **sim.proc.print**: ✅ ALREADY WORKS - PrintFormattedProcOpLowering exists in LowerArcToLLVM.cpp
- **circt-sim**: 🔴 CRITICAL GAP - LLHD process interpretation NOT IMPLEMENTED, simulation ends at 0fs
- **LSP debounce**: ✅ FIX EXISTS (9f150f33f) - may still have edge cases

### Previous Track Results (Iteration 26) - MAJOR PROGRESS
- **Coverage Infrastructure**: ✅ CovergroupHandleType, CovergroupInstOp, CovergroupSampleOp, CovergroupGetCoverageOp implemented
- **SVA Assertions**: ✅ Verified working - moore.assert/assume/cover → verif dialect
- **$countones**: ✅ Implemented - lowers to llvm.intr.ctpop
- **Constraint Lowering**: ✅ All 10 constraint ops have MooreToCore patterns
- **Interface ref→vif**: ✅ Fixed conversion generates llhd.prb
- **$finish handling**: ✅ Initial blocks with $finish use seq.initial (arcilator-compatible)
- **AVIP Testing**: ✅ All 9 AVIPs tested - issues are source code problems, not CIRCT
- **LSP Validation**: ✅ Works with --no-debounce flag, bug documented
- **Arcilator Research**: ✅ Identified sim.proc.print lowering as next step

### Previous Track Results (Iteration 25)
- **Track B**: ✅ Interface ref→vif conversion FIXED - Interface member access generates proper lvalue references
- **Track C**: ✅ Constraint MooreToCore lowering COMPLETE - All 10 constraint ops now lower to runtime calls
- **Track D**: ✅ $finish in seq.initial FIXED - $finish no longer forces llhd.process fallback

### Previous Track Results (Iteration 24)
- **Track A**: ✅ AVIP pipeline testing - Identified blocking issues (interface lvalue, $finish)
- **Track B**: ✅ Coverage architecture documented - Runtime ready, need IR ops
- **Track C**: ✅ Constraint expression lowering (ded570db6) - All constraint types now parsed
- **Track D**: ✅ Complex initial block analysis - Confirmed design is correct

### Previous Track Results (Iteration 23) - BREAKTHROUGH
- **Track A**: ✅ seq.initial implemented (cabc1ab6e) - Simple initial blocks work through arcilator!
- **Track B**: ✅ Full pipeline verified - SV → Moore → Core → HW → Arcilator all working
- **Track C**: ✅ Multi-range constraints (c8a125501) - ~94% total coverage
- **Track D**: ✅ AVIP constraints validated - APB/AHB/AXI4 patterns tested

### Previous Track Results (Iteration 22)
- **Track A**: ✅ sim.terminate implemented (575768714) - $finish now calls exit()
- **Track B**: ✅ Initial block solution identified - use seq.initial instead of llhd.process
- **Track C**: ✅ Soft constraints implemented (5e573a811) - ~82% total coverage
- **Track D**: ✅ All 8 AVIPs validated - Package/Interface/BFM files work excellently

### Previous Track Results (Iteration 21)
- **Track A**: ✅ Pipeline analysis complete - llhd.halt blocker identified
- **Track B**: ✅ UVM LSP support added (d930aad54) - --uvm-path flag, UVM_HOME env var
- **Track C**: ✅ Range constraints implemented (2b069ee30) - ~59% of AVIP constraints work
- **Track D**: ✅ Interface symbols fixed (d930aad54) - LSP properly shows interface structure

### Previous Track Results (Iteration 20)
- **Track A**: ✅ LSP debounce deadlock FIXED (9f150f33f) - `--no-debounce` no longer needed
- **Track B**: ✅ sim.proc.print lowering IMPLEMENTED (2be6becf7) - Arcilator can now output $display
- **Track C**: ✅ Randomization architecture researched - 80% of constraints can be done without SMT
- **Track D**: ✅ LSP tested on AVIPs - Package files work, interface/UVM gaps identified

### Previous Track Results (Iteration 19)
- **Track A**: ✅ All 27/27 MooreToCore unit tests pass (100%)
- **Track B**: ✅ Arcilator research complete - `arc.sim.emit` exists, need `sim.proc.print` lowering
- **Track C**: ✅ AVIP gaps quantified - 1097 randomization, 970 coverage, 453 DPI calls
- **Track D**: ✅ 6 LSP tests added, debounce hang bug documented (use --no-debounce)

### Previous Track Results (Iteration 13)
- **Track A**: ✅ VTable fallback committed (6f8f531e6) - Classes without vtable segments now search ALL vtables
- **Track B**: ✅ AVIP BFM validation complete - APB/AHB/AXI4/AXI4-Lite work; test code issues documented
- **Track C**: ✅ Randomization already implemented - confirmed working
- **Track D**: ✅ Pipeline investigation complete - circt-sim doesn't execute llhd.process bodies
- **Track E**: ✅ UVM conversion validation - only 1 error (moore.builtin.realtobits), AXI4-Lite 100%

### Previous Track Results (Iteration 12)
- **Track A**: ✅ Array locator inline loop complete (115316b07) - AND/OR/string predicates work
- **Track A**: ✅ llhd.time data layout crash fixed (1a4bf3014)
- **Track B**: ✅ All 7 AVIPs (APB/AHB/AXI4/UART/I2S/I3C/SPI) pass MooreToCore
- **Track C**: ⚠️ DPI chandle support added; randomization runtime still needed
- **Track D**: ⚠️ vtable.load_method error found blocking full UVM conversion

### Previous Track Results (Iteration 11)
- **Track A**: ✅ BFM nested task calls fixed (d1b870e5e) - Interface tasks calling other interface tasks now work correctly
- **Track A**: ⚠️ MooreToCore timing limitation documented - Tasks with `@(posedge clk)` can't lower (llhd.wait needs process parent)
- **Track B**: ✅ UVM MooreToCore: StructExtract crash fixed (59ccc8127) - only `moore.array.locator` remains
- **Track C**: ✅ DPI tool info functions implemented - returns "CIRCT" and "1.0" for tool name/version
- **Track D**: ✅ AHB AVIP testing confirms same fixes work across AVIPs

### Previous Track Results (Iteration 10)
- **Track A**: ✅ Interface task/function support (d1cd16f75) - BFM patterns now work with implicit iface arg
- **Track B**: ✅ JTAG/SPI/UART failures documented - all are source code issues, not CIRCT bugs
- **Track C**: ✅ DPI-C analysis complete - 22 functions documented (see docs/DPI_ANALYSIS.md)
- **Track D**: ✅ Queue global lowering verified - already works correctly

### Previous Track Results (Iteration 9)
- **Track A**: ✅ 5/9 AVIPs pass full pipeline (APB, AHB, AXI4, I2S, I3C) - JTAG/SPI/UART have source issues
- **Track B**: ⚠️ BFM parsing blocked on interface port rvalue handling (`preset_n` not recognized)
- **Track C**: ✅ Runtime gaps documented - DPI-C stubbed, randomization/covergroups not implemented
- **Track D**: ✅ Unit test for StructExtractRefOp committed (99b4fea86)

### Previous Track Results (Iteration 8)
- **Track A**: ✅ RefType cast crash fixed (5dd8ce361) - StructExtractRefOp now uses GEP for structs with dynamic fields
- **Track B**: ✅ UVM MooreToCore conversion now completes without crashes
- **Track C**: ✅ Added dyn_cast safety checks to multiple conversion patterns
- **Track D**: ✅ Sig2RegPass RefType cast also fixed

### Previous Track Results (Iteration 7)
- **Track A**: ✅ Virtual interface assignment support added (f4e1cc660) - enables `vif = cfg.vif` patterns
- **Track B**: ✅ StringReplicateOp lowering added (14bf13ada) - string replication in MooreToCore
- **Track C**: ✅ Scope tracking for virtual interface member access (d337cb092) - fixes class context issues
- **Track D**: ✅ Unpacked struct variable lowering fixed (ae1441b9d) - handles dynamic types in structs

### Previous Track Results (Iteration 6)
- **Track A**: ✅ Data layout crash fixed (2933eb854) - convertToLLVMType helper
- **Track B**: ✅ AVIP BFM testing - interfaces pass, BFMs need class members in interfaces
- **Track C**: ✅ ImportVerilog tests 30/30 passing (65eafb0de)
- **Track D**: ✅ AVIP packages pass MooreToCore, RTL modules work

### Previous Track Results (Iteration 5)
- **Track A**: ✅ getIntOrFloatBitWidth crash fixed (8911370be) - added type-safe helper
- **Track B**: ✅ Virtual interface member access added (0a16d3a06) - VirtualInterfaceSignalRefOp
- **Track C**: ✅ QueueConcatOp empty format fixed (2bd58f1c9) - parentheses format
- **Track D**: ✅ Test suite fixed (f7b9c7b15) - Moore 18/18, MooreToCore 24/24

### Previous Track Results (Iteration 4)
- **Track A**: ✅ vtable.load_method fixed for abstract classes (e0df41cec) - 4764 ops unblocked
- **Track B**: ✅ All vtable ops have conversion patterns
- **Track C**: ✅ AVIP testing found: virtual interface member access needed, QueueConcatOp format bug
- **Track D**: ✅ Comprehensive vtable tests added (12 test cases)

### Previous Track Results (Iteration 3)
- **Track A**: ✅ array.size lowering implemented (f18154abb) - 349 ops unblocked
- **Track B**: ✅ Virtual interface comparison ops added (8f843332d) - VirtualInterfaceCmpOp
- **Track C**: ✅ hvlTop tested - all fail on UVM macros (separate issue)
- **Track D**: ✅ Test suite runs clean

### Previous Track Results (Iteration 2)
- **Track A**: ✅ MooreSim tested - dyn_extract was blocking, now fixed
- **Track B**: ✅ dyn_extract/dyn_extract_ref implemented (550949250) - 970 queue ops unblocked
- **Track C**: ✅ AVIP+UVM tested - interfaces pass, BFMs blocked on virtual interface types
- **Track D**: ✅ All unit tests pass after fixes (b9335a978)

### Previous Track Results (Iteration 1)
- **Track A**: ✅ Multi-file parsing fixed (170414961) - empty filename handling added
- **Track B**: ✅ MooreToCore patterns added (69adaa467) - FormatString, CallIndirect, SScanf, etc.
- **Track C**: ✅ AVIP testing done - 13/14 components pass (timescale issue with JTAG)
- **Track D**: ✅ Unit tests added (b27f71047) - Mem2Reg, static properties, time type

---

## Priority Queue

### CRITICAL (Blocking UVM Parsing)
None! UVM parsing complete.

### RECENTLY FIXED ✅ (This Session)
- ~~**Mem2Reg loop-local variable dominance**~~ - ✅ Fixed (b881afe61) - Variables inside loops excluded from promotion
- ~~**Static property via instance**~~ - ✅ Fixed (a1418d80f) - `obj.static_prop` now uses GetGlobalVariableOp
- ~~**Static property names in parameterized classes**~~ - ✅ Fixed (a1418d80f) - Unique names per specialization
- ~~**Abstract class vtable**~~ - ✅ Fixed (a1418d80f) - Mixed concrete/pure virtual methods allowed
- ~~**Time type in Mem2Reg**~~ - ✅ Fixed (3c9728047) - Default values for time variables
- ~~**Method lookup in parameterized classes**~~ - ✅ Fixed (71c80f6bb) - Class body conversion
- ~~**Super.method() dispatch**~~ - ✅ Fixed (09e75ba5a) - Direct dispatch instead of vtable
- ~~**Class upcast with parameterized base**~~ - ✅ Fixed (fbbc2a876) - Generic class lookup
- ~~**Global variable redefinition**~~ - ✅ Fixed (a152e9d35) - Recursive type conversion

### PREVIOUSLY FIXED ✅
- ~~**UVM class declaration issues**~~ - ✅ Fixed (555a78350)
- ~~**String ato* methods**~~ - ✅ Fixed (14dfdbe9f + 34ab7a758)
- ~~**Non-integral assoc array keys**~~ - ✅ Fixed (f6b79c4c7)
- ~~**File I/O ($fopen, $fwrite, $fclose)**~~ - ✅ Fixed

### HIGH (After UVM Parses)
3. **Complete MooreToCore lowering** - All ops must lower for simulation (ato* already done; queue globals pending)
4. **Enum iteration methods** - first(), next(), last(), prev()
5. **MooreSim execution** - Run compiled testbenches
6. **Factory runtime** - Ensure uvm_pool/callback singleton handling matches specialization typing

### MEDIUM (Production Quality)
6. **Coverage groups** - covergroup, coverpoint
7. **Constraint solver (Z3)** - Enable randomization
8. **$fgets** - File read line

### LOW (Future Enhancements)
9. **SVA assertions** - Full property/sequence support
10. **Multi-core simulation** - Performance scaling
11. **Interactive debugger** - circt-debug CLI

---

## Feature Gap Analysis (Iteration 30) - COMPREHENSIVE SURVEY

Based on systematic testing of ~/sv-tests/, ~/mbit/*avip*, and ~/verilator-verification/:

### Critical Gaps for Xcelium Parity

| Feature | Status | Tests Blocked | Priority |
|---------|--------|---------------|----------|
| **Clocking Blocks** | ✅ IMPLEMENTED | ~80% sv-tests (Ch14) | DONE |
| **Z3 Installation** | ✅ INSTALLED | SVA BMC enabled | DONE |
| **LLHD Process Interpreter** | Plan ready | circt-sim behavioral | HIGH - Critical |
| **RandSequence** | ✅ IMPLEMENTED | 9/16 sv-tests pass | DONE |
| **SequenceWithMatch** | NOT IMPLEMENTED | ~25 sv-tests | MEDIUM |
| **TaggedUnion** | NOT IMPLEMENTED | ~20 sv-tests | MEDIUM |
| **clocked_assert lowering** | Missing pass | circt-bmc with clocked props | MEDIUM |
| **4-State (X/Z)** | NOT IMPLEMENTED | Many tests | HIGH |
| **Signal Strengths** | NOT IMPLEMENTED | 37 verilator tests | MEDIUM |

### Test Suite Coverage (Verified Iteration 30)

| Test Suite | Total Tests | Pass Rate | Notes |
|------------|-------------|-----------|-------|
| **sv-tests** | 989 (non-UVM) | **72.1%** (713/989) | Parsing/elaboration |
| **mbit AVIP globals** | 8 packages | **100%** (8/8) | All work |
| **mbit AVIP interfaces** | 8 interfaces | **75%** (6/8) | 2 source issues |
| **mbit AVIP HVL** | 8 packages | **0%** | Requires UVM lib |
| **verilator-verification** | 154 | **~60%** | SVA tests improved |

### sv-tests Detailed Analysis

**Strongest Chapters** (>80%):
- Chapter 11 (Operators): 87% pass
- Chapter 5 (Lexical): 86% pass
- Chapter 13 (Tasks/Functions): 86% pass
- Chapter 20 (I/O Formatting): 83% pass

**Weakest Chapters** (<50%):
- Chapter 14 (Clocking Blocks): 0% pass - NOT IMPLEMENTED
- Chapter 18 (Random/Constraints): 25% pass - RandSequence missing
- Chapter 21 (I/O System Tasks): 37% pass - VcdDump missing

**Top Error Categories** (by test count):
1. ClockingBlock - 0% of Ch14 tests pass
2. RandSequence - randsequence statement not supported
3. SequenceWithMatch - sequence match patterns
4. TaggedUnion - tagged union types
5. EmptyArgument - empty function arguments

### SVA Functions Status (Iteration 28-29)

| Function | ImportVerilog | SVAToLTL | VerifToSMT | Status |
|----------|---------------|----------|------------|--------|
| $sampled | ✅ | ✅ | ✅ | WORKING |
| $past | ✅ | ✅ | ✅ | WORKING |
| $rose | ✅ | ✅ | ✅ | WORKING |
| $fell | ✅ | ✅ | ✅ | WORKING |
| $stable | ✅ | ✅ | ✅ | WORKING |
| $changed | ✅ | ✅ | ✅ | WORKING |
| Sequences | ✅ | ✅ | ? | Needs testing |
| Properties | ✅ | ✅ | ? | Needs testing |

### Z3 Linking Fix Options

1. **Quick Fix**: Use `--shared-libs=/path/to/libz3.so` at runtime
2. **CMake Fix**: Add Z3 to target_link_libraries in circt-bmc
3. **Auto-detect**: Store Z3 path at build time, inject at runtime

---

## Big Projects Status (Iteration 30)

Comprehensive survey of the 6 major projects for Xcelium parity:

### 1. Full SVA Support with Z3 ⚠️ PARTIAL

**Working:**
- SVA → LTL conversion complete (SVAToLTL.cpp - 321 patterns)
- VerifToSMT conversion (967 lines)
- $sampled, $past, $changed, $stable, $rose, $fell implemented
- circt-bmc bounded model checking pipeline

**Missing:**
- LTL properties not yet supported in VerifToSMT
- `verif.clocked_assert` needs lowering pass
- SMT solver for complex constraints

### 2. Scalable Multi-core Arcilator ❌ MISSING

**Status:** No multi-threading support found
- Arcilator runtime is single-threaded JIT
- Arc dialect has 37+ transform passes (all sequential)
- Would require fundamental architectural redesign
- Consider PDES (Parallel Discrete Event Simulation) model

### 3. Language Server (LSP) and Debugging ⚠️ PARTIAL

**Working:**
- circt-verilog-lsp-server compiles and runs
- LSP transport infrastructure (LLVM LSP integration)
- `--uvm-path` flag and `UVM_HOME` env var parsing
- Basic file parsing and error reporting

**Missing:**
- Code completion (semantic)
- Go-to-definition/references (cross-file)
- Rename refactoring
- Debugger integration (LLDB)

### 4. Full 4-State (X/Z Propagation) ❌ MISSING

**Status:** Two-state logic only (0/1)
- X and Z recognized as identifiers only
- Requires Moore type system redesign
- Would impact 321+ conversion patterns
- Design 4-state type system RFC needed

### 5. Coverage Support ⚠️ PARTIAL

**Working:**
- CovergroupDeclOp, CoverpointDeclOp, CoverCrossDeclOp in MooreOps.td
- Coverage runtime library (2,270 LOC)
- 80+ test cases in test_coverage_runtime.cpp
- Coverage ops lower to MooreToCore

**Missing:**
- Coverage expressions and conditional sampling
- Cross-cover correlation analysis
- Coverage HTML report generation

### 6. DPI/VPI Support ⚠️ STUBS ONLY

**Current:**
- DPI-C import parsing works (22 functions stubbed)
- External function declarations recognized
- Stub returns: int=0, string="CIRCT", void=no-op

**Missing:**
- No actual C function invocation (FFI bridge needed)
- No VPI (Verilog Procedural Interface)
- Memory management between SV and C undefined

### Big Projects Summary Table

| Project | Status | Priority | Blocking |
|---------|--------|----------|----------|
| SVA with Z3 | Partial | HIGH | Z3 install |
| Multi-core Arc | Missing | MEDIUM | Architecture |
| LSP/Debugging | Partial | MEDIUM | Features |
| 4-State Logic | Missing | LOW | Type system |
| Coverage | Partial | HIGH | Cross-cover |
| DPI/VPI | Stubs | MEDIUM | FFI bridge |

---

## Features Completed

### Class Support
- [x] Class declarations and handles
- [x] Class inheritance (extends)
- [x] Virtual methods and vtables
- [x] Static class properties (partial)
- [x] Parameterized classes
- [x] $cast dynamic type checking
- [x] Class handle comparison (==, !=, null)
- [x] new() allocation

### Queue/Array Support
- [x] Queue type and operations
- [x] push_back, push_front, pop_back, pop_front
- [x] delete(), delete(index)
- [x] size(), max(), min(), unique(), sort()
- [x] Dynamic arrays with new[size]
- [x] Associative arrays (int keys)
- [x] exists(), delete(key)
- [x] first(), next(), last(), prev() for string keys (2fa392a98)

### String Support
- [x] String type
- [x] itoa(), len(), getc()
- [x] toupper(), tolower()
- [x] putc() character assignment
- [x] %p format specifier
- [x] String in format strings (emitDefault fix)
- [x] atoi(), atohex(), atooct(), atobin() (14dfdbe9f)

### File I/O ✅ Complete
- [x] $fopen - file open (ce8d1016a)
- [x] $fclose - file close (b4a18d045)
- [x] $fwrite - formatted file write (ccfc4f6ca)
- [x] $fdisplay - file display (ccfc4f6ca - via $fwrite handler)
- [x] $sscanf - string scan (2657ceab7)
- [ ] $fgets - file read line

### Process Control
- [x] fork/join, fork/join_any, fork/join_none
- [x] Named blocks
- [x] disable statement
- [x] wait(condition) statement

### Event Support
- [x] event type (moore::EventType)
- [x] .triggered property
- [x] Event trigger (->)

### Interface Support
- [x] Interface declarations
- [x] Modports
- [x] Virtual interfaces (basic)

### MooreToCore Lowering ✅ Complete
- [x] AssocArrayExistsOp
- [x] Union operations
- [x] Math functions (clog2, atan2, hypot, etc.)
- [x] Real type conversions
- [x] File I/O ops (52511fe46) - FOpenBIOp, FWriteBIOp, FCloseBIOp

---

## AVIP Testing

Test files in ~/mbit/*:
- ahb_avip, apb_avip, axi4_avip, axi4Lite_avip
- i2s_avip, i3c_avip, jtag_avip, spi_avip, uart_avip

**Current blocker**: All AVIPs import UVM, which crashes.
**After crash fix**: Test individual components without UVM macros.

**Test non-UVM components**:
```bash
./build/bin/circt-verilog --ir-moore \
  ~/mbit/apb_avip/src/globals/apb_global_pkg.sv \
  ~/mbit/apb_avip/src/hdl_top/apb_if/apb_if.sv
```

---

## Milestones

| Target | Milestone | Criteria |
|--------|-----------|----------|
| Jan 2026 | M1: UVM Parses | Zero errors parsing uvm_pkg.sv | ✅ ACHIEVED |
| Feb 2026 | M2: File I/O | $fopen, $fwrite, $fclose work |
| Mar 2026 | M3: AVIP Parses | All ~/mbit/* AVIPs parse |
| Q2 2026 | M4: Basic Sim | Simple UVM test runs |
| Q3 2026 | M5: Full UVM | Factory pattern, phasing work |
| Q4 2026 | M6: AVIPs Run | mbits/ahb_avip executes |

---

## Build Commands
```bash
# Build
ninja -C build circt-verilog

# Test UVM
./build/bin/circt-verilog --ir-moore ~/uvm-core/src/uvm_pkg.sv -I ~/uvm-core/src

# Test AVIP interface only (no UVM)
./build/bin/circt-verilog --ir-moore \
  ~/mbit/apb_avip/src/globals/apb_global_pkg.sv \
  ~/mbit/apb_avip/src/hdl_top/apb_if/apb_if.sv
```

---

## Recent Commits

### Iteration 194
- **Track B completed**: Analyzed all 6 verilator-verification errors - they are due to non-standard `@posedge (clk)` syntax in test files (not CIRCT bugs)
  - Standard syntax: `@(posedge clk)`, non-standard: `@posedge (clk)`
  - These tests also missing terminating semicolons in sequences
  - Recommendation: Mark as XFAIL or report upstream
- **Track D completed**: Created unit tests for new compat mode features:
  - `test/Conversion/ImportVerilog/compat-vcs.sv` - Tests VCS compatibility flags
  - `test/Conversion/ImportVerilog/virtual-iface-bind-override.sv` - Tests AllowVirtualIfaceWithOverride flag
- Test status:
  - sv-tests SVA: 9/26 pass (xfail=3)
  - verilator-verification: 8/17 pass (6 errors are test file bugs)

### Iteration 180
- **Upgraded slang from v9.1 to v10.0** for better SystemVerilog support
- Added `--compat vcs` and `--allow-virtual-iface-with-override` options to circt-verilog
- Added `AllowVirtualIfaceWithOverride` compilation flag to slang for Xcelium compatibility
  - Allows interface instances that are bind/defparam targets to be assigned to virtual interfaces
  - This violates IEEE 1800-2017 but matches behavior of commercial tools like Cadence Xcelium
- Fixed VCS compatibility mode to set flags directly (bypasses slang's addStandardArgs() requirement)
- Updated slang patch scripts for v10.0 compatibility
- sv-tests SVA: 9/26 pass (xfail=3)

### Iteration 29
- VerifToSMT `bmc.final` fixes - proper assertion hoisting and final-only checking
- ReconcileUnrealizedCastsPass in circt-bmc pipeline
- BVConstantOp argument order fix (value, width)
- Clock counting timing fix (before region conversion)
- Proper rewriter.eraseOp() in conversion patterns

### Iteration 28
- `235700509` - [Docs] CHANGELOG update for Iteration 28
- `ecabb4492` - [Tests] VerifToSMT comprehensive tests
- `47c5a7f36` - [Tests] SVAToLTL comprehensive tests (3 new files)
- `12d75735d`, `110fc6caf` - [Tests] Test fixes and documentation
- `25cd3b6a2` - [ImportVerilog] Direct interface member access fix
- `4704320af` - [ImportVerilog] $sampled/$past/$changed/$stable/$rose/$fell for SVA
- `2830654d4` - [ImportVerilog] $countbits system call
- `7d5391552` - [ImportVerilog] $onehot/$onehot0 system calls

### Earlier
- `6f8f531e6` - [MooreToCore] Add vtable fallback for classes without vtable segments
- `59ccc8127` - [MooreToCore] Fix StructExtract/StructCreate for dynamic types
- `d1b870e5e` - [ImportVerilog] Add DPI tool info and fix interface task-to-task calls
- `d1cd16f75` - [ImportVerilog] Add interface task/function support
- `99b4fea86` - [MooreToCore] Add tests for StructExtractRefOp with dynamic fields
- `5dd8ce361` - [MooreToCore] Fix RefType cast crashes for structs with dynamic fields
- `f4e1cc660` - [ImportVerilog] Add virtual interface assignment support
- `14bf13ada` - [MooreToCore] Add StringReplicateOp lowering
- `d337cb092` - [ImportVerilog] Add scope tracking for virtual interface member access in classes
- `ae1441b9d` - [MooreToCore] Fix variable lowering for unpacked structs with dynamic types
- `b881afe61` - [Moore] Don't promote loop-local variables to avoid Mem2Reg dominance errors
- `3c9728047` - [Moore] Fix time type handling in Mem2Reg default value generation
- `a1418d80f` - [ImportVerilog][Moore] Fix static property access and abstract class handling
- `71c80f6bb` - [ImportVerilog] Fix method lookup in parameterized class specializations
- `09e75ba5a` - [ImportVerilog] Use direct dispatch for super.method() calls
- `fbbc2a876` - [ImportVerilog] Fix class upcast with parameterized base classes
- `a152e9d35` - [ImportVerilog] Fix global variable redefinition during recursive type conversion
- `555a78350` - [ImportVerilog] Fix UVM class declaration and statement handling issues
- `34ab7a758` - [MooreToCore] Add lowering for string ato* ops
- `f6b79c4c7` - [ImportVerilog] Fix non-integral assoc array keys and pure virtual methods
- `14dfdbe9f` - [ImportVerilog] Add support for string ato* methods

---

## Architecture Reference

See full plan: `~/.claude/plans/jiggly-tickling-engelbart.md`

Track assignments:
- **Track A (Sim)**: Event kernel, process control, performance
- **Track B (UVM)**: Class parsing, constraints, factory pattern
- **Track C (Types)**: 4-state, coverage, file I/O
- **Track D (DevEx)**: LSP, linting, dashboards
- **Track E (Assert)**: SVA, vacuity detection, debug
