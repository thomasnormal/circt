# CIRCT UVM Parity Project Plan

## Goal
Bring CIRCT up to parity with Cadence Xcelium for running UVM testbenches.
Run `~/uvm-core` and `~/mbit/*avip` testbenches using only CIRCT tools.

---

## Remaining Limitations & Next Steps

### CRITICAL: Simulation Runtime Blockers

> **See `.claude/plans/ticklish-sleeping-pie.md` for detailed implementation plan.**

1. **Simulation Time Advancement** ❌ CRITICAL: `circt-sim` ends at 0fs because `ProcessScheduler::advanceTime()` doesn't check `EventScheduler` for queued events. The LLHD interpreter exists (1700+ lines) but process resumption after delays never triggers.

2. **DPI/VPI Real Hierarchy** ⚠️ HIGH: DPI stubs use disconnected in-memory map. `uvm_hdl_read("path")` returns 0, not actual signal value. Need to bridge signal registry from `LLHDProcessInterpreter` to `MooreRuntime`.

3. **Virtual Interface Binding** ⚠️ HIGH: Signal access works, but runtime binding (`driver.vif = apb_if`) doesn't propagate. Need `moore.virtual_interface.bind` operation.

4. **4-State X/Z Propagation** ⚠️ MEDIUM: Type system supports it (`Domain::FourValued`, `FVInt`), but lowering loses X/Z info. Constants parsed but not preserved in IR.

### Track Status & Next Tasks (Iteration 71+)

**Simulation Runtime (Critical Path)**:
| Track | Focus Area | Current Status | Next Priority |
|-------|-----------|----------------|---------------|
| **A** | Simulation Runtime | Time advancement broken (ends at 0fs) | Fix ProcessScheduler↔EventScheduler integration |
| **B** | DPI/VPI Hierarchy | Stubs with in-memory map only | Build signal registry bridge to real signals |

**Feature Development (Parallel)**:
| Track | Focus Area | Current Status | Next Priority |
|-------|-----------|----------------|---------------|
| **C** | Coverage | UCDB file format complete | Coverage GUI/reports, virtual interface binding |
| **D** | LSP Tooling | Inlay hints complete | Semantic highlighting |
| **E** | Randomization/UVM | RandSequence fractional N complete | 4-state X/Z |

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
| Virtual interfaces | ✅ | ✅ | ⚠️ | ❌ | ⚠️ |
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

## Current Status: ITERATION 71 - RandSequence Fractional N Support (January 21, 2026)

**Summary**: Fixed `rand join (N)` to support fractional N values per IEEE 1800-2017 Section 18.17.5.

### Iteration 71 Highlights

**Track E: RandSequence Improvements**
- Fixed `rand join (N)` where N is a real number (e.g., 0.5)
- Per IEEE 1800-2017, fractional N (0 <= N <= 1) means execute `round(N * numProds)` productions
- Previously crashed on fractional N; now properly handles both integer and real values
- All 12 non-negative sv-tests for section 18.17 now pass (up from 11)
- Added test case for `rand join (0.5)` in randsequence.sv

**Files Modified**:
- `lib/Conversion/ImportVerilog/Statements.cpp` - Handle real values in randJoinExpr
- `test/Conversion/ImportVerilog/randsequence.sv` - Added fractional ratio test

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
- ⛔ Goto/non-consecutive repeat still single-step in BMC
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
