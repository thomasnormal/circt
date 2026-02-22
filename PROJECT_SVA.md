# CIRCT SVA (SystemVerilog Assertions) Status

This document tracks the status of SVA support in CIRCT, including bugs, missing features, and test coverage.

**Last Updated**: February 22, 2026

## Overview

SVA (SystemVerilog Assertions) is handled by the **Codex agent** for dedicated development.
This document serves as a central tracking point for SVA-related issues.

## Test Suites

### sv-tests Chapter-16 (Assertions)

**Current Status**: 23/26 passing (88.5%), 3 XFAIL, 0 FAIL, 0 ERROR
(`utils/run_sv_tests_circt_bmc.sh` with `RISING_CLOCKS_ONLY=1`, tags 16.* + 9.4.4;
log: `sv-tests-bmc-run.txt`)

Note: The BMC harness treats tests with `:type:` containing `parsing` as
compile-only and skips BMC evaluation for those cases.
Additional compile-only or XFAIL expectations live in
`utils/sv-tests-bmc-expect.txt`.

| Test Category | Status | Notes |
|---------------|--------|-------|
| Property local variables (16.10) | Parsing-only | Compile-only in sv-tests (skipped in BMC) |
| Sequence local variables (16.10) | Parsing-only | Compile-only in sv-tests (skipped in BMC) |
| Sequence subroutines (16.11) | Partial | Match-item subroutine calls are ignored in formal lowering |
| Property declarations (16.12) | Compile-only | No stimulus for BMC in sv-tests |
| Multiclock sequences (16.13) | Partial | LTLToCore gates nested `ltl.clock`; sv-tests only has UVM coverage |
| Assume property (16.14) | Partial | |
| Property disable iff (16.15) | Passing | |
| Expect (16.17) | Partial | Smoke coverage via sv-tests mini UVM harness |
| Assert final (16.2) | Partial | Lowered to bmc.final; smoke coverage via sv-tests mini UVM harness |

### Yosys SVA BMC Tests

**Current Status**: 14/14 passing (100%), 2 skipped (VHDL)

| Test | Status | Notes |
|------|--------|-------|
| basic00.sv | PASS | Simple assertion |
| basic01.sv | PASS | Multiple assertions |
| basic02.sv | PASS | Bind assertions |
| basic03.sv | PASS | $past comparisons, sampled-value alignment |
| basic04.sv | PASS | |
| basic05.sv | PASS | |
| counter.sv | PASS | Counter with assertions |
| extnets.sv | PASS | External nets |
| nested_clk_else.sv | PASS | |
| sva_not.sv | PASS | Negation |
| sva_range.sv | PASS | Range repetition |
| sva_throughout.sv | PASS | Throughout operator |
| sva_value_change_*.sv | PASS | Value change functions |

### verilator-verification

**SVA-related failures**: 0 tests (17/17 passing)

## Missing Features (Priority Order)

### High Priority

1. **Property local variables in UVM patterns**
   - Tests: 16.10--*-uvm.sv
   - Status: Parses with real UVM; `--ir-hw` compiles after inline-calls fix
   - BMC/LEC: Added end-to-end coverage in
     `test/Tools/circt-bmc/sva-uvm-local-var-e2e.sv`,
     `test/Tools/circt-bmc/sva-uvm-seq-local-var-e2e.sv`,
     `test/Tools/circt-lec/lec-uvm-local-var.sv`, and
     `test/Tools/circt-lec/lec-uvm-seq-local-var.sv`
   - Added sv-tests mini UVM smoke harness coverage in
     `test/Tools/circt-bmc/sv-tests-uvm-smoke.mlir` and
     `test/Tools/circt-lec/sv-tests-uvm-lec-smoke-mini.mlir`.
   - Still missing: full sv-tests UVM harness coverage on the upstream suite
     (runtime-heavy tests).

### Medium Priority

2. **BMC environment modeling for bare properties (16.12)**
   - Tests: `16.12--property`, `16.12--property-disj`
   - Needs: Explicit clock/input assumptions or harness guidance
   - Smoke coverage: `test/Tools/circt-bmc/sv-tests-bare-property-smoke.mlir`

3. **Multiclock sequences (16.13)**
   - LTLToCore now gates nested `ltl.clock` on sampled ticks
   - BMC can interleave multiple `seq.to_clock` inputs with `--allow-multi-clock`
   - `circt-bmc` now exposes `--allow-multi-clock`
   - Added MLIR end-to-end SVA coverage with multiple clocked asserts
   - Added SV import coverage for multi-clock SVA assertions
   - Added SV end-to-end BMC coverage for multi-clock asserts
   - UVM case parses with real UVM and compiles in `--ir-hw`
   - BMC/LEC: Added end-to-end UVM coverage in
     `test/Tools/circt-bmc/sva-uvm-multiclock-e2e.sv` and
     `test/Tools/circt-lec/lec-uvm-multiclock.sv`
   - Added sv-tests mini UVM smoke harness coverage in
     `test/Tools/circt-bmc/sv-tests-uvm-smoke.mlir` and
     `test/Tools/circt-lec/sv-tests-uvm-lec-smoke-mini.mlir`.
   - Still missing: full sv-tests UVM harness coverage on the upstream suite.

4. **Property interfaces (16.12)**
   - Interface-based property definitions
   - Basic interface property instantiation now supported (virtual interface refs)
   - Interface port connections now lower to signal assignments (enables clocks)
   - BMC coverage added in `test/Tools/circt-bmc/sva-interface-property-e2e.sv`
   - LEC coverage added in `test/Tools/circt-lec/lec-interface-property.sv`
     (LLHD interface storage is stripped in `circt-lec`)
   - UVM BMC/LEC coverage added in
     `test/Tools/circt-bmc/sva-uvm-interface-property-e2e.sv` and
     `test/Tools/circt-lec/lec-uvm-interface-property.sv`
   - Added sv-tests mini UVM smoke harness coverage in
     `test/Tools/circt-bmc/sv-tests-uvm-smoke.mlir` and
     `test/Tools/circt-lec/sv-tests-uvm-lec-smoke-mini.mlir`.
   - Still missing: UVM property interface coverage on the full sv-tests suite.

5. **Sequence subroutines (16.11)**
   - Match-item subroutine calls now parse and lower
   - display/write-style system match-item calls now preserve side effects
     in ImportVerilog lowering (instead of being dropped)
   - UVM sequence subroutine case compiles in `--ir-hw`
   - BMC/LEC: Added end-to-end UVM coverage in
     `test/Tools/circt-bmc/sva-uvm-seq-subroutine-e2e.sv` and
     `test/Tools/circt-lec/lec-uvm-seq-subroutine.sv`
   - Remaining: model subroutine side effects in simulation/LEC

### Low Priority

6. **Property disable iff edge cases**
   - Test: 16.15--property-disable-iff-fail.sv

## Roadmap (Feb 6, 2026)

1. **Core semantics + regressions**
   - `first_match` boundedness, `throughout/within/intersect`, open-ended ranges,
     repetition operators, and sequence/property arguments (with default clocking).
2. **Clocking/disable semantics**
   - Multi-clock sampling; default clocking/disable propagation; reset-aware
     sampled-value functions.
3. **Bind + hierarchy**
   - Robust bind across param/genvar-heavy hierarchies with stable naming.
4. **Liveness**
   - k-induction with final checks and liveness operators.
5. **Trace UX**
   - Hierarchical, clock-aware counterexample mapping.

## Implementation Status

### Completed Features

- **SVA assertion functions** (Iteration 28-29):
  - `$sampled` - Current value sampling
  - `$past` - Previous value access (basic + explicit clocking, enable for explicit/timed/property, sampled-control support for associative arrays)
  - `$changed` - Value change detection
  - `$stable` - Value stability check
  - `$rose` - Rising edge detection
  - `$fell` - Falling edge detection
  - sampled edge support over unpacked aggregates now includes associative arrays
    (`assoc_array` / wildcard associative arrays) in both assertion-clocked and
    explicit-clock helper lowering paths
  - sampled stability/change support over unpacked aggregates now includes
    associative arrays (`assoc_array` / wildcard associative arrays) in both
    assertion-clocked and explicit-clock helper lowering paths
  - sampled-value real operand support:
    - `$stable` / `$changed` use real equality (`moore.feq`) for comparisons
    - `$rose` / `$fell` booleanize real samples via `value != 0.0`
    - supported in both explicit-clock helper and direct clocked-assertion
      lowering paths
  - sampled-value string stability/change parity:
    - direct and explicit-clock helper lowering for `$stable` / `$changed`
      now preserve native string comparison (`moore.string_cmp`)
    - removes lossy string-to-int conversion in sampled stability/change paths
  - sampled-value string edge parity:
    - direct and explicit-clock helper lowering for `$rose` / `$fell`
      now uses native string booleanization (`moore.bool_cast` on `string`)
    - removes sampled edge reliance on `string_to_int` conversion
  - sampled-value event operand support:
    - direct and explicit-clock helper lowering now supports `event` operands
      for `$stable`, `$changed`, `$rose`, and `$fell`
    - event sampled semantics use native event booleanization
      (`moore.bool_cast` on `event`) in both paths
  - `$past` sampled-value controls now support real operands in explicit
    clocking / enable-control lowering paths
  - `$past` sampled-value controls now preserve string operands natively
    (no lossy string/int round-trip in helper lowering)
  - `$past` sampled-value controls now preserve `time` operands in native
    helper storage (no helper-side bitvector history state)

- **LTL operators**:
  - `|->` - Overlapping implication
  - `|=>` - Non-overlapping implication
  - `##N` - Fixed delay
  - `##[n:m]` - Range delay
  - `[*N]` - Fixed repetition
  - `[*n:m]` - Range repetition
  - `throughout` - Throughout operator
  - `eventually` (weak) - Lowered with weak tag (no final liveness check)
  - `s_eventually [n:m]` - Lowered to `ltl.delay` with bounded range
  - `eventually [n:m]` - Lowered to `ltl.delay` with bounded range
  - bounded strong `s_eventually [n:m]` now enforces finite delayed-progress
    obligations for both property and sequence operands
  - `s_until` / `s_until_with` - Lowered to `ltl.until` AND `ltl.eventually`
  - `s_nexttime` / `s_always` - Lowered to delay/repeat (strong forms)
  - property-typed `s_nexttime` and bounded `s_always [n:m]` now enforce
    finite delayed-cycle progress (non-vacuous strong lowering)
  - property-typed `s_eventually [n:$]` now enforces delayed-cycle existence
    before eventual satisfaction
  - sequence-typed `s_nexttime` and bounded `s_always [n:m]` now add explicit
    strong finite-progress obligations (`expr && eventually(expr)`)

- **Assertion types**:
  - `assert property` - Immediate and concurrent
  - `assume property` - Assumptions
  - `cover property` - Coverage
- **SVA expression parity hardening**:
  - associative-array equality/case-equality (`==`, `!=`, `===`, `!==`) in
    assertion expressions now lower via aggregate element comparison
  - typed associative-array equality now compares key streams (`indices`) and
    value streams (including string-key associative arrays)
  - wildcard associative-array equality/stability lowering no longer emits
    illegal direct size ops on `wildcard_assoc_array` values
  - typed associative-array sampled stability now compares both key streams and
    value streams (including string-key associative arrays)
  - sequence match-item unary `++/--` now supports real local assertion
    variables in addition to integer locals
  - sequence match-item unary `++/--` now also supports `time` local assertion
    variables with timescale-aware increment/decrement semantics
  - sequence match-item display/write system subroutine calls now lower to
    side-effecting Moore display builtins (no longer dropped with a remark)
  - sequence match-item severity system subroutines (`$info/$warning/$error`)
    now lower to side-effecting Moore severity builtins
  - sequence match-item `$fatal` system subroutine now lowers to fatal
    severity + finish side effects
  - sequence match-item monitor/strobe family calls now preserve side effects:
    - `$strobe*` lowers to display-side effect marker
    - `$monitor*` lowers to monitor-side effect marker
    - `$monitoron/$monitoroff` lower to monitor control builtins
  - sequence match-item file-oriented subroutine calls now preserve side
    effects:
    - `$fdisplay*` / `$fwrite*` lower to `moore.builtin.fwrite`
    - `$fstrobe*` lowers to `moore.builtin.fstrobe`
    - `$fmonitor*` lowers to `moore.builtin.fmonitor`
  - sequence match-item file-control subroutine calls now preserve side
    effects:
    - `$fflush` lowers to `moore.builtin.fflush`
    - `$fclose` lowers to `moore.builtin.fclose`
- **Deferred assertions**:
  - `assert/assume/cover final` - Deferred final checks flagged for BMC
- **Sequence event controls (`@seq`)**:
  - Lowered to clocked wait loops with NFA-based sequence matching
  - sv-tests 9.4.2.4 event sequence now supported
- **Clocking `iff` gating**:
  - `@(posedge clk iff cond)` now gates the clocked input with `cond`
- **Event-typed assertion ports**:
  - Timing-control ports now substitute into event controls (e.g., `$past(..., @(e))`)

### Pipeline

```
ImportVerilog → Moore → MooreToCore → LLHD/Verif → SVAToLTL → VerifToSMT → Z3
```

Note: BMC/LEC harnesses use `--ir-llhd` so immediate assertions inside
procedural blocks survive until `StripLLHDProcesses` hoists them.

Key files:
- `lib/Conversion/ImportVerilog/Assertions.cpp` - SVA parsing
- `lib/Conversion/SVAToLTL/SVAToLTL.cpp` - SVA to LTL conversion
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - BMC encoding

## Known Bugs

### Active Bugs

1. **Sampled-value alignment for clocked assertions**
   - Affects: Other suites; basic03.sv now passes in Yosys BMC
   - Symptom: Incorrect timing for $past in clocked context

2. **Strong unary operators treated as weak**
   - Affects: remaining strong unary edge forms
   - Status: property-typed and sequence-typed `s_nexttime` plus bounded
     `s_always [n:m]` fixed for finite-progress obligations; remaining edge
     forms tracked in regular parity backlog

### Fixed Bugs

- External nets fail case not detected in BMC (Yosys `extnets.sv` FAIL)
- Immediate assertions inside `llhd.process` were dropped during canonicalize
- Weak `eventually` now lowers to a trivial true check in VerifToSMT
- Assert/assume/cover final checks now respect `if` enables in LTLToCore + VerifToSMT

- Verilator sequence_named parser issue resolved (trailing comma + sequence syntax)
- Iteration 117: Clear error for sequence event controls (e8052e464)
- Iteration 91: $past assertion type preservation
- Iteration 86: @posedge sensitivity fix

## Test Commands

```bash
# Run sv-tests chapter-16
cd ~/sv-tests && python3 scripts/run_tests.py --tool circt_verilog tests/chapter-16/

# Run sv-tests SVA BMC harness (use rising clocks only for SVA semantics)
cd ~/circt && RISING_CLOCKS_ONLY=1 utils/run_sv_tests_circt_bmc.sh
# Set ALLOW_MULTI_CLOCK=1 for multi-clock designs when needed.
# Set BMC_SMOKE_ONLY=1 to treat any successful BMC run as PASS (pipeline smoke).
# Set FORCE_BMC=1 to run BMC even for parsing-only sv-tests.
# Set NO_PROPERTY_AS_SKIP=1 to classify propertyless designs as SKIP.
# Set INCLUDE_UVM_TAGS=1 to include tests tagged only with `uvm`.
# UVM smoke run example (no solver): UVM_PATH=lib/Runtime/uvm ALLOW_MULTI_CLOCK=1 \
#   BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir INCLUDE_UVM_TAGS=1 \
#   utils/run_sv_tests_circt_bmc.sh

# Run sv-tests LEC harness (smoke pipeline only)
cd ~/circt && LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
  utils/run_sv_tests_circt_lec.sh
# UVM LEC smoke run example:
#   UVM_PATH=lib/Runtime/uvm FORCE_LEC=1 LEC_SMOKE_ONLY=1 INCLUDE_UVM_TAGS=1 \
#   CIRCT_LEC_ARGS=--emit-mlir \
#   utils/run_sv_tests_circt_lec.sh

# Run verilator-verification LEC harness (smoke pipeline only)
cd ~/circt && LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
  utils/run_verilator_verification_circt_lec.sh
# UVM-tagged sv-tests now auto-add `--uvm-path` (defaults to
# `lib/Runtime/uvm`); override with UVM_PATH as needed.

# Run Yosys SVA BMC tests (use rising clocks only for SVA semantics)
cd ~/circt && RISING_CLOCKS_ONLY=1 utils/run_yosys_sva_circt_bmc.sh
# Smoke-only run example (no solver):
#   BMC_SMOKE_ONLY=1 CIRCT_BMC_ARGS=--emit-mlir utils/run_yosys_sva_circt_bmc.sh

# Run yosys SVA LEC harness (smoke pipeline only)
cd ~/circt && LEC_SMOKE_ONLY=1 CIRCT_LEC_ARGS=--emit-mlir \
  utils/run_yosys_sva_circt_lec.sh

# Check specific assertion
./build/bin/circt-verilog --ir-hw test.sv
./build/bin/circt-bmc test.mlir
```

## Contribution Guidelines

1. **Before fixing SVA issues**: Check if Codex agent is already working on it
2. **Test additions**: Add both positive and negative tests
3. **Commit format**: `[SVA] Description`
4. **Update this file** when fixing SVA bugs or adding features

## Related Files

- `lib/Conversion/ImportVerilog/Assertions.cpp`
- `lib/Conversion/SVAToLTL/SVAToLTL.cpp`
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp`
- `lib/Conversion/LTLToCore/LTLToCore.cpp`
- `include/circt/Dialect/LTL/LTLOps.td`
- `include/circt/Dialect/Verif/VerifOps.td`
- `test/Conversion/ImportVerilog/sva-*.sv`
- `test/Conversion/SVAToLTL/*.mlir`
