# CIRCT SVA (SystemVerilog Assertions) Status

This document tracks the status of SVA support in CIRCT, including bugs, missing features, and test coverage.

**Last Updated**: January 23, 2026

## Overview

SVA (SystemVerilog Assertions) is handled by the **Codex agent** for dedicated development.
This document serves as a central tracking point for SVA-related issues.

## Test Suites

### sv-tests Chapter-16 (Assertions)

**Current Status**: 24/53 passing (45.3%)

| Test Category | Status | Notes |
|---------------|--------|-------|
| Property local variables (16.10) | Partial | Basic support works, UVM patterns fail |
| Sequence local variables (16.10) | Partial | Basic support works, UVM patterns fail |
| Sequence subroutines (16.11) | Not supported | Method calls in sequences |
| Property interfaces (16.12) | Not supported | Interface-based properties |
| Multiclock sequences (16.13) | Not supported | Multiple clock domain sequences |
| Assume property (16.14) | Partial | |
| Property disable iff (16.15) | Partial | |
| Expect (16.17) | Partial | |
| Assert final (16.2) | Not supported | |

### Yosys SVA BMC Tests

**Current Status**: 12/14 passing (86%)

| Test | Status | Notes |
|------|--------|-------|
| basic00.sv | PASS | Simple assertion |
| basic01.sv | PASS | Multiple assertions |
| basic02.sv | PASS | Bind assertions |
| basic03.sv | FAIL | $past comparisons, sampled-value alignment |
| basic04.sv | PASS | |
| basic05.sv | PASS | |
| counter.sv | PASS | Counter with assertions |
| extnets.sv | PASS | External nets |
| nested_clk_else.sv | FAIL | Nested clocked assertions with else |
| sva_not.sv | PASS | Negation |
| sva_range.sv | PASS | Range repetition |
| sva_throughout.sv | PASS | Throughout operator |
| sva_value_change_*.sv | PASS | Value change functions |

### verilator-verification

**SVA-related failures**: 6 tests (tracked separately)

## Missing Features (Priority Order)

### High Priority

1. **Sequence event controls (`@seq`)** - sv-tests 9.4.2.4
   - Error: "sequence/property event controls are not yet supported"
   - Needed for: Complex synchronization patterns
   - Files: `lib/Conversion/ImportVerilog/TimingControls.cpp`

2. **Property local variables in UVM patterns**
   - Tests: 16.10--*-uvm.sv
   - Issue: Complex property variable scoping

3. **Assert final**
   - Test: 16.2--assert-final-uvm.sv
   - SystemVerilog `assert final` construct

### Medium Priority

4. **Multiclock sequences (16.13)**
   - Multiple clock domain assertions
   - Requires: Clock domain crossing in LTL

5. **Property interfaces (16.12)**
   - Interface-based property definitions
   - Needed for: Modular assertion libraries

6. **Sequence subroutines (16.11)**
   - Method calls within sequences
   - Needed for: Complex sequence patterns

7. **Nested clocked assertions with else**
   - Yosys test: nested_clk_else.sv
   - Issue: Clocking block interaction with else clause

### Low Priority

8. **$past with delays in clocked context**
   - Yosys test: basic03.sv
   - Sampled-value alignment issues

9. **Property disable iff edge cases**
   - Test: 16.15--property-disable-iff-fail.sv

## Implementation Status

### Completed Features

- **SVA assertion functions** (Iteration 28-29):
  - `$sampled` - Current value sampling
  - `$past` - Previous value access (basic)
  - `$changed` - Value change detection
  - `$stable` - Value stability check
  - `$rose` - Rising edge detection
  - `$fell` - Falling edge detection

- **LTL operators**:
  - `|->` - Overlapping implication
  - `|=>` - Non-overlapping implication
  - `##N` - Fixed delay
  - `##[n:m]` - Range delay
  - `[*N]` - Fixed repetition
  - `[*n:m]` - Range repetition
  - `throughout` - Throughout operator

- **Assertion types**:
  - `assert property` - Immediate and concurrent
  - `assume property` - Assumptions
  - `cover property` - Coverage

### Pipeline

```
ImportVerilog → Moore → MooreToCore → LLHD/Verif → SVAToLTL → VerifToSMT → Z3
```

Key files:
- `lib/Conversion/ImportVerilog/Assertions.cpp` - SVA parsing
- `lib/Conversion/SVAToLTL/SVAToLTL.cpp` - SVA to LTL conversion
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - BMC encoding

## Known Bugs

### Active Bugs

1. **Sampled-value alignment for clocked assertions**
   - Affects: basic03.sv, $past comparisons
   - Symptom: Incorrect timing for $past in clocked context

2. **verif.clocked_assert lowering incomplete**
   - Affects: Some nested assertions
   - Workaround: Use simple assert property

### Fixed Bugs

- Iteration 117: Clear error for sequence event controls (e8052e464)
- Iteration 91: $past assertion type preservation
- Iteration 86: @posedge sensitivity fix

## Test Commands

```bash
# Run sv-tests chapter-16
cd ~/sv-tests && python3 scripts/run_tests.py --tool circt_verilog tests/chapter-16/

# Run Yosys SVA BMC tests
cd ~/circt && utils/run_yosys_sva_circt_bmc.sh

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
