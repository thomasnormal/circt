# Iteration 45 - Multi-Step BMC Unrolling (Track C)

## Quick Summary

**Goal**: Enable BMC to verify temporal properties spanning multiple clock cycles.

**Status**: ⚠️ **Analysis Complete, Implementation Deferred**

**Finding**: Current architecture cannot properly handle LTL delay operators (`##N`). All temporal properties are incorrectly treated as trivially true.

## The Problem

```systemverilog
// This assertion is NOT properly verified:
assert property (@(posedge clk) req |-> ##2 ack);
```

Current behavior:
- `ltl.delay %prop, 0` → ✅ `%prop` (correct)
- `ltl.delay %prop, N>0` → ❌ `true` (WRONG - should track obligation)

## Why This Matters

- **UVM testbenches** rely heavily on temporal assertions
- **SystemVerilog import** converts `##N` to `ltl.delay`
- **All temporal properties** are currently vacuously satisfied
- **No counterexamples** can be found for delayed properties

## Workaround (Proven to Work)

Instead of:
```mlir
%delayed_ack = ltl.delay %ack, 1, 0
%prop = ltl.implication %req, %delayed_ack
```

Use explicit registers:
```mlir
%prev_req = seq.compreg %req, %clk
%not_prev_req = comb.xor %prev_req, %true
%prop = comb.or %not_prev_req, %ack
verif.assert %prop
```

See `test/Conversion/VerifToSMT/bmc-manual-multistep.mlir` for working example.

## What Was Done

### Documentation ✅
1. **BMC_MULTISTEP_DESIGN.md** - Full architecture proposal
2. **BMC_ITERATION45_APPROACH.md** - Analysis of approaches
3. **ITERATION45_SUMMARY.md** - Detailed findings
4. **Enhanced code comments** - VerifToSMT.cpp lines 195-255

### Tests ✅
1. **bmc-manual-multistep.mlir** - Manual workaround (working)
2. **bmc-multistep-delay.mlir** - Delay operator tests (shows limitation)
3. **simple-delay-test.mlir** - Basic BMC infrastructure verification

### Code Changes ✅
- Enhanced documentation in `lib/Conversion/VerifToSMT/VerifToSMT.cpp`
- Clear explanation of limitation and workaround
- TODO comment for future implementation

## What's Next (Future Iterations)

To properly implement delay tracking:

1. **Pre-scan** circuit for `ltl.delay` operations
2. **Allocate** delay buffers in `scf.for` iter_args
3. **Thread** buffers through loop iterations
4. **Modify** LTLDelayOpConversion to read from buffers
5. **Assert** mature obligations

Estimated effort: Medium (requires VerifToSMT refactoring)

## Files

### Documentation
- `BMC_MULTISTEP_DESIGN.md` - Architecture design
- `BMC_ITERATION45_APPROACH.md` - Approach analysis
- `ITERATION45_SUMMARY.md` - Detailed summary
- `ITERATION45_README.md` - This file

### Code
- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - Enhanced comments

### Tests
- `test/Conversion/VerifToSMT/bmc-manual-multistep.mlir`
- `test/Conversion/VerifToSMT/bmc-multistep-delay.mlir`
- `test/Tools/circt-bmc/simple-delay-test.mlir`

## Testing

```bash
cd ~/circt/build

# Build tools
ninja circt-opt circt-bmc

# Test manual workaround (should work)
./bin/circt-opt ../test/Conversion/VerifToSMT/bmc-manual-multistep.mlir \
  --externalize-registers \
  --lower-to-bmc="top-module=manual_req_ack bound=5" \
  --convert-hw-to-smt --convert-comb-to-smt \
  --convert-verif-to-smt --reconcile-unrealized-casts

# Test simple counter
./bin/circt-opt ../test/Tools/circt-bmc/simple-delay-test.mlir \
  --externalize-registers \
  --lower-to-bmc="top-module=simple_counter bound=10"
```

## Key Takeaways

1. ✅ **BMC infrastructure works** for single-step and register-based properties
2. ❌ **Temporal delays don't work** - architectural limitation
3. ✅ **Workaround exists** - manual encoding with registers
4. ✅ **Path forward is clear** - detailed design documented
5. ⏭️ **Implementation deferred** - needs proper architectural solution

## Impact Assessment

**Current Users**:
- Can use manual workaround for temporal properties
- Should avoid complex SVA temporal operators
- Single-step properties work fine

**Future Users**:
- Will get proper temporal support in future iteration
- Design is documented and ready for implementation

---

**Iteration**: 45
**Track**: C (BMC Multi-Step Unrolling)
**Date**: 2026-01-17
**Status**: Analysis complete, implementation deferred
