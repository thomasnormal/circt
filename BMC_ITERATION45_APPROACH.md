# Iteration 45: Simplified BMC Multi-Step Approach

## Problem
The current architecture makes it hard to implement general delay tracking because:
1. Circuit is extracted into a function before delay ops are converted
2. LTLDelayOp conversion happens in isolation, without loop context
3. Functions can't access scf.for iter_args

## Simplified Approach for Iteration 45

Instead of implementing full delay tracking infrastructure, we'll demonstrate the **concept** with a working proof-of-concept.

### Step 1: Document Current Limitations

Create clear documentation explaining:
- Why `ltl.delay N>0` currently returns `true` (trivial)
- What would be needed for proper multi-step BMC
- Architecture challenges

### Step 2: Implement Delay Analysis Pass (Optional)

A pre-processing pass that:
- Scans for `ltl.delay` operations
- Reports delays to user
- Suggests workarounds

### Step 3: Manual Multi-Step Test

Create a test that **manually** implements 2-step tracking:
- Use registers to store previous cycle values
- Manually implement delay logic
- Verify BMC can check it

This proves the BMC infrastructure *can* handle multi-step properties when properly encoded.

### Step 4: Enhanced Documentation

Update VerifToSMT.cpp comments to explain:
```cpp
// TODO(Iteration 45+): Proper delay tracking requires:
// 1. Scanning circuit for ltl.delay ops before function extraction
// 2. Adding delay_buffer[] to circuit arguments
// 3. Threading delay values through scf.for iter_args
// 4. Modifying LTLDelayOpConversion to use buffer values
// Currently, delays > 0 are treated as trivially true.
```

## Concrete Deliverables

1. ✅ **BMC_MULTISTEP_DESIGN.md** - Architecture document
2. ✅ **bmc-multistep-delay.mlir** - Test file (will show current limitations)
3. ⏭️ **Manual workaround example** - Show how to encode delays manually
4. ⏭️ **Documentation updates** - Enhanced comments in code

## Alternative: Prototype Implementation

If we want actual working code, we can implement a **limited prototype**:

### Constraints
- Only support **one** `ltl.delay` op in the entire circuit
- Only support `delay = 1` (next cycle)
- Hard-code the tracking logic

### Implementation
1. Detect single `ltl.delay %prop, 1, 0` in circuit
2. Add `%prev_prop: i1` to circuit arguments
3. Add loop iter_arg for previous property value
4. Convert `ltl.delay %prop, 1, 0` → `%prev_prop` (argument)
5. Update iter_arg with current prop value

This is hacky but:
- ✅ Actually works for simple cases
- ✅ Demonstrates feasibility
- ✅ Can be tested
- ❌ Not generalizable
- ❌ Hard to maintain

## Recommendation

For Iteration 45, focus on:
1. **Documentation** - Explain the problem clearly
2. **Manual example** - Show it *can* be done
3. **Test infrastructure** - Prepare for future implementation

Then in Iteration 46+, implement the full solution with proper architecture.

## Manual Example: req |-> ##1 ack

```mlir
// Instead of using ltl.delay, manually track using registers
hw.module @manual_delay(
  in %clk: !seq.clock,
  in %req: i1,
  in %ack: i1,
  in %prev_req_state: i1,  // Register stores previous req
  out prev_req_next: i1
) {
  // Property: if req was high last cycle, ack must be high this cycle
  %not_prev_req = comb.xor %prev_req_state, %true
  %prop = comb.or %not_prev_req, %ack  // !prev_req || ack
  verif.assert %prop : i1

  // Store current req for next cycle
  hw.output %req : i1
}
```

This can be verified by BMC today, proving multi-step verification is possible.

---

**Recommendation**: Focus on documentation and manual examples for Iteration 45, implement proper infrastructure in Iteration 46.
