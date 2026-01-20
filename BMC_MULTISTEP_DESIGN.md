# Multi-Step BMC Unrolling Design

## Problem Statement

Current BMC infrastructure can only verify properties at a single time step. Temporal properties like:
```systemverilog
assert property (@(posedge clk) req |-> ##2 ack);
```
Cannot be properly verified because `##2` (2-cycle delay) is converted to a trivial `true`.

## Current Architecture

### 1. Register Externalization (ExternalizeRegisters.cpp)
- Converts `seq.compreg` → module I/O ports
- Each register becomes: `reg_state` (input) + `reg_next` (output)
- Annotates module with `num_regs` and `initial_values` attributes

### 2. BMC Lowering (LowerToBMC.cpp)
- Creates `verif.bmc` operation with three regions:
  - **init**: Initializes clock and state
  - **loop**: Updates clock on each iteration (toggles for pos/neg edges)
  - **circuit**: The actual circuit logic to verify

### 3. VerifToSMT Conversion (VerifToSMT.cpp)
- Converts `verif.bmc` to SMT solver invocation
- Creates `scf.for` loop that iterates up to `bound`
- Each iteration:
  1. Calls circuit function with symbolic inputs
  2. Checks assertions (looking for violations)
  3. Updates register state for next iteration
  4. Declares fresh symbolic inputs for next step

### Current LTL Delay Handling (Lines 201-227)
```cpp
if (delay == 0) {
  rewriter.replaceOp(op, input);  // Pass through
} else {
  rewriter.replaceOpWithNewOp<smt::BoolConstantOp>(op, true);  // WRONG!
}
```

This is fundamentally incorrect for BMC - it doesn't track obligations!

## Proposed Solution: Loop-Carried Delay Tracking

### Key Insight
The BMC `scf.for` loop already carries state between iterations:
- Clock values
- Register states
- Fresh symbolic inputs

We can **extend this to carry delayed property obligations**.

### Design Overview

#### Phase 1: Delay Analysis (Pre-conversion)
Before converting the BMC circuit, scan for all `ltl.delay` operations:
- Identify maximum delay value `max_delay`
- Create a "delay buffer" for each delayed property
- Buffer size = `max_delay`

#### Phase 2: Enhanced Loop-Carried Values
Extend the `scf.for` iter_args to include:
```
iter_args = [
  clocks...,
  register_states...,
  delay_buffer_0[max_delay],  // For each delayed property
  delay_buffer_1[max_delay],
  ...
  wasViolated
]
```

Each buffer is a queue of SMT bool values representing property obligations.

#### Phase 3: Modified Circuit Evaluation
On each iteration `i`:

1. **Shift delay buffers**: Move all obligations forward in time
   ```
   new_buffer[0] = buffer[1]
   new_buffer[1] = buffer[2]
   ...
   new_buffer[delay-2] = buffer[delay-1]
   new_buffer[delay-1] = current_property_value
   ```

2. **Evaluate delayed property**:
   - `ltl.delay %prop, N` → Returns `buffer[N-1]` (the obligation from N steps ago)

3. **Assert mature obligations**:
   - Properties in `buffer[0]` have matured and should be asserted this cycle

### Simplified Implementation (Iteration 45 Goal)

For this iteration, we'll implement a **simpler 2-step unrolling** approach:

#### Approach: Property History Tracking

Instead of general delay buffers, track specific property values across iterations:

```mlir
scf.for %i = 0 to bound step 1
  iter_args(%clk, %regs..., %prev_props..., %violated) {

  // Evaluate circuit at current step
  %current_vals = call @circuit(%clk, %inputs..., %regs...)

  // For delay(prop, 1): check if prev_prop holds
  %delayed_prop = %prev_props[0]  // Value from last iteration

  // Update property history
  %new_prev_props = [%current_prop_value]  // Shift window

  scf.yield %new_clk, %new_regs..., %new_prev_props..., %new_violated
}
```

#### Implementation Steps

1. **Modify `VerifBoundedModelCheckingOpConversion::matchAndRewrite`**:
   - Before creating the for-loop, scan circuit for `ltl.delay` ops
   - For each delay N, add N loop-carried values (initialized to `false`)

2. **Track delay mappings**:
   - Map each `ltl.delay` op → its buffer index in iter_args

3. **Inside for-loop body**:
   - Before calling circuit: Extract previous delay values from iter_args
   - After calling circuit: Update delay values (shift + append current)

4. **Modified `LTLDelayOpConversion`**:
   - `delay(prop, 0)` → `prop` (unchanged)
   - `delay(prop, N>0)` → Look up the corresponding iter_arg value

### Testing Strategy

1. **Unit test**: `ltl.delay %prop, 1, 0` with simple boolean property
2. **Integration test**: `req |-> ##1 ack` implication property
3. **Register test**: Verify delayed properties with stateful circuit
4. **Multi-delay test**: `##2` delay (2-cycle)

### Limitations of Initial Implementation

- Only handles delays in assertions (not in arbitrary circuit logic)
- Maximum delay determined at compile time
- No support for unbounded delays (`##[1:$]`)
- Properties must be in circuit region (not in submodules)

### Future Extensions (Beyond Iteration 45)

1. **General delay infrastructure**: Full delay buffer implementation
2. **LTL.until operator**: Proper temporal until semantics
3. **LTL.eventually**: Track over all future states
4. **Nested delays**: Compositional delay handling
5. **Multiple clocks**: Per-clock delay tracking

## Implementation Files

- `lib/Conversion/VerifToSMT/VerifToSMT.cpp` - Main conversion logic
- `test/Conversion/VerifToSMT/bmc-multistep-delay.mlir` - Test cases

## Success Criteria

1. `ltl.delay %prop, 1, 0` correctly evaluates to previous cycle's property
2. `req |-> ##1 ack` implication can detect violations
3. BMC can prove/disprove 2-step temporal properties
4. Tests pass with bounded model checking

---

**Implementation Status**: Design phase complete, ready for coding.
