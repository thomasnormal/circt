# Assertion Test Failures - Complete Analysis Report

## Executive Summary

Two related assertion tests are failing:
1. **syscall-asserton.sv**: Clocked assertions ignore `$assertoff`/`$asserton` syscalls
2. **procedural-assert.sv**: Failed procedural assertions don't print error messages

## Test 1: syscall-asserton.sv

### Test Code
```systemverilog
module top;
  reg clk = 0;
  reg a = 0;

  always @(posedge clk) begin
    assert (a == 1) else $display("ASSERT_FAILED");
  end

  initial begin
    // Disable assertions
    $assertoff;
    #1 clk = 1; #1 clk = 0;
    // No failure expected here

    // Re-enable assertions
    $asserton;
    #1 clk = 1; #1 clk = 0;
    // CHECK: ASSERT_FAILED
    // CHECK: done
    $display("done");
    $finish;
  end
endmodule
```

### Expected Behavior
1. `$assertoff` disables assertion checking
2. First clock pulse: assertion should NOT fire (a=0, assertion disabled)
3. `$asserton` re-enables assertion checking
4. Second clock pulse: assertion should fire and print "ASSERT_FAILED"

### Actual Behavior
- Assertion might fire at both clock pulses (or not fire at all)
- Output missing "ASSERT_FAILED" message at the right time

### Root Cause

File: `/home/thomas-ahle/circt/lib/Conversion/ImportVerilog/Statements.cpp`

**Issue**: The `ConcurrentAssertionStatement` visitor (starting line 2433) does NOT check or gate assertions with `readProceduralAssertionsEnabled()`.

Compare:
- **ImmediateAssertionStatement** (line 2328): `auto assertionsEnabled = readProceduralAssertionsEnabled();` ✓
- **ConcurrentAssertionStatement** (line 2433): No such check ✗

The concurrent assertion handler includes enable conditions for:
- Guard conditions (line 2627)
- Disable-iff conditions (line 2635)  
- But NOT procedural assertions enabled flag!

### Generated MLIR (Incorrect)
The current MLIR for the assertion in the always block contains:
```mlir
verif.ClockedAssertOp(property, clock, enable, ...)
```

Where `enable` is composed of guard + disable_iff, but NOT the procedural assertions flag.

### Expected MLIR
The `enable` should be something like:
```mlir
enable = guard AND disable_iff_enable AND procedural_assertions_enabled
```

### Fix Details

**Location**: `Statements.cpp`, lines 2620-2680 in `visit(const slang::ast::ConcurrentAssertionStatement &stmt)`

**Where**: Around line 2620 where enable conditions are being combined:
```cpp
if (context.currentAssertionGuard) {
  // ... guard handling ...
}
if (disableIffEnable) {
  // ... disable_iff handling ...
}
// NEW CODE NEEDED HERE
if (enable && !enable.getType().isInteger(1)) {
```

**What to add**:
```cpp
// Gate concurrent assertions with procedural assertions enabled
auto assertionsEnabled = readProceduralAssertionsEnabled();
if (!assertionsEnabled)
  return failure();
assertionsEnabled = context.convertToI1(assertionsEnabled);
if (!assertionsEnabled)
  return failure();
enable = enable ? arith::AndIOp::create(builder, loc, enable, assertionsEnabled)
                : assertionsEnabled;
```

This matches the pattern used for immediate assertions (line 2375-2377 in Statements.cpp).

---

## Test 2: procedural-assert.sv

### Test Code
```systemverilog
module top;
  int x;

  initial begin
    x = 5;
    // This assertion passes - execution continues
    assert(x == 5);
    // CHECK-DAG: PASS: after passing assert
    $display("PASS: after passing assert");

    // This assertion fails - execution should still continue
    // CHECK-DAG: Assertion failed
    assert(x == 99);
    // CHECK-DAG: PASS: after failing assert
    $display("PASS: after failing assert");

    $finish;
  end
endmodule
```

### Expected Behavior
1. First assertion `assert(x == 5)`: passes, continues
2. Print "PASS: after passing assert"
3. Second assertion `assert(x == 99)`: **FAILS**, prints "Assertion failed", **continues**
4. Print "PASS: after failing assert"

### Actual Behavior
- Both "PASS" messages are printed (assertions continue execution) ✓
- But "Assertion failed" message is NOT printed ✗

### Evidence

**MLIR Generated** (correct):
```mlir
%10 = comb.icmp eq %9, %c99_i32 : i32  ;; cond = (x == 99)
%11 = llvm.load %2 : !llvm.ptr -> i1   ;; load proc_assertions_enabled
%12 = comb.xor %11, %true : i1         ;; NOT proc_assertions_enabled
%13 = comb.or %10, %12 : i1            ;; gatedCond = cond OR NOT enabled
verif.assert %13 label "" : i1         ;; assert gatedCond
```

Logic:
- x=5 (from signal drive)
- `%10 = (5 == 99)` = false
- `%11 = true` (assertions enabled)
- `%12 = NOT true` = false
- `%13 = false OR false` = **false**
- Should assert false, which should trigger failure message

**Interpreter Code** (line 12220, `LLHDProcessInterpreter.cpp`):
```cpp
if (auto assertOp = dyn_cast<verif::AssertOp>(op)) {
  InterpretedValue cond = getValue(procId, assertOp.getProperty());
  if (!cond.isX() && cond.getAPInt().isZero()) {
    maybeTraceImmediateAssertionFailed(label, assertOp.getLoc());
  }
  return success();
}
```

This should:
1. Get the value of `%13`
2. Check if it's zero (false)
3. Print "Assertion failed"

But it's not printing, which means either:
- `cond.isX()` returns true (value is unknown)
- `cond.getAPInt().isZero()` returns false (value is not zero)

### Root Cause Hypothesis

The `getValue(procId, assertOp.getProperty())` call is not correctly evaluating the assertion condition value. Possible reasons:

1. **Signal probe timing**: `%9 = llhd.prb %x` might return wrong value
   - Signal drives happen with `after` delay
   - Probes immediately in same clock cycle might see old value

2. **Combinational operation evaluation**: The icmp/xor/or operations might not be evaluated
   - getValue might only handle certain operation types
   - Might return X for unsupported operations

3. **Value cache issue**: getValue might be caching stale values

4. **Operand resolution**: The getValue call might not recursively evaluate operands

### Investigation Steps

To debug this, add logging to `LLHDProcessInterpreter.cpp` around line 12220:
```cpp
if (auto assertOp = dyn_cast<verif::AssertOp>(op)) {
  InterpretedValue cond = getValue(procId, assertOp.getProperty());
  llvm::errs() << "[DEBUG] assertion property: " << assertOp.getProperty() << "\n";
  llvm::errs() << "[DEBUG] cond value: " << cond.isX() << " zero: " 
               << (cond.isX() ? -1 : cond.getAPInt().getZExtValue()) << "\n";
  if (!cond.isX() && cond.getAPInt().isZero()) {
    maybeTraceImmediateAssertionFailed(label, assertOp.getLoc());
  }
  return success();
}
```

Then run the test and observe what values are returned.

### Expected Fix

Likely one of:
1. Ensure signal probes execute at correct time relative to drives
2. Extend `getValue()` to handle more operation types (icmp, xor, or, etc.)
3. Ensure `getValue()` recursively evaluates all operand values
4. Fix the timing so combinational operations evaluate with correct operand values

---

## File Locations

### Test Files
- `/home/thomas-ahle/circt/test/Tools/circt-sim/syscall-asserton.sv`
- `/home/thomas-ahle/circt/test/Tools/circt-sim/procedural-assert.sv`

### Code Files to Fix
- `/home/thomas-ahle/circt/lib/Conversion/ImportVerilog/Statements.cpp` (Test 1)
- `/home/thomas-ahle/circt/tools/circt-sim/LLHDProcessInterpreter.cpp` (Test 2)
- `/home/thomas-ahle/circt/tools/circt-sim/LLHDProcessInterpreterTrace.cpp` (Test 2 debugging)

---

## Test Execution

Run these tests with:
```bash
cd /home/thomas-ahle/circt/build-test
python3.9 /home/thomas-ahle/circt/llvm/build/bin/llvm-lit -v \
  test/Tools/circt-sim/syscall-asserton.sv \
  test/Tools/circt-sim/procedural-assert.sv
```

Expected output after fixes:
```
PASS: CIRCT :: Tools/circt-sim/syscall-asserton.sv
PASS: CIRCT :: Tools/circt-sim/procedural-assert.sv
```
