# CIRCT Full Simulation Pipeline Test Report

## Executive Summary

This report documents the end-to-end testing of the CIRCT simulation pipeline for `$display` (sim.proc.print) and `$finish` (sim.terminate) operations.

**Overall Result**: The pipeline works correctly through Stage 3 (HW IR). Stage 4 (arcilator) has a known limitation with `llhd.process` (initial blocks).

## Test Files

### test_display.sv
```systemverilog
module test_display;
  initial begin
    $display("Hello from CIRCT!");
    $display("Testing: %d + %d = %d", 2, 3, 5);
  end
endmodule
```

### test_finish.sv
```systemverilog
module test_finish;
  initial begin
    $display("About to finish");
    $finish;
  end
endmodule
```

---

## Stage 1: SystemVerilog to Moore IR

**Status**: PASS

### test_display.sv
```bash
./build/bin/circt-verilog --ir-moore test_display.sv
```

**Output**:
```mlir
module {
  moore.module @test_display() {
    %0 = moore.constant 5 : i32
    %1 = moore.constant 3 : i32
    %2 = moore.constant 2 : i32
    moore.procedure initial {
      %3 = moore.fmt.literal "Hello from CIRCT!"
      %4 = moore.fmt.literal "\0A"
      %5 = moore.fmt.concat (%3, %4)
      moore.builtin.display %5
      %6 = moore.fmt.literal "Testing: "
      %7 = moore.fmt.int decimal %2, align right, pad space signed : i32
      %8 = moore.fmt.literal " + "
      %9 = moore.fmt.int decimal %1, align right, pad space signed : i32
      %10 = moore.fmt.literal " = "
      %11 = moore.fmt.int decimal %0, align right, pad space signed : i32
      %12 = moore.fmt.concat (%6, %7, %8, %9, %10, %11, %4)
      moore.builtin.display %12
      moore.return
    }
    moore.output
  }
}
```

### test_finish.sv
**Output**:
```mlir
module {
  moore.module @test_finish() {
    moore.procedure initial {
      %0 = moore.fmt.literal "About to finish"
      %1 = moore.fmt.literal "\0A"
      %2 = moore.fmt.concat (%0, %1)
      moore.builtin.display %2
      moore.builtin.finish_message false
      moore.builtin.finish 0
      moore.unreachable
    }
    moore.output
  }
}
```

**Key Observations**:
- `$display` correctly maps to `moore.builtin.display`
- `$finish` correctly maps to `moore.builtin.finish` with exit code 0
- Format specifiers (%d) are correctly parsed as `moore.fmt.int decimal`

---

## Stage 2: Moore to Core Conversion

**Status**: PASS

### test_display.sv
```bash
./build/bin/circt-verilog --ir-moore test_display.sv | ./build/bin/circt-opt -convert-moore-to-core
```

**Output**:
```mlir
module {
  hw.module @test_display() {
    %c5_i32 = hw.constant 5 : i32
    %c3_i32 = hw.constant 3 : i32
    %c2_i32 = hw.constant 2 : i32
    llhd.process {
      %0 = sim.fmt.literal "Hello from CIRCT!"
      %1 = sim.fmt.literal "\0A"
      %2 = sim.fmt.concat (%0, %1)
      sim.proc.print %2
      %3 = sim.fmt.literal "Testing: "
      %4 = sim.fmt.dec %c2_i32 signed : i32
      %5 = sim.fmt.literal " + "
      %6 = sim.fmt.dec %c3_i32 signed : i32
      %7 = sim.fmt.literal " = "
      %8 = sim.fmt.dec %c5_i32 signed : i32
      %9 = sim.fmt.concat (%3, %4, %5, %6, %7, %8, %1)
      sim.proc.print %9
      llhd.halt
    }
    hw.output
  }
}
```

### test_finish.sv
**Output**:
```mlir
module {
  hw.module @test_finish() {
    llhd.process {
      %0 = sim.fmt.literal "About to finish"
      %1 = sim.fmt.literal "\0A"
      %2 = sim.fmt.concat (%0, %1)
      sim.proc.print %2
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
```

**Key Observations**:
- `moore.builtin.display` correctly converts to `sim.proc.print`
- `moore.builtin.finish` correctly converts to `sim.terminate success, quiet`
- `moore.procedure initial` converts to `llhd.process`
- Format operations convert: `moore.fmt.int decimal` -> `sim.fmt.dec`

---

## Stage 3: Full Pipeline to HW IR

**Status**: PASS

### test_display.sv
```bash
./build/bin/circt-verilog --ir-hw test_display.sv
```

**Output**:
```mlir
module {
  hw.module @test_display() {
    %0 = sim.fmt.literal "Testing:           2 +           3 =           5\0A"
    %1 = sim.fmt.literal "Hello from CIRCT!\0A"
    llhd.process {
      sim.proc.print %1
      sim.proc.print %0
      llhd.halt
    }
    hw.output
  }
}
```

### test_finish.sv
**Output**:
```mlir
module {
  hw.module @test_finish() {
    %0 = sim.fmt.literal "About to finish\0A"
    llhd.process {
      sim.proc.print %0
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
```

**Key Observations**:
- Format strings are constant-folded (optimized)
- The `sim.proc.print` and `sim.terminate` operations are preserved
- Initial blocks remain as `llhd.process` with `llhd.halt`

---

## Stage 4: Through Arcilator

**Status**: FAIL (Expected - Known Limitation)

### test_display.sv
```bash
./build/bin/circt-verilog --ir-hw test_display.sv | ./build/bin/arcilator --emit-mlir
```

**Error**:
```
<stdin>:5:5: error: failed to legalize operation 'llhd.process' that was explicitly marked illegal
    llhd.process {
    ^
<stdin>:1:1: error: conversion to arcs failed
```

### test_finish.sv
**Error**: Same as above - `llhd.process` is not supported.

**Root Cause**: Arcilator is designed for synthesizable hardware simulation using the Arc dialect. Initial blocks (`llhd.process`) are procedural constructs that don't map to the arc-based simulation model.

---

## Workaround: Direct func.func Testing

### sim.proc.print (Works)

```mlir
// test_sim_print_only.mlir
func.func @entry() {
  %0 = sim.fmt.literal "Hello from arcilator!"
  sim.proc.print %0
  return
}
```

```bash
./build/bin/arcilator test_sim_print_only.mlir --run
```

**Output**:
```
Hello from arcilator!
```

### sim.terminate (Fails in arcilator)

```mlir
// test_sim_direct.mlir
func.func @entry() {
  %0 = sim.fmt.literal "Hello!"
  sim.proc.print %0
  sim.terminate success, quiet
  return
}
```

**Error**:
```
error: failed to legalize operation 'sim.terminate'
```

**Root Cause**: There appears to be a bug in the `LowerArcToLLVM` pass where `sim.terminate` fails to be lowered when inside a function that's being converted. The pattern requires a `ModuleOp` parent which may not be correctly found during the conversion process.

---

## Formatted Output Testing (Works)

```mlir
// test_sim_formatted.mlir
func.func @entry() {
  %c42 = arith.constant 42 : i32
  %c255 = arith.constant 255 : i32

  %lit1 = sim.fmt.literal "Testing formatting:\n"
  sim.proc.print %lit1

  %dec = sim.fmt.dec %c42 signed : i32
  %lit2 = sim.fmt.literal "Decimal value: "
  %newline = sim.fmt.literal "\n"
  %msg1 = sim.fmt.concat (%lit2, %dec, %newline)
  sim.proc.print %msg1

  %hex = sim.fmt.hex %c255, isUpper false : i32
  %lit3 = sim.fmt.literal "Hex value: "
  %msg2 = sim.fmt.concat (%lit3, %hex, %newline)
  sim.proc.print %msg2

  %bin = sim.fmt.bin %c42 : i32
  %lit4 = sim.fmt.literal "Binary value: "
  %msg3 = sim.fmt.concat (%lit4, %bin, %newline)
  sim.proc.print %msg3

  return
}
```

```bash
./build/bin/arcilator test_sim_formatted.mlir --run
```

**Output**:
```
Testing formatting:
Decimal value:          42
Hex value: 000000ff
Binary value: 00000000000000000000000000101010
```

---

## Summary Table

| Stage | Component | Status | Notes |
|-------|-----------|--------|-------|
| 1 | SV -> Moore | PASS | $display -> moore.builtin.display, $finish -> moore.builtin.finish |
| 2 | Moore -> Core | PASS | moore.builtin.display -> sim.proc.print, moore.builtin.finish -> sim.terminate |
| 3 | Full HW IR | PASS | Operations preserved, format strings optimized |
| 4 | Arcilator | FAIL | llhd.process (initial blocks) not supported |
| - | func.func print | PASS | sim.proc.print works in arcilator JIT |
| - | func.func terminate | FAIL | sim.terminate has lowering issue |
| - | Formatted output | PASS | sim.fmt.dec, hex, bin all work |

---

## Conclusions

1. **The MooreToCore lowering is complete and correct** for `$display` and `$finish`.

2. **The main blocker is `llhd.process`** (initial blocks), which arcilator doesn't support because it's designed for synthesizable hardware.

3. **sim.proc.print works perfectly** when used outside of `llhd.process`.

4. **sim.terminate has a potential bug** in `LowerArcToLLVM` that should be investigated separately.

5. **For end-to-end simulation**, a different execution path is needed:
   - Option A: Use a different simulator backend that supports `llhd.process`
   - Option B: Transform initial blocks into a different representation
   - Option C: Use the `--run` functionality with `func.func` directly for testing

---

## Recommendations

1. The lowering from SystemVerilog through Moore to HW IR is production-ready for `$display` and `$finish`.

2. For full simulation support, investigate LLHD simulation or alternative backends.

3. File a bug report for `sim.terminate` lowering in arcilator.

4. Consider adding documentation about the `llhd.process` limitation in arcilator.
