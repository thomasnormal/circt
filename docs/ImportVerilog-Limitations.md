# ImportVerilog Known Limitations

This document tracks known limitations in the ImportVerilog conversion.

## Task Clocking Events with Module-Level Variables

**Status**: Known Limitation
**Severity**: Low
**Workaround**: Pass clock signals as task arguments

### Description

Tasks that use timing controls (`@(posedge clk)`) referencing module-level variables will fail with a region isolation error:

```systemverilog
module example;
  logic clk;

  task automatic wait_for_clk();
    @(posedge clk);  // ERROR: clk is module-level, not accessible in task
  endtask

  initial begin
    wait_for_clk();
  end
endmodule
```

**Error**:
```
error: 'moore.read' op using value defined outside the region
note: required by region isolation constraints
```

### Root Cause

Tasks are converted to `func.func` operations which have the `IsolatedFromAbove` trait. This means they cannot reference SSA values from their parent scope (the module). When a timing control like `@(posedge clk)` is converted, it generates a `moore.wait_event` operation that contains a `moore.read` of the clock signal. However, the clock signal is a module-level value that isn't accessible inside the task's isolated region.

### Workaround

Pass the clock signal as an argument to the task:

```systemverilog
module example;
  logic clk;

  task automatic wait_for_clk(input logic clock);
    @(posedge clock);  // OK: clock is a task argument
  endtask

  initial begin
    wait_for_clk(clk);
  end
endmodule
```

### Files Affected

- `lib/Conversion/ImportVerilog/TimingControls.cpp` - Event control conversion
- `lib/Conversion/ImportVerilog/Structure.cpp` - Task/function declaration
- `include/circt/Dialect/Moore/MooreOps.td` - WaitEventOp definition

### Potential Fix

To properly support this pattern, we would need to:
1. Detect when timing controls reference module-level variables
2. Automatically capture those variables as task arguments
3. Update all call sites to pass the captured variables

This is a significant refactoring that would require changes to how tasks are declared and called.

### Impact

Low - Most SystemVerilog code passes clock signals explicitly as task arguments rather than relying on implicit module-scope access. This is considered better coding practice and is the recommended pattern in UVM and other verification methodologies.
