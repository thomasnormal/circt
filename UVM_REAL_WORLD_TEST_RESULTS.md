# UVM Real-World Testbench Testing Results

## Test Date: 2026-01-17

## Overview

This document summarizes the results of testing CIRCT's UVM support against real-world testbenches from the mbit AVIP collection and the UVM core library.

## Test Environment

- **CIRCT Build**: ~/circt/build/bin/circt-verilog
- **UVM Core**: ~/uvm-core (IEEE 1800.2-2020 compliant)
- **Test Subjects**: 9 AVIP implementations from ~/mbit/
  - APB AVIP (77 SV files)
  - AXI4 AVIP
  - AXI4-Lite AVIP
  - AHB AVIP
  - UART AVIP
  - SPI AVIP
  - I2S AVIP
  - I3C AVIP
  - JTAG AVIP

---

## Summary Results

### UVM Core Library (uvm_pkg.sv)

| Metric | Count |
|--------|-------|
| **Errors** | 1 |
| **Warnings** | 3 |
| **Remarks (informational)** | 13 |

**Result**: UVM core library parses with only 1 blocking error.

### AVIP Testbenches

All 9 AVIP testbenches parse successfully with the same single UVM core error. The AVIPs use:
- UVM base classes
- Factory registration macros
- Sequences and sequencers
- Drivers and monitors
- Scoreboards
- Virtual interfaces
- SystemVerilog Assertions
- Coverage groups

---

## Detailed Findings

### 1. CRITICAL ERROR: Incorrect 'this' Pointer During Constructor Call

**Location**: `uvm_heartbeat.svh:109`

**Error Message**:
```
error: receiver class @"uvm_pkg::uvm_heartbeat_callback" is not the same as, or derived from, expected base class "uvm_pkg::uvm_heartbeat"
```

**Code**:
```systemverilog
class uvm_heartbeat extends uvm_object;
  protected uvm_heartbeat_callback m_cb;  // Type: uvm_heartbeat_callback
  protected uvm_component m_cntxt;        // Type: uvm_component

  function new(string name, uvm_component cntxt, uvm_objection objection=null);
    // ... m_cntxt is assigned ...
    m_cb = new({name,"_cb"}, m_cntxt);  // Line 109 - ERROR HERE
    //     ^^^ constructor returns uvm_heartbeat_callback
    //                             ^^^^^^^ should use 'this' (uvm_heartbeat)
  endfunction
endclass

class uvm_heartbeat_callback extends uvm_objection_callback;
  function new(string name, uvm_object target);  // Takes uvm_object
    // ...
  endfunction
endclass
```

**Debug Output Analysis**:
```
visitClassProperty: property 'm_cntxt' declared in "uvm_pkg::uvm_heartbeat",
  current this type = !moore.class<@"uvm_pkg::uvm_heartbeat_callback">,  <-- WRONG!
  need upcast to !moore.class<@"uvm_pkg::uvm_heartbeat">
```

**Root Cause**: The `getImplicitThisRef()` function returns the wrong 'this' pointer during constructor calls that create new objects:

1. Inside `uvm_heartbeat::new()`, the 'this' pointer should be `uvm_heartbeat`
2. When evaluating `m_cntxt` (a property of `uvm_heartbeat`), the code needs `this`
3. BUT `getImplicitThisRef()` returns the type of the NEW object being constructed (`uvm_heartbeat_callback`)
4. Then `visitClassProperty()` tries to upcast `uvm_heartbeat_callback` to `uvm_heartbeat`
5. This fails because `uvm_heartbeat_callback` does NOT extend `uvm_heartbeat`

**The actual comparison being made**:
- WRONG: "Is `uvm_heartbeat_callback` derived from `uvm_heartbeat`?" --> NO (different hierarchies)
- Should be: "Is `uvm_component` derived from `uvm_object`?" --> YES (via inheritance)

**Fix Location**: `/home/thomas-ahle/circt/lib/Conversion/ImportVerilog/Expressions.cpp`

The issue is in how `context.getImplicitThisRef()` is managed during constructor call argument evaluation. When evaluating `m_cntxt` as an argument to the `uvm_heartbeat_callback::new()` constructor, the implicit 'this' should NOT be changed from the enclosing method's receiver.

**Impact**: CRITICAL - Blocks ALL UVM testbenches from compiling

**Priority**: P0 - Must fix for UVM support

---

### 2. Warnings (Non-blocking)

#### 2a. Unknown Escape Sequence
**Location**: `uvm_config_db_implementation.svh:375`
```
warning: unknown character escape sequence '\.' [-Wunknown-escape-code]
```
**Code**: `separator = "\.(" ;`

**Analysis**: This is valid regex syntax used in string matching. The warning is harmless but could be suppressed for regex-like patterns.

**Priority**: P3 - Low (cosmetic)

#### 2b. Unreachable Code
**Location**: `uvm_phase_hopper.svh:394`
```
warning: unreachable code
```
**Code**: `wait_for_objection(UVM_ALL_DROPPED);`

**Analysis**: This is intentional UVM code structure. The warning is technically correct but the code is used for signaling.

**Priority**: P3 - Low (informational)

#### 2c. Missing Top-Level Module
```
warning: no top-level modules found in design [-Wmissing-top]
```

**Analysis**: Expected when compiling library packages without a testbench top module.

**Priority**: P4 - Informational only

---

### 3. Remarks (Informational)

#### 3a. Class Built-in Functions Not Supported
```
remark: Class builtin functions (needed for randomization, constraints, and covergroups) are not yet supported and will be dropped during lowering.
```

**Impact**: Randomization via `randomize()` and constraint blocks will not work at runtime. This affects:
- `rand` and `randc` member randomization
- `constraint` blocks
- `pre_randomize()` / `post_randomize()` callbacks

**Priority**: P1 - High (required for full UVM stimulus generation)

#### 3b. DPI-C Import Stubs
```
remark: DPI-C import 'uvm_hdl_deposit' will use runtime stub (link with MooreRuntime)
remark: DPI-C import 'uvm_hdl_force' will use runtime stub (link with MooreRuntime)
remark: DPI-C import 'uvm_re_compexecfree' will use runtime stub (link with MooreRuntime)
... (12 total DPI functions)
```

**Impact**: DPI functions will use runtime stubs. This is informational and expected behavior - the actual DPI implementation must be provided at link time.

**Priority**: P3 - Working as designed

---

## Features That Work

1. **Package compilation** - UVM packages parse and elaborate correctly
2. **Class definitions** - All UVM base classes are recognized
3. **Single-level inheritance** - Direct class extension works
4. **Factory registration macros** - `uvm_object_utils`, `uvm_component_utils` work
5. **UVM macros** - `uvm_info`, `uvm_error`, `uvm_fatal` work
6. **Virtual interfaces** - Interface references compile
7. **SystemVerilog Assertions** - Properties and assertions parse correctly
8. **Parameterized classes** - Generic class specialization works
9. **DPI-C imports** - Functions are recognized (use runtime stubs)
10. **TLM ports** - Analysis ports and FIFOs compile

---

## Features With Issues

| Feature | Status | Issue |
|---------|--------|-------|
| 'this' pointer scoping in constructor args | BROKEN | P0 - Blocks UVM (see Bug #1) |
| Randomization (`randomize()`) | NOT SUPPORTED | P1 - Needed for stimulus |
| Constraint blocks | NOT SUPPORTED | P1 - Needed for constrained random |
| Covergroups | NOT SUPPORTED | P2 - Needed for coverage |

---

## TOP 3 Most Impactful Gaps to Fix

### 1. Implicit 'this' Reference During Constructor Call Arguments (P0)

**What**: When a constructor call like `m_cb = new(..., m_cntxt)` is processed, the evaluation of `m_cntxt` incorrectly uses the NEW object's type as 'this' instead of the enclosing method's receiver.

**Bug Location**: The issue is likely in how constructor argument expressions are evaluated in `Expressions.cpp`. The `methodReceiverOverride` or similar mechanism is incorrectly setting the 'this' pointer to the new object type before the constructor arguments are evaluated.

**Code Pattern That Fails**:
```systemverilog
class Parent {
  OtherClass obj;
  ParentProperty prop;

  function new();
    obj = new(prop);  // Evaluating 'prop' uses wrong 'this' type
  endfunction
endclass
```

**Exact Bug Location**: `lib/Conversion/ImportVerilog/Expressions.cpp` lines 4059-4067

```cpp
// BUGGY CODE in visit(NewClassExpression):
// Pass the newObj as the implicit this argument of the ctor.
auto savedThis = context.currentThisRef;
context.currentThisRef = newObj;  // BUG: Set BEFORE argument evaluation
auto restoreThis = llvm::make_scope_exit(
    [&] { context.currentThisRef = savedThis; });
// Emit a call to ctor
if (!visitCall(*callConstructor, *subroutine))  // Arguments evaluated HERE
  return {};
```

**Suggested Fix**:
```cpp
// FIX: Don't change currentThisRef until after arguments are evaluated
// Option 1: Move the newObj assignment to after argument conversion
// Option 2: Pass newObj explicitly to visitCall for constructor mode
// Option 3: Save currentThisRef inside visitCall and restore for argument eval

// The visitCall function at lines 2579-2583 has similar logic for default
// arguments - the same pattern should be applied for constructor arguments:
auto savedThis = context.currentThisRef;
if (isDefaultArg)
  context.currentThisRef = methodReceiver;
auto restoreThis =
    llvm::make_scope_exit([&] { context.currentThisRef = savedThis; });
```

**Impact**: Unblocks ALL UVM testbenches

### 2. Multi-Level Class Inheritance Verification (P1, may be working)

**What**: The `isClassDerivedFrom()` function needs to properly traverse multi-level inheritance chains.

**Note**: The current code at lines 5817-5851 appears to have the correct logic to walk up the inheritance chain. However, we cannot fully verify this because Bug #1 prevents us from reaching this code path with the correct types.

**Action**: After fixing Bug #1, retest to confirm multi-level inheritance works for:
- `uvm_component` -> `uvm_report_object` -> `uvm_object` -> `uvm_void`

**Impact**: Required for UVM class hierarchy

### 3. Randomization/Constraint Support (P1)

**What**: The remark "Class builtin functions needed for randomization, constraints, and covergroups are not yet supported" indicates this is a known gap.

**Current State**: Parsing works, but lowering drops the functionality.

**Impact**: Required for constrained random verification - core UVM stimulus generation

---

## Test Commands Reference

```bash
# Test UVM core
cd ~/uvm-core
~/circt/build/bin/circt-verilog --ir-moore -I src src/uvm_pkg.sv 2>&1 | head -200

# Test APB AVIP (full compilation)
cd ~/mbit/apb_avip
~/circt/build/bin/circt-verilog --ir-moore \
  -I ~/uvm-core/src \
  -I src/globals \
  -I src/hvl_top/master \
  -I src/hvl_top/slave \
  -I src/hvl_top/env \
  ~/uvm-core/src/uvm_pkg.sv \
  src/globals/apb_global_pkg.sv \
  src/hdl_top/apb_if/apb_if.sv \
  src/hdl_top/master_agent_bfm/apb_master_driver_bfm.sv \
  src/hdl_top/master_agent_bfm/apb_master_monitor_bfm.sv \
  src/hvl_top/master/apb_master_pkg.sv \
  src/hvl_top/slave/apb_slave_pkg.sv \
  src/hvl_top/env/apb_env_pkg.sv \
  2>&1

# Count errors/warnings
... 2>&1 | grep -c "error:"
... 2>&1 | grep -c "warning:"
```

---

## Conclusion

CIRCT's UVM support is remarkably close to full functionality. The **single blocking issue** is a bug in how the implicit 'this' pointer is managed during constructor call argument evaluation. When processing `m_cb = new(..., m_cntxt)`, the evaluation of `m_cntxt` incorrectly uses the type of the NEW object (`uvm_heartbeat_callback`) instead of the enclosing method's receiver (`uvm_heartbeat`).

This causes a cascading failure where CIRCT tries to upcast `uvm_heartbeat_callback` to `uvm_heartbeat` (which is impossible since they're in different class hierarchies) instead of the intended operation of upcasting `uvm_component` to `uvm_object` (which IS valid).

Once this 'this' pointer scoping bug is fixed, all 9 AVIP testbenches and the UVM core library should compile successfully. The remaining gaps (randomization, constraints, covergroups) are documented as "not yet supported" and have clear paths to implementation.

**Estimated effort to unblock UVM**:
- Bug #1 ('this' pointer scoping): 0.5-1 day (focused fix in constructor argument handling)
- Bug #2 (inheritance chain, if needed): Should work after #1
- Feature #3 (randomization): Larger effort, separate track
