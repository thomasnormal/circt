# crun vs xrun: Behavioral Differences

Systematic comparison of circt-sim/crun against Cadence Xcelium/xrun (24.03,
CDNS-UVM-1.1d), based on cross-validation of 168 UVM regression tests.

**Date:** 2026-03-01
**xrun version:** Xcelium 24.03-s001, CDNS-UVM-1.1d
**crun UVM library:** Accellera IEEE 1800.2-2020 v2020.3.1

## Test Results Overview

| Metric | crun | xrun |
|--------|------|------|
| Total UVM tests | 168 | 168 |
| PASS (non-XFAIL) | 145 | 143 |
| XFAIL | 23 | N/A |
| xrun-only failures | 0 | 2 (DAP, recorder — UVM 1.2+ classes) |
| Both fail | 23 | — |

## Genuine crun Strengths

### 1. Forward Reference Resolution in Field Macros

**Affected tests:** 7 (field-int-only, field-enum, field-string-only,
field-string-object, field-array, object-field-automation,
object-copy-independence)

crun (via slang) correctly implements IEEE 1800-2017 Section 8.2 class scope
semantics: all members declared within a class are visible throughout the
entire class body, regardless of declaration order. This means
`uvm_field_int(x, UVM_ALL_ON)` works even when `x` is declared after the
`uvm_object_utils_begin/end` block.

```systemverilog
class my_item extends uvm_object;
  `uvm_object_utils_begin(my_item)
    `uvm_field_int(x, UVM_ALL_ON)   // x declared below — valid per IEEE 1800
  `uvm_object_utils_end
  int x;  // forward-referenced from macro expansion above
endclass
```

**xrun behavior:** `xmvlog: *E,UNDIDN: 'x': undeclared identifier` — xrun
performs a strict textual ordering check in the macro expansion context rather
than respecting full class-scope lookup.

**Spec reference:** IEEE 1800-2017 Section 8.2 says class members are visible
throughout the class scope. The `uvm_field_*` macros expand into method bodies
(`__m_uvm_field_automation`) that are part of the class and therefore have full
class-scope visibility. xrun is overly strict here.

**Impact:** High. Many UVM tutorials and code examples place field macros
before field declarations. This is the most common pattern in user code.

### 2. UVM 1.2+ / IEEE 1800.2-2020 API Support

**Affected tests:** 5+ (coreservice, factory-create-by-name, spell-chkr,
object-print, tree-printer, integ-objection-sequence, dap-basic, recorder)

crun ships the Accellera reference implementation of IEEE 1800.2-2020
(v2020.3.1). xrun bundles CDNS-UVM-1.1d (2014). APIs exclusive to crun:

| API | UVM 1.2+ | UVM 1.1d equivalent |
|-----|----------|---------------------|
| `uvm_coreservice_t::get()` | Singleton access | `uvm_factory::get()` etc. |
| `uvm_set_before_get_dap` | Data Access Policy | (does not exist) |
| `uvm_text_tr_database` | Transaction recording | (different architecture) |
| `printer.print_field_int()` | Renamed method | `printer.print_int()` |
| `seq.set_starting_phase(ph)` | Method access | `seq.starting_phase = ph` |
| `phase.get_objection_count()` | Direct on phase | `phase.get_objection().get_objection_count()` |

**Impact:** High. IEEE 1800.2-2020 is the current standard. Users writing
modern UVM code will find it works out of the box in crun. xrun users can
upgrade their UVM library independently, but the bundled version is outdated.

### 3. TLM Response Routing Correctness

**Affected tests:** 1 (tlm-transport-channel)

crun correctly routes responses through `uvm_tlm_req_rsp_channel`: requests go
to the request FIFO, responses go to the response FIFO. The architecture uses
two separate internal FIFOs with clear routing semantics defined in the UVM
specification.

**xrun behavior:** Delivers incorrect response data from
`uvm_tlm_req_rsp_channel`. This appears to be a bug in the CDNS-UVM-1.1d
library's TLM channel implementation.

**Impact:** Medium. Affects users of bidirectional TLM channels. May have been
fixed in later Xcelium UVM versions.

## Ambiguous Differences

### 4. Fork Scheduling: wait_trigger_data Race Condition

**Affected tests:** 1 (event-data — fixed, original code demonstrated the
difference)

When using `fork/join_any` + `disable fork` with `wait_trigger_data`, the
behavior depends on implementation-defined process scheduling order within a
time step.

```systemverilog
fork
  ev.wait_trigger_data(retrieved);  // thread A
  begin #10ns; ev.trigger(p); end   // thread B
join_any
disable fork;
// Is `retrieved` set or null?
```

The UVM `wait_trigger_data` implementation is explicitly non-atomic:
```systemverilog
virtual task wait_trigger_data(output T data);
  wait_trigger();          // step 1: blocks until trigger
  data = get_trigger_data();  // step 2: reads data (zero-time)
endtask
```

When thread B calls `ev.trigger(p)`, thread A's `wait_trigger()` returns. The
question is whether `data = get_trigger_data()` executes before `join_any`
returns and `disable fork` fires.

| | crun | xrun |
|---|------|------|
| `retrieved` after `disable fork` | non-null (correct data) | null |
| Subsequent `$cast` | succeeds | `*E,TRNULLID` null pointer dereference |

**crun behavior:** Data capture completes atomically — once `wait_trigger()`
returns, the assignment executes without any intervening scheduling point.

**xrun behavior:** `disable fork` kills the waiter between `wait_trigger()`
returning and `get_trigger_data()` executing.

**Spec reference:** IEEE 1800-2017 Section 9.6.1 defines `disable fork` as
terminating "all active child processes." The ordering of concurrent processes
within the same time step is implementation-defined (Section 4.7). Both
behaviors are spec-compliant.

**Practical note:** crun's behavior is arguably more useful because
`wait_trigger_data` was designed as a convenience method that should behave as
if atomic. The workaround is to use `join` (not `join_any`) or call
`wait_trigger()` and `get_trigger_data()` separately with proper guards.

### 5. find_all Component Filtering

**Affected tests:** 1 (component-find-all)

crun's `find_all("*agent*", comps)` returns only user-created components. xrun
may additionally return internal UVM infrastructure components (e.g., internal
`sequencer_base` children).

This is not really filtering by crun — the Accellera reference implementation
simply does not create the internal helper components that CDNS-UVM-1.1d does.
The IEEE 1800.2 spec says `find_all` returns all components matching the
pattern, without distinguishing "internal" from "user" components.

**Impact:** Low. Tests should use `>= N` rather than `== N` checks to be
robust across implementations.

## Known crun Bugs

### 6. OBJTN_ZERO: drop_objection Below Zero Silently Ignored

**Status:** Confirmed bug (under investigation)

When `phase.drop_objection(this)` is called more times than
`phase.raise_objection(this)`, crun silently continues execution. xrun
correctly issues `UVM_FATAL [OBJTN_ZERO]` and terminates.

```systemverilog
phase.raise_objection(this);
phase.drop_objection(this);
phase.drop_objection(this);  // count goes below zero
$display("[AFTER]");          // prints in crun (BUG), not in xrun (correct)
```

The UVM source (`uvm_objection.svh`) explicitly checks for this condition and
calls `uvm_report_fatal("OBJTN_ZERO", ...)`. crun's objection interceptor
either skips this check or absorbs the fatal.

**Impact:** Medium. Affects negative testing and can mask real bugs where
objection management is incorrect.

## Not a Difference (Corrected)

These were initially reported as crun advantages but are actually identical
behavior in both simulators:

| Category | Both simulators... |
|----------|--------------------|
| **ILLCRT** (late component creation) | Correctly terminate with UVM_FATAL |
| **CLDEXT** (duplicate child name) | Correctly terminate with UVM_FATAL |
| **UVM_FATAL** (explicit) | Correctly terminate |

Tests that appeared to show crun "surviving" these conditions were using
`uvm_report_catcher` to demote the fatals — which is standard UVM behavior
that works in both simulators.

## Methodology

All tests are in `test/Tools/crun/uvm-*.sv` and use the standard lit test
format:
```
// RUN: crun %s --top tb_top -v 0 2>&1 | FileCheck %s
// REQUIRES: crun, uvm
```

xrun cross-validation command:
```bash
xrun -uvm -quiet -64 -timescale 1ns/1ps +UVM_NO_RELNOTES <test>.sv -top tb_top
```

The xrun test runner uses unique temp directories per test and a 120-second
timeout. CHECK patterns containing `[circt-sim]` are skipped for xrun
validation since they are simulator-specific.
