// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s
// REQUIRES: slang

// Test coverage for all system calls that should be supported without errors.
// This test verifies that none of the system calls produce "unsupported system
// call" errors during ImportVerilog lowering.

// CHECK-NOT: unsupported system call

// --------------------------------------------------------------------------
// Real/integer conversion functions (IEEE 1800-2017 Section 20.5)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @RealIntConversions
function void RealIntConversions();
  real r;
  int i;
  // $rtoi truncates real to integer, $itor converts int to real.
  // Slang constant-folds these for constant args.
  // CHECK: moore.constant_real
  i = $rtoi(3.14);
  r = $itor(42);
endfunction

// --------------------------------------------------------------------------
// Power function (IEEE 1800-2017 Section 20.8.2)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @PowerFunction
function void PowerFunction();
  real r;
  // $pow(x, y) returns x^y — constant-folded by slang
  // CHECK: moore.constant_real 8.000000e+00
  r = $pow(2.0, 3.0);
endfunction

// --------------------------------------------------------------------------
// $psprintf alias for $sformatf (non-standard but widely supported)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @PSprintfAlias
function void PSprintfAlias();
  string s;
  // $psprintf is an alias for $sformatf — slang constant-folds this
  // CHECK: moore.constant_string "value=42"
  s = $psprintf("value=%0d", 42);
endfunction

// --------------------------------------------------------------------------
// Assertion control tasks (IEEE 1800-2017 Section 20.12)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @AssertionControl
function void AssertionControl();
  // All assertion control tasks should be accepted as no-ops
  // CHECK: return
  $assertcontrol(0);
  $asserton;
  $assertoff;
  $assertkill;
  $assertpasson;
  $assertpassoff;
  $assertfailon;
  $assertfailoff;
  $assertnonvacuouson;
  $assertvacuousoff;
endfunction

// --------------------------------------------------------------------------
// Checkpoint/restart tasks (IEEE 1800-2017 Section 21.8)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @CheckpointRestart
function void CheckpointRestart();
  // CHECK: return
  $save("checkpoint.sav");
  $restart("checkpoint.sav");
  $incsave("incremental.sav");
endfunction

// --------------------------------------------------------------------------
// Debug/PLI tasks (IEEE 1800-2017 Section 21.2, 21.9)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @DebugTasks
function void DebugTasks();
  // CHECK: return
  $stacktrace;
  $showvars;
  $showscopes;
  $list;
endfunction

// --------------------------------------------------------------------------
// Logging control tasks
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @LoggingControl
function void LoggingControl();
  // CHECK: return
  $log;
  $nolog;
  $key;
  $nokey;
endfunction

// --------------------------------------------------------------------------
// SDF annotation (IEEE 1800-2017 Section 30)
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @SDFAnnotation
function void SDFAnnotation();
  // CHECK: return
  $sdf_annotate("timing.sdf");
endfunction

// --------------------------------------------------------------------------
// Query functions (IEEE 1800-2017 Section 20.6)
// These are typically constant-folded by slang for static types.
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @QueryFunctions
function void QueryFunctions();
  int arr[10];
  int i;
  // CHECK: return
  i = $size(arr);
  i = $dimensions(arr);
  i = $unpacked_dimensions(arr);
  i = $increment(arr);
endfunction

// --------------------------------------------------------------------------
// Legacy functions
// --------------------------------------------------------------------------
// CHECK-LABEL: func.func private @LegacyFunctions
function void LegacyFunctions();
  int i;
  // CHECK: return
  i = $reset_count;
  i = $reset_value;
endfunction

module top;
  initial begin
    RealIntConversions();
    PowerFunction();
    PSprintfAlias();
    AssertionControl();
    CheckpointRestart();
    DebugTasks();
    LoggingControl();
    SDFAnnotation();
    QueryFunctions();
    LegacyFunctions();
    $finish;
  end
endmodule
