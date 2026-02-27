// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=1
//
// COMPILED: out=1

llvm.mlir.global internal @g_evt(false) : i1

func.func @wait_for_event_ptr() -> i32 {
  %ptr = llvm.mlir.addressof @g_evt : !llvm.ptr
  moore.wait_event {
    %v = llvm.load %ptr : !llvm.ptr -> i1
    %evt = builtin.unrealized_conversion_cast %v : i1 to !moore.event
    moore.detect_event any %evt : event
  }
  %one = hw.constant 1 : i32
  return %one : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @top() {
  %c1_i64 = hw.constant 1000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %true = hw.constant true
  %ptr = llvm.mlir.addressof @g_evt : !llvm.ptr

  llhd.process {
    %rv0 = func.call @wait_for_event_ptr() : () -> i32
    %rv = func.call @keep_alive(%rv0) : (i32) -> i32
    %prefix = sim.fmt.literal "out="
    %nl = sim.fmt.literal "\0A"
    %fmt = sim.fmt.dec %rv signed : i32
    %all = sim.fmt.concat (%prefix, %fmt, %nl)
    sim.proc.print %all
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c1_i64
    llhd.wait delay %d, ^wake
  ^wake:
    llvm.store %true, %ptr : i1, !llvm.ptr
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %c20_i64
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
