// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: out=2
//
// COMPILED: out=2

llvm.mlir.global internal @g_evt_i32(0 : i32) : i32

func.func @wait_for_event_i32() -> i32 {
  %ptr = llvm.mlir.addressof @g_evt_i32 : !llvm.ptr
  moore.wait_event {
    %v = llvm.load %ptr : !llvm.ptr -> i32
    %evt = builtin.unrealized_conversion_cast %v : i32 to !moore.i32
    moore.detect_event any %evt : i32
  }
  %two = hw.constant 2 : i32
  return %two : i32
}

func.func @keep_alive(%x: i32) -> i32 {
  return %x : i32
}

hw.module @top() {
  %c1_i64 = hw.constant 1000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %one = hw.constant 1 : i32
  %ptr = llvm.mlir.addressof @g_evt_i32 : !llvm.ptr

  llhd.process {
    %rv0 = func.call @wait_for_event_i32() : () -> i32
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
    llvm.store %one, %ptr : i32, !llvm.ptr
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
