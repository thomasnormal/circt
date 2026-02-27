// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: circt-sim %s | FileCheck %s --check-prefix=SIM
// RUN: circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: native func.call can return a host pointer that is later consumed
// by an interpreted UVM-intercepted callee. The pointer must be registered as a
// native block so interpreted loads read host memory instead of falling back to
// unknown/zero.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// SIM: class_id=620
// COMPILED: class_id=620

llvm.func @malloc(i64) -> !llvm.ptr

func.func private @mk_obj() -> !llvm.ptr {
  %size = llvm.mlir.constant(16 : i64) : i64
  %cid = llvm.mlir.constant(620 : i32) : i32
  %p = llvm.call @malloc(%size) : (i64) -> !llvm.ptr
  llvm.store %cid, %p : i32, !llvm.ptr
  return %p : !llvm.ptr
}

func.func private @"uvm_pkg::uvm_phase::get_domain"(%arg0: !llvm.ptr) -> i32 {
  %cid = llvm.load %arg0 : !llvm.ptr -> i32
  return %cid : i32
}

hw.module @top() {
  %prefix = sim.fmt.literal "class_id="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %p = func.call @mk_obj() : () -> !llvm.ptr
    %cid = func.call @"uvm_pkg::uvm_phase::get_domain"(%p) : (!llvm.ptr) -> i32
    %v = sim.fmt.dec %cid signed : i32
    %msg = sim.fmt.concat (%prefix, %v, %nl)
    sim.proc.print %msg
    llhd.halt
  }

  llhd.process {
    %d = llhd.int_to_time %t10
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
