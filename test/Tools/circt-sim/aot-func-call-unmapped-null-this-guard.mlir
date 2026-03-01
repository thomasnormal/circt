// RUN: circt-compile -v %s -o %t.so
// RUN: env CIRCT_AOT_STATS=1 CIRCT_AOT_ALLOW_UNMAPPED_NATIVE=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: allow-all unmapped native dispatch must not call native UVM-like
// instance methods with a null arg0 (`this`) pointer. Demote these calls to the
// interpreter instead of crashing.
//
// RUNTIME: Compiled function calls:          0
// RUNTIME: Interpreted function calls:       1
// RUNTIME: direct_calls_native:              0
// RUNTIME: direct_calls_interpreted:         1
// RUNTIME: Top interpreted func.call fallback reasons (top 50):
// RUNTIME: uvm_pkg::uvm_component::get_children [pointer-safety=1]
// RUNTIME: safe=1

llvm.mlir.global internal @dummy(0 : i8) : i8

func.func private @"uvm_pkg::uvm_component::get_children"(%this: !llvm.ptr,
                                                           %q: !llvm.ptr) {
  // Native path would segfault on null %this; interpreted fallback handles
  // unknown/null pointers conservatively.
  %v = llvm.load %this : !llvm.ptr -> i8
  llvm.store %v, %q : i8, !llvm.ptr
  return
}

hw.module @top() {
  %one = hw.constant 1 : i64
  %prefix = sim.fmt.literal "safe="
  %nl = sim.fmt.literal "\0A"
  %gp = llvm.mlir.addressof @dummy : !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %ok = hw.constant 1 : i32

  llhd.process {
    func.call @"uvm_pkg::uvm_component::get_children"(%null, %gp) : (!llvm.ptr, !llvm.ptr) -> ()
    %v = sim.fmt.dec %ok signed : i32
    %msg = sim.fmt.concat (%prefix, %v, %nl)
    sim.proc.print %msg
    %d = llhd.int_to_time %one
    llhd.wait delay %d, ^done
  ^done:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
