// RUN: circt-sim-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: interpreted -> native func.call with pointer-typed args must
// normalize low-bit-tagged UVM-style object pointers before native entry.
//
// COMPILE: [circt-sim-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-sim-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-sim-compile] 1 functions + 0 processes ready for codegen
//
// RUNTIME: Compiled function calls:          1
// RUNTIME: Interpreted function calls:       0
// RUNTIME: direct_calls_native:              1
// RUNTIME: direct_calls_interpreted:         0
// RUNTIME: out=0{{$}}

llvm.func @malloc(i64) -> !llvm.ptr

func.func @"uvm_pkg::uvm_object::native_tagbit"(%arg0: !llvm.ptr) -> i64 {
  %i = llvm.ptrtoint %arg0 : !llvm.ptr to i64
  %one = llvm.mlir.constant(1 : i64) : i64
  %b = llvm.and %i, %one : i64
  return %b : i64
}

func.func @interp_caller() -> i64 {
  %run = hw.constant true
  cf.cond_br %run, ^live, ^dead
^dead:
  // Dead control op is enough to force trampoline demotion of this function.
  sim.pause quiet
  cf.br ^live
^live:
  %c8 = llvm.mlir.constant(8 : i64) : i64
  %mask1 = llvm.mlir.constant(1 : i64) : i64

  %obj = llvm.call @malloc(%c8) : (i64) -> !llvm.ptr

  // Simulate UVM-style tagged handle bits crossing into native dispatch.
  %objI = llvm.ptrtoint %obj : !llvm.ptr to i64
  %taggedI = llvm.or %objI, %mask1 : i64
  %tagged = llvm.inttoptr %taggedI : i64 to !llvm.ptr

  %r = func.call @"uvm_pkg::uvm_object::native_tagbit"(%tagged) : (!llvm.ptr) -> i64
  return %r : i64
}

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64

  llhd.process {
    %v = func.call @interp_caller() : () -> i64
    %vf = sim.fmt.dec %v signed : i64
    %msg = sim.fmt.concat (%prefix, %vf, %nl)
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
