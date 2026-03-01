// RUN: circt-compile -v %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=RUNTIME

// Regression: interpreted func.call_indirect must demote native dispatch when
// arg0 ("this") is lowered as i64 and pointer normalization collapses a
// non-zero payload to null. Native dispatch in that state can dereference a
// null receiver.
//
// COMPILE: [circt-compile] Functions: 2 total, 0 external, 0 rejected, 2 compilable
// COMPILE: [circt-compile] Demoted 1 intercepted functions to trampolines
// COMPILE: [circt-compile] 1 functions + 0 processes ready for codegen
//
// RUNTIME-NOT: PLEASE submit a bug report
// RUNTIME: out=1

func.func @"uvm_pkg::uvm_default_coreservice_t::get_resource_pool"(
    %self: i64) -> i1 {
  %zero = hw.constant 0 : i64
  %isNull = comb.icmp eq %self, %zero : i64
  %one = hw.constant true
  %zeroI1 = hw.constant false
  %ret = comb.mux %isNull, %zeroI1, %one : i1
  return %ret : i1
}

func.func @caller_indirect(%fptr: !llvm.ptr, %self: i64) -> i1 {
  %run = hw.constant true
  cf.cond_br %run, ^live, ^dead
^dead:
  // Dead control op forces trampoline demotion so call_indirect executes in
  // the interpreter and reaches runtime entry-table dispatch.
  sim.pause quiet
  cf.br ^live
^live:
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64) -> i1
  %r = func.call_indirect %fn(%self) : (i64) -> i1
  return %r : i1
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_default_coreservice_t::get_resource_pool"]
  ]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %prefix = sim.fmt.literal "out="
  %nl = sim.fmt.literal "\0A"
  %t10 = hw.constant 10000000 : i64
  // Normalizes to null in native pointer normalization.
  %low = llvm.mlir.constant(12513024 : i64) : i64 // 0x00beef00

  llhd.process {
    %vt = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
    %slot0 = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
    %r = func.call @caller_indirect(%fptr, %low) : (!llvm.ptr, i64) -> i1
    %ri32 = arith.extui %r : i1 to i32
    %vf = sim.fmt.dec %ri32 signed : i32
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
