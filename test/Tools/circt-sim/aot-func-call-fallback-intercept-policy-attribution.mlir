// RUN: circt-compile %s -o %t.so
// RUN: env CIRCT_AOT_STATS=1 circt-sim %s --top top --compiled=%t.so 2>&1 | FileCheck %s

// Repro: this callee is compiled but intentionally excluded from native
// func.call dispatch by UVM reporting interception policy.
// The fallback attribution should report intercept-policy (not generic
// no-native), so perf triage can distinguish policy vs true missing native ptr.

module {
  // Keep one non-intercepted function so circt-compile emits a .so.
  func.func private @dummy_compiled(%x: i32) -> i32 {
    %c1 = arith.constant 1 : i32
    %y = arith.addi %x, %c1 : i32
    return %y : i32
  }

  func.func private @"uvm_pkg::uvm_object::new"(
      %this: !llvm.ptr) -> i32 {
    %c42 = arith.constant 42 : i32
    return %c42 : i32
  }

  hw.module @top() {
    %fmtPrefix = sim.fmt.literal "out="
    %fmtNl = sim.fmt.literal "\0A"
    llhd.process {
      %one = llvm.mlir.constant(1 : i64) : i64
      %this = llvm.alloca %one x i8 : (i64) -> !llvm.ptr
      %r = func.call @"uvm_pkg::uvm_object::new"(%this) :
          (!llvm.ptr) -> i32
      %fmtDec = sim.fmt.dec %r signed : i32
      %fmtOut = sim.fmt.concat (%fmtPrefix, %fmtDec, %fmtNl)
      sim.proc.print %fmtOut
      llhd.halt
    }
    hw.output
  }
}

// CHECK: Compiled function calls:          0
// CHECK: Top interpreted func.call fallback reasons (top 50):
// CHECK: uvm_pkg::uvm_object::new [intercept-policy=1]
// CHECK: out=42
