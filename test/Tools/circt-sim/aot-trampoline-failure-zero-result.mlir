// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-compile %s -o %t.so 2>&1 | FileCheck %s --check-prefix=COMPILE
// RUN: env CIRCT_AOT_INTERCEPT_ALL_UVM=1 circt-sim %s --compiled=%t.so 2>&1 | FileCheck %s --check-prefix=COMPILED

// Regression: compiled->interpreter trampoline dispatch must not return
// uninitialized data when interpreted execution fails.
//
// Prior bug: dispatchTrampoline ignored interpretFuncBody failure and unpacked
// return slots that were never written, producing garbage in compiled mode.
//
// COMPILE: [circt-compile] Generated 1 interpreter trampolines
// COMPILED: r0=1 r1=0 sum=1

module {
  func.func @"uvm_pkg::uvm_demo::maybe_fail"(%do_fail: i1, %x: i32) -> i32 {
    cf.cond_br %do_fail, ^fail, ^ok
  ^ok:
    %one = arith.constant 1 : i32
    %r = arith.addi %x, %one : i32
    return %r : i32
  ^fail:
    // Force an interpreted function-body failure.
    %zero64 = arith.constant 0 : i64
    %bogus = builtin.unrealized_conversion_cast %zero64 : i64 to !llhd.ref<i32>
    %eps = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %bogus, %x after %eps : i32
    %z = arith.constant 0 : i32
    return %z : i32
  }

  // Keep one compiled function alive so AOT codegen succeeds.
  func.func @keep_alive(%x: i32) -> i32 {
    return %x : i32
  }

  llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_demo::maybe_fail"]]
  } : !llvm.array<1 x ptr>

  hw.module @test() {
    %fmt_r0p = sim.fmt.literal "r0="
    %fmt_r1p = sim.fmt.literal " r1="
    %fmt_sp = sim.fmt.literal " sum="
    %fmt_nl = sim.fmt.literal "\0A"

    %c0_i32 = hw.constant 0 : i32
    %c1_i32 = hw.constant 1 : i32
    %false = hw.constant false
    %true = hw.constant true
    %c10_i64 = hw.constant 10000000 : i64

    llhd.process {
      %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
      %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i1, i32) -> i32

      %r0 = func.call_indirect %fn(%false, %c0_i32) : (i1, i32) -> i32
      %r1 = func.call_indirect %fn(%true, %c1_i32) : (i1, i32) -> i32
      %sum = arith.addi %r0, %r1 : i32

      %fr0 = sim.fmt.dec %r0 signed : i32
      %fr1 = sim.fmt.dec %r1 signed : i32
      %fs = sim.fmt.dec %sum signed : i32
      %msg = sim.fmt.concat (%fmt_r0p, %fr0, %fmt_r1p, %fr1, %fmt_sp, %fs, %fmt_nl)
      sim.proc.print %msg
      llhd.halt
    }

    llhd.process {
      %d = llhd.int_to_time %c10_i64
      llhd.wait delay %d, ^done
    ^done:
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
