// RUN: circt-opt --lower-to-bmc="top-module=top bound=1" %s | FileCheck %s

// CHECK-LABEL: func.func @helper
// CHECK: llvm.alloca
// CHECK: llvm.store
// CHECK: llvm.load
// CHECK-NOT: llhd.
// CHECK: verif.bmc

module {
  func.func @helper() -> i32 {
    %c0 = hw.constant 0 : i32
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %sig = llhd.sig %c0 : i32
    llhd.drv %sig, %c0 after %t0 : i32
    %v = llhd.prb %sig : i32
    %now = llhd.current_time
    %tint = llhd.time_to_int %now
    %tshort = arith.trunci %tint : i64 to i32
    %sum = comb.add %v, %tshort : i32
    return %sum : i32
  }

  func.func @ref_helper(%arg0: !llvm.ptr) -> i32 {
    %c0 = hw.constant 0 : i32
    %t0 = llhd.constant_time <0ns, 0d, 1e>
    %ref = builtin.unrealized_conversion_cast %arg0 : !llvm.ptr to !llhd.ref<i32>
    %v = llhd.prb %ref : i32
    llhd.drv %ref, %c0 after %t0 : i32
    return %v : i32
  }

  hw.module @top() attributes {num_regs = 0 : i32, initial_values = []} {
    hw.output
  }
}
