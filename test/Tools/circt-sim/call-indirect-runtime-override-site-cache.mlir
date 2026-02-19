// RUN: CIRCT_SIM_TRACE_CALL_INDIRECT_SITE_CACHE=1 circt-sim %s 2>&1 | FileCheck %s

// Regression: cache static func.call_indirect vtable-slot extraction per call
// site so repeated dispatches avoid re-tracing the SSA chain every time.

module {
  func.func private @"animal::legs"(%self: !llvm.ptr) -> i32 {
    %c2 = arith.constant 2 : i32
    return %c2 : i32
  }

  llvm.mlir.global internal @"animal::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"animal::legs"]]
  } : !llvm.array<1 x ptr>

  hw.module @call_indirect_runtime_override_site_cache_test() {
    %fmt_prefix = sim.fmt.literal "sum = "
    %fmt_nl = sim.fmt.literal "\0A"

    llhd.process {
      %vtable_addr = llvm.mlir.addressof @"animal::__vtable__" : !llvm.ptr
      %null = llvm.mlir.zero : !llvm.ptr

      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c3_i32 = arith.constant 3 : i32
      cf.br ^loop(%c0_i32, %c0_i32 : i32, i32)

    ^loop(%iter: i32, %sum: i32):
      %cond = arith.cmpi slt, %iter, %c3_i32 : i32
      cf.cond_br %cond, ^body(%iter, %sum : i32, i32), ^done(%sum : i32)

    ^body(%iter_in: i32, %sum_in: i32):
        %slot_addr = llvm.getelementptr %vtable_addr[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
        %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
        %legs_fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i32
        %legs = func.call_indirect %legs_fn(%null) : (!llvm.ptr) -> i32
        %next_iter = arith.addi %iter_in, %c1_i32 : i32
        %next_sum = arith.addi %sum_in, %legs : i32
        cf.br ^loop(%next_iter, %next_sum : i32, i32)

    ^done(%final_sum: i32):
      %fmt_sum = sim.fmt.dec %final_sum specifierWidth 0 : i32
      %fmt = sim.fmt.concat (%fmt_prefix, %fmt_sum, %fmt_nl)
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

// CHECK: [CI-SITE-CACHE] store method_index=0
// CHECK: [CI-SITE-CACHE] hit method_index=0
// CHECK: sum = 6
