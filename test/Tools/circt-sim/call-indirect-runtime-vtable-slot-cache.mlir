// RUN: env CIRCT_SIM_TRACE_CALL_INDIRECT_SITE_CACHE=1 circt-sim %s --skip-passes 2>&1 | FileCheck %s

// Regression: cache runtime vtable slot lookups for call_indirect runtime
// override (runtime_vtable_addr + method_index).

module {
  func.func private @"animal::legs"(%self: !llvm.ptr) -> i32 {
    %c2 = arith.constant 2 : i32
    return %c2 : i32
  }

  func.func private @"spider::legs"(%self: !llvm.ptr) -> i32 {
    %c8 = arith.constant 8 : i32
    return %c8 : i32
  }

  llvm.func @malloc(i64) -> !llvm.ptr

  llvm.mlir.global internal @"animal::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"animal::legs"]]
  } : !llvm.array<1 x ptr>

  llvm.mlir.global internal @"spider::__vtable__"(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"spider::legs"]]
  } : !llvm.array<1 x ptr>

  hw.module @call_indirect_runtime_vtable_slot_cache_test() {
    %fmt_prefix = sim.fmt.literal "sum = "
    %fmt_nl = sim.fmt.literal "\0A"

    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i64 = arith.constant 4 : i64
    %c16_i64 = arith.constant 16 : i64

    llhd.process {
      // Allocate an object and write runtime vtable pointer at byte offset 4.
      %obj = llvm.call @malloc(%c16_i64) : (i64) -> !llvm.ptr
      llvm.store %c0_i32, %obj : i32, !llvm.ptr
      %spider_vtable = llvm.mlir.addressof @"spider::__vtable__" : !llvm.ptr
      %obj_vtable_ptr = llvm.getelementptr %obj[4] : (!llvm.ptr) -> !llvm.ptr, i8
      llvm.store %spider_vtable, %obj_vtable_ptr : !llvm.ptr, !llvm.ptr

      %animal_vtable = llvm.mlir.addressof @"animal::__vtable__" : !llvm.ptr
      cf.br ^loop(%c0_i32, %c0_i32, %animal_vtable, %obj : i32, i32, !llvm.ptr, !llvm.ptr)

    ^loop(%iter: i32, %sum: i32, %vtable: !llvm.ptr, %self: !llvm.ptr):
      %cond = arith.cmpi slt, %iter, %c3_i32 : i32
      cf.cond_br %cond, ^body(%iter, %sum, %vtable, %self : i32, i32, !llvm.ptr, !llvm.ptr), ^done(%sum : i32)

    ^body(%iter_in: i32, %sum_in: i32, %vtable_in: !llvm.ptr, %self_in: !llvm.ptr):
      %slot_addr = llvm.getelementptr %vtable_in[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr = llvm.load %slot_addr : !llvm.ptr -> !llvm.ptr
      %legs_fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (!llvm.ptr) -> i32
      %legs = func.call_indirect %legs_fn(%self_in) : (!llvm.ptr) -> i32
      %next_iter = arith.addi %iter_in, %c1_i32 : i32
      %next_sum = arith.addi %sum_in, %legs : i32
      cf.br ^loop(%next_iter, %next_sum, %vtable_in, %self_in : i32, i32, !llvm.ptr, !llvm.ptr)

    ^done(%final_sum: i32):
      %fmt_sum = sim.fmt.dec %final_sum specifierWidth 0 : i32
      %fmt = sim.fmt.concat (%fmt_prefix, %fmt_sum, %fmt_nl)
      sim.proc.print %fmt
      llhd.halt
    }

    hw.output
  }
}

// CHECK: [CI-SITE-CACHE] runtime-slot-store
// CHECK: [CI-SITE-CACHE] runtime-slot-hit
// CHECK: sum = 24
