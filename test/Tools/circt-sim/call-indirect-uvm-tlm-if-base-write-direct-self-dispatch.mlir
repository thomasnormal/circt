// RUN: env CIRCT_SIM_TRACE_ANALYSIS=1 circt-sim %s 2>&1 | FileCheck %s

// Regression: when call_indirect reaches uvm_tlm_if_base::*::write and the
// native connection map has no terminals, the first argument may already be
// the terminal imp object. In that case, dispatch via the self object's
// runtime vtable slot 11 instead of dropping to a no-op/base body.
//
// CHECK: [ANALYSIS-WRITE] uvm_pkg::uvm_tlm_if_base_8050::write
// CHECK: [ANALYSIS-WRITE] dispatching to my_pkg::my_imp_9001::write
// CHECK: IMPL_WRITE
// CHECK-NOT: BASE_WRITE_BODY
// CHECK: done

func.func private @"my_pkg::my_imp_9001::write"(%self: i64, %tx: i64) {
  %fmt_impl = sim.fmt.literal "IMPL_WRITE\0A"
  sim.proc.print %fmt_impl
  return
}

func.func private @"uvm_pkg::uvm_tlm_if_base_8050::write"(%self: i64, %tx: i64) {
  %fmt_base = sim.fmt.literal "BASE_WRITE_BODY\0A"
  sim.proc.print %fmt_base
  return
}

func.func @call_write_indirect(%fptr: !llvm.ptr, %self: i64, %tx: i64) {
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64, i64) -> ()
  func.call_indirect %fn(%self, %tx) : (i64, i64) -> ()
  return
}

func.func @invoke_base_write_on_imp_object(%tx: i64) {
  %c1 = arith.constant 1 : i64
  %obj = llvm.alloca %c1 x !llvm.array<24 x i8> : (i64) -> !llvm.ptr
  %obj_i8 = llvm.getelementptr %obj[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<24 x i8>
  %obj_vtbl = llvm.getelementptr %obj_i8[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %imp_vtbl = llvm.mlir.addressof @"my_pkg::__imp_vtable__" : !llvm.ptr
  llvm.store %imp_vtbl, %obj_vtbl : !llvm.ptr, !llvm.ptr
  %self_i64 = llvm.ptrtoint %obj_i8 : !llvm.ptr to i64

  %base_vtbl = llvm.mlir.addressof @"uvm_pkg::__base_vtable__" : !llvm.ptr
  %slot0 = llvm.getelementptr %base_vtbl[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  %base_write_fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
  func.call @call_write_indirect(%base_write_fptr, %self_i64, %tx) : (!llvm.ptr, i64, i64) -> ()
  return
}

llvm.mlir.global internal @"uvm_pkg::__base_vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_tlm_if_base_8050::write"]
  ]
} : !llvm.array<1 x ptr>

llvm.mlir.global internal @"my_pkg::__imp_vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [11, @"my_pkg::my_imp_9001::write"]
  ]
} : !llvm.array<12 x ptr>

hw.module @test() {
  %c_tx = hw.constant 55 : i64
  %fmt_done = sim.fmt.literal "done\0A"
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    func.call @invoke_base_write_on_imp_object(%c_tx) : (i64) -> ()
    sim.proc.print %fmt_done
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
