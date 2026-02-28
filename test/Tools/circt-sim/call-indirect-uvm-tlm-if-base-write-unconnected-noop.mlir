// RUN: env CIRCT_SIM_TRACE_ANALYSIS=1 circt-sim %s 2>&1 | FileCheck %s

// Regression: when uvm_tlm_if_base::*::write is reached through the native
// analysis-write intercept path but no native terminals are known, do not
// execute the base UVM "not implemented" body.
//
// CHECK: [ANALYSIS-WRITE] uvm_pkg::uvm_tlm_if_base_8050::write
// CHECK-NOT: BASE_WRITE_BODY
// CHECK: done

func.func private @"uvm_pkg::uvm_tlm_if_base_8050::write"(%self: i64, %tx: i64) {
  %fmt_body = sim.fmt.literal "BASE_WRITE_BODY\0A"
  sim.proc.print %fmt_body
  return
}

func.func @call_write_indirect(%fptr: !llvm.ptr, %self: i64, %tx: i64) {
  %fn = builtin.unrealized_conversion_cast %fptr : !llvm.ptr to (i64, i64) -> ()
  func.call_indirect %fn(%self, %tx) : (i64, i64) -> ()
  return
}

func.func @invoke_write_from_vtable(%self: i64, %tx: i64) {
  %vtable = llvm.mlir.addressof @"uvm_pkg::__vtable__" : !llvm.ptr
  %slot0 = llvm.getelementptr %vtable[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
  %fptr = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
  func.call @call_write_indirect(%fptr, %self, %tx) : (!llvm.ptr, i64, i64) -> ()
  return
}

llvm.mlir.global internal @"uvm_pkg::__vtable__"(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [
    [0, @"uvm_pkg::uvm_tlm_if_base_8050::write"]
  ]
} : !llvm.array<1 x ptr>

hw.module @test() {
  %c_self = hw.constant 99 : i64
  %c_tx = hw.constant 1234 : i64
  %fmt_done = sim.fmt.literal "done\0A"
  %c10_i64 = hw.constant 10000000 : i64

  llhd.process {
    func.call @invoke_write_from_vtable(%c_self, %c_tx) : (i64, i64) -> ()
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
