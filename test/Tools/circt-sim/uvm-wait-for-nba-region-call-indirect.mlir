// RUN: circt-sim %s 2>&1 | FileCheck %s
//
// Regression: ensure uvm_wait_for_nba_region interception works for
// func.call_indirect dispatch as well as direct func.call.
//
// CHECK: after_indirect_wait
// CHECK-NOT: SHOULD_NOT_RUN

func.func @"uvm_pkg::uvm_wait_for_nba_region"() {
  %msg = sim.fmt.literal "SHOULD_NOT_RUN\0A"
  sim.proc.print %msg
  return
}

llvm.mlir.global internal @nba_wait_vtable(#llvm.zero) {
  addr_space = 0 : i32,
  circt.vtable_entries = [[0, @"uvm_pkg::uvm_wait_for_nba_region"]]
} : !llvm.array<1 x ptr>

hw.module @top() {
  %after = sim.fmt.literal "after_indirect_wait"
  %nl = sim.fmt.literal "\0A"
  llhd.process {
    %vt = llvm.mlir.addressof @nba_wait_vtable : !llvm.ptr
    %fnAddr = llvm.getelementptr %vt[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
    %fnPtr = llvm.load %fnAddr : !llvm.ptr -> !llvm.ptr
    %fn = builtin.unrealized_conversion_cast %fnPtr : !llvm.ptr to () -> ()
    func.call_indirect %fn() : () -> ()
    %line = sim.fmt.concat (%after, %nl)
    sim.proc.print %line
    llhd.halt
  }

  hw.output
}
