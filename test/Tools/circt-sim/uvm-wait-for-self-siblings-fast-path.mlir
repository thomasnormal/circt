// RUN: circt-sim %s --max-time 100000000 2>&1 | FileCheck %s
//
// Verify that wait_for_self_and_siblings_to_drop is handled by fast-path
// dispatch for both func.call and call_indirect without executing the callee
// body.
//
// CHECK: after direct body_executed = 0
// CHECK: after indirect body_executed = 0

module {
  llvm.mlir.global internal @body_executed(0 : i32) : i32

  llvm.mlir.global internal @phase_vtable(#llvm.zero) {
    addr_space = 0 : i32,
    circt.vtable_entries = [[0, @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"]]
  } : !llvm.array<1 x ptr>

  func.func @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(
      %phase: !llvm.ptr) {
    %flag = llvm.mlir.addressof @body_executed : !llvm.ptr
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %flag : i32, !llvm.ptr
    return
  }

  hw.module @top() {
    %fmtDirect = sim.fmt.literal "after direct body_executed = "
    %fmtIndirect = sim.fmt.literal "after indirect body_executed = "
    %fmtNl = sim.fmt.literal "\0A"

    llhd.process {
      %phase64 = llvm.mlir.constant(4096 : i64) : i64
      %phase = llvm.inttoptr %phase64 : i64 to !llvm.ptr
      %flag = llvm.mlir.addressof @body_executed : !llvm.ptr
      %zero = llvm.mlir.constant(0 : i32) : i32

      func.call @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(%phase) :
          (!llvm.ptr) -> ()
      %directVal = llvm.load %flag : !llvm.ptr -> i32
      %directFmt = sim.fmt.dec %directVal signed : i32
      %directLine = sim.fmt.concat (%fmtDirect, %directFmt, %fmtNl)
      sim.proc.print %directLine

      llvm.store %zero, %flag : i32, !llvm.ptr

      %vt = llvm.mlir.addressof @phase_vtable : !llvm.ptr
      %slot0 = llvm.getelementptr %vt[0, 0] :
          (!llvm.ptr) -> !llvm.ptr, !llvm.array<1 x ptr>
      %fptr0 = llvm.load %slot0 : !llvm.ptr -> !llvm.ptr
      %fn0 = builtin.unrealized_conversion_cast %fptr0 :
          !llvm.ptr to (!llvm.ptr) -> ()
      func.call_indirect %fn0(%phase) : (!llvm.ptr) -> ()

      %indirectVal = llvm.load %flag : !llvm.ptr -> i32
      %indirectFmt = sim.fmt.dec %indirectVal signed : i32
      %indirectLine = sim.fmt.concat (%fmtIndirect, %indirectFmt, %fmtNl)
      sim.proc.print %indirectLine

      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
