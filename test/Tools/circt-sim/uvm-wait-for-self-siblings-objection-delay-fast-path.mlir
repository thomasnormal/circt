// RUN: circt-sim %s --max-time 100000000 --max-deltas 100 2>&1 | FileCheck %s
//
// With objections raised, wait_for_self_and_siblings_to_drop should not spin
// in delta cycles. It must poll on time delay so simulation can advance and
// the delayed objection drop can run.
//
// CHECK: wait done
// CHECK: drop done
// CHECK: [circt-sim] Simulation completed at time 10000000 fs

module {
  llvm.func @__moore_objection_create(!llvm.ptr, i64) -> i64
  llvm.func @__moore_objection_raise(i64, !llvm.ptr, i64, !llvm.ptr, i64, i64)
  llvm.func @__moore_objection_drop(i64, !llvm.ptr, i64, !llvm.ptr, i64, i64)

  llvm.mlir.global internal @objection_handle(0 : i64) : i64

  // Fast-path target. Body should be bypassed.
  func.func @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(
      %phase: !llvm.ptr) {
    return
  }

  hw.module @top() {
    %dropLine = sim.fmt.literal "drop done\0A"
    %waitLine = sim.fmt.literal "wait done\0A"

    llhd.process {
      %null = llvm.mlir.zero : !llvm.ptr
      %c0_i64 = hw.constant 0 : i64
      %c1_i64 = hw.constant 1 : i64
      %phaseRaw = llvm.mlir.constant(4096 : i64) : i64
      %phase = llvm.inttoptr %phaseRaw : i64 to !llvm.ptr
      %hAddr = llvm.mlir.addressof @objection_handle : !llvm.ptr

      %handle = llvm.call @__moore_objection_create(%null, %c0_i64) :
          (!llvm.ptr, i64) -> i64
      llvm.store %handle, %hAddr : i64, !llvm.ptr

      llvm.call @__moore_objection_raise(
          %handle, %null, %c0_i64, %null, %c0_i64, %c1_i64) :
          (i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> ()

      func.call @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(%phase) :
          (!llvm.ptr) -> ()

      sim.proc.print %waitLine
      llhd.halt
    }

    llhd.process {
      %null = llvm.mlir.zero : !llvm.ptr
      %c0_i64 = hw.constant 0 : i64
      %c1_i64 = hw.constant 1 : i64
      %dropDelayFs = hw.constant 10000000 : i64
      %hAddr = llvm.mlir.addressof @objection_handle : !llvm.ptr

      %delay = llhd.int_to_time %dropDelayFs
      llhd.wait delay %delay, ^bb1
    ^bb1:
      %handle = llvm.load %hAddr : !llvm.ptr -> i64
      llvm.call @__moore_objection_drop(
          %handle, %null, %c0_i64, %null, %c0_i64, %c1_i64) :
          (i64, !llvm.ptr, i64, !llvm.ptr, i64, i64) -> ()

      sim.proc.print %dropLine
      sim.terminate success, quiet
      llhd.halt
    }

    hw.output
  }
}
