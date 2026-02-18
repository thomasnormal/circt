// RUN: env CIRCT_SIM_TRACE_FORK_JOIN=1 circt-sim %s --max-time=10000000 --max-deltas=128 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_TRACE_FORK_JOIN=1 circt-sim %s --parallel=4 --work-stealing --auto-partition --max-time=10000000 --max-deltas=128 2>&1 | FileCheck %s

// Verify execute_phase monitor-fork interception uses objection-zero waiter
// (instead of delta polling churn) when objections are active.
//
// CHECK: [FORK-INTERCEPT]
// CHECK: wait_mode=objection_zero
// CHECK-DAG: drop done
// CHECK-DAG: phase done
// CHECK: [circt-sim] Simulation completed

module {
  llvm.func @__moore_wait_condition(i32)
  llvm.func @__moore_delay(i64)

  func.func @"uvm_pkg::uvm_phase::raise_objection"(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) {
    return
  }
  func.func @"uvm_pkg::uvm_phase::drop_objection"(%arg0: i64, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32) {
    return
  }
  func.func @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(%arg0: i64) {
    return
  }

  hw.module @top() {
    %phaseDone = sim.fmt.literal "phase done\0A"
    %dropDone = sim.fmt.literal "drop done\0A"

    llhd.process {
      %phase = llvm.mlir.constant(4096 : i64) : i64
      %null = llvm.mlir.zero : !llvm.ptr
      %c0_i32 = hw.constant 0 : i32
      %c1_i32 = hw.constant 1 : i32
      %c1_i64 = llvm.mlir.constant(1 : i64) : i64

      func.call @"uvm_pkg::uvm_phase::raise_objection"(%phase, %null, %null, %c1_i32) : (i64, !llvm.ptr, !llvm.ptr, i32) -> ()

      %handle = sim.fork join_type "join_any" {
        llvm.call @__moore_wait_condition(%c0_i32) : (i32) -> ()
        sim.fork.terminator
      }, {
        func.call @"uvm_pkg::uvm_phase::wait_for_self_and_siblings_to_drop"(%phase) : (i64) -> ()
        sim.fork.terminator
      }, {
        llvm.call @__moore_delay(%c1_i64) : (i64) -> ()
        sim.fork.terminator
      }

      sim.proc.print %phaseDone
      sim.terminate success, quiet
      llhd.halt
    }

    llhd.process {
      %phase = llvm.mlir.constant(4096 : i64) : i64
      %null = llvm.mlir.zero : !llvm.ptr
      %c1_i32 = hw.constant 1 : i32
      %delayFs = llvm.mlir.constant(64 : i64) : i64
      llvm.call @__moore_delay(%delayFs) : (i64) -> ()
      func.call @"uvm_pkg::uvm_phase::drop_objection"(%phase, %null, %null, %c1_i32) : (i64, !llvm.ptr, !llvm.ptr, i32) -> ()
      sim.proc.print %dropDone
      llhd.halt
    }

    hw.output
  }
}
