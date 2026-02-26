// RUN: circt-sim %s --top=top --sim-stats | FileCheck %s

// Regression: non-UVM wait(condition) backed by a single memory load should
// not leave a stale timed poll that advances final simulation time.
//
// CHECK: done
// CHECK: [circt-sim] Simulation completed at time 5 fs

module {
  llvm.func @__moore_wait_condition(i32)
  llvm.func @__moore_delay(i64)

  llvm.mlir.global internal @flag(0 : i32) : i32

  hw.module @top() {
    %done = sim.fmt.literal "done\0A"

    llhd.process {
      %flagAddr = llvm.mlir.addressof @flag : !llvm.ptr
      %cond = llvm.load %flagAddr : !llvm.ptr -> i32
      llvm.call @__moore_wait_condition(%cond) : (i32) -> ()
      sim.proc.print %done
      llhd.halt
    }

    llhd.process {
      %delay = llvm.mlir.constant(5 : i64) : i64
      %flagAddr = llvm.mlir.addressof @flag : !llvm.ptr
      %one = llvm.mlir.constant(1 : i32) : i32
      llvm.call @__moore_delay(%delay) : (i64) -> ()
      llvm.store %one, %flagAddr : i32, !llvm.ptr
      llhd.halt
    }

    hw.output
  }
}
