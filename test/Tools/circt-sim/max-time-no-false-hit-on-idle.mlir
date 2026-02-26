// RUN: circt-sim %s --top=top --max-time=1000 2>&1 | FileCheck %s
//
// Regression: when simulation naturally goes idle before --max-time, do not
// classify the exit as max-time reached.
//
// CHECK-NOT: Main loop exit: maxTime reached
// CHECK-DAG: at5
// CHECK-DAG: [circt-sim] advanceTime() returned false at time 5 fs
// CHECK: [circt-sim] Simulation completed at time 5 fs

hw.module @top() {
  %false = hw.constant false
  %true = hw.constant true
  %delay5 = arith.constant 5 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %msg = sim.fmt.literal "at5\0A"

  %sig = llhd.sig %false : i1

  llhd.process {
    %t = llhd.int_to_time %delay5
    llhd.wait delay %t, ^bb1
  ^bb1:
    llhd.drv %sig, %true after %eps : i1
    sim.proc.print %msg
    llhd.halt
  }

  hw.output
}
