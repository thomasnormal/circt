// RUN: circt-sim %s --top=top --max-time=5 2>&1 | FileCheck %s
// RUN: env CIRCT_SIM_EXPERIMENTAL_PARALLEL=1 circt-sim %s --top=top --parallel=2 --work-stealing --auto-partition --max-time=5 --max-wall-ms=2000 2>&1 | FileCheck %s --check-prefix=CHECK-PAR
//
// Regression: work scheduled exactly at --max-time must execute before exit.
//
// CHECK-DAG: at5
// CHECK-DAG: [circt-sim] Simulation completed at time 5 fs
// CHECK-PAR: disabled for LLHD interpreter workloads due deadlock risk
// CHECK-PAR: at5
// CHECK-PAR: [circt-sim] Simulation completed at time 5 fs

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
