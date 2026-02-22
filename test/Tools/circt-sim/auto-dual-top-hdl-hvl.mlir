// RUN: circt-sim %s | FileCheck %s
//
// CHECK: [circt-sim] Simulating 2 top modules: hdl_top, hvl_top
// CHECK-DAG: hdl-top-init
// CHECK-DAG: hvl-top-init

hw.module @hdl_top() {
  seq.initial() {
    %msg = sim.fmt.literal "hdl-top-init\0A"
    sim.proc.print %msg
  } : () -> ()
  hw.output
}

hw.module @hvl_top() {
  seq.initial() {
    %msg = sim.fmt.literal "hvl-top-init\0A"
    sim.proc.print %msg
  } : () -> ()
  hw.output
}
