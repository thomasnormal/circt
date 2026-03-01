// RUN: circt-sim %s | FileCheck %s
//
// CHECK: [circt-sim] Simulating 2 top modules: Axi4LiteHdlTop, Axi4LiteHvlTop
// CHECK-DAG: axi4lite-hdl-top-init
// CHECK-DAG: axi4lite-hvl-top-init

hw.module @Axi4LiteHdlTop() {
  seq.initial() {
    %msg = sim.fmt.literal "axi4lite-hdl-top-init\0A"
    sim.proc.print %msg
  } : () -> ()
  hw.output
}

hw.module @Axi4LiteHvlTop() {
  seq.initial() {
    %msg = sim.fmt.literal "axi4lite-hvl-top-init\0A"
    sim.proc.print %msg
  } : () -> ()
  hw.output
}
