// RUN: circt-sim %s -v=1 2>&1 | FileCheck %s

// CHECK: [circt-sim] Stage: parse (prev:
// CHECK: [circt-sim] Stage: passes (prev:
// CHECK: [circt-sim] Stage: init (prev:
// CHECK: [circt-sim] Stage: run (prev:

hw.module @test() {
  %fmt_prefix = sim.fmt.literal "ok"
  %fmt_nl = sim.fmt.literal "\0A"
  %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_nl)

  llhd.process {
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }
  hw.output
}
