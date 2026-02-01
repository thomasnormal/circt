// RUN: circt-sim %s | FileCheck %s

// CHECK: and=0
// CHECK: or=14

hw.module @test() {
  %a = hw.constant 10 : i4
  %b = hw.constant 12 : i4
  %c = hw.constant 0 : i4

  %fmt_and = sim.fmt.literal "and="
  %fmt_or = sim.fmt.literal "or="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %and = comb.and %a, %b, %c : i4
    %or = comb.or %a, %b, %c : i4
    %and_val = sim.fmt.dec %and : i4
    %or_val = sim.fmt.dec %or : i4
    %and_out = sim.fmt.concat (%fmt_and, %and_val, %fmt_nl)
    %or_out = sim.fmt.concat (%fmt_or, %or_val, %fmt_nl)
    sim.proc.print %and_out
    sim.proc.print %or_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
