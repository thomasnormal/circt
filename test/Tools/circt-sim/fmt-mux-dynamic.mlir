// RUN: circt-sim %s --max-time=1000 | FileCheck %s

// Dynamic comb.mux over format strings should be evaluated by sim.print.
module {
  hw.module @test() {
    %t = sim.fmt.literal "MUX_TRUE\0A"
    %f = sim.fmt.literal "MUX_FALSE\0A"
    llhd.process {
      %false = hw.constant false
      cf.br ^bb1(%false : i1)
    ^bb1(%c: i1):
      %fmt = comb.mux %c, %t, %f : !sim.fstring
      sim.proc.print %fmt
      cf.cond_br %c, ^done, ^bb2
    ^bb2:
      %true = hw.constant true
      cf.br ^bb1(%true : i1)
    ^done:
      llhd.halt
    }
    hw.output
  }
}

// CHECK: MUX_FALSE
// CHECK: MUX_TRUE
// CHECK-NOT: <unsupported format>
