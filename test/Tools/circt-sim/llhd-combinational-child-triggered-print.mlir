// RUN: circt-sim %s | FileCheck %s

// CHECK: child=0
// CHECK: child=1

hw.module @child(in %in: !llhd.ref<i1>) {
  %fmt_prefix = sim.fmt.literal "child="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.combinational {
    %in_val = llhd.prb %in : i1
    %fmt_val = sim.fmt.dec %in_val : i1
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.yield
  }

  hw.output
}

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %a = llhd.sig %false : i1
  hw.instance "u0" @child(in: %a: !llhd.ref<i1>) -> ()

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %a, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c2_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
