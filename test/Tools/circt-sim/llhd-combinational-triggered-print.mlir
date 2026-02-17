// RUN: circt-sim %s | FileCheck %s

// CHECK: comb=0
// CHECK: comb=1

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_prefix = sim.fmt.literal "comb="
  %fmt_nl = sim.fmt.literal "\0A"

  %a = llhd.sig %false : i1

  llhd.combinational {
    %a_val = llhd.prb %a : i1
    %fmt_val = sim.fmt.dec %a_val : i1
    %fmt_out = sim.fmt.concat (%fmt_prefix, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    llhd.yield
  }

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
