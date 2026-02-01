// RUN: circt-sim %s | FileCheck %s

// CHECK: b=1
// CHECK: b=0

hw.module @child(in %in : i1, out out: i1) {
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %in_sig = llhd.sig %false : i1
  llhd.drv %in_sig, %in after %eps : i1
  %in_val = llhd.prb %in_sig : i1
  hw.output %in_val : i1
}

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c1_i64 = hw.constant 1000000 : i64
  %c5_i64 = hw.constant 5000000 : i64
  %c6_i64 = hw.constant 6000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_b = sim.fmt.literal "b="
  %fmt_nl = sim.fmt.literal "\0A"

  %a = llhd.sig %false : i1
  %b = llhd.sig %false : i1

  %a_val = llhd.prb %a : i1
  %not_a = comb.xor %a_val, %true : i1
  %inst.out = hw.instance "inst" @child(in: %not_a : i1) -> (out: i1)
  llhd.drv %b, %inst.out after %eps : i1

  llhd.process {
    llhd.drv %a, %false after %eps : i1
    %delay = llhd.int_to_time %c5_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %a, %true after %eps : i1
    %delay2 = llhd.int_to_time %c5_i64
    llhd.wait delay %delay2, ^bb2
  ^bb2:
    llhd.halt
  }

  llhd.process {
    %delay0 = llhd.int_to_time %c1_i64
    llhd.wait delay %delay0, ^bb1
  ^bb1:
    %b_val = llhd.prb %b : i1
    %fmt_b_val = sim.fmt.dec %b_val : i1
    %fmt_out = sim.fmt.concat (%fmt_b, %fmt_b_val, %fmt_nl)
    sim.proc.print %fmt_out
    %delay1 = llhd.int_to_time %c6_i64
    llhd.wait delay %delay1, ^bb2
  ^bb2:
    %b_val2 = llhd.prb %b : i1
    %fmt_b_val2 = sim.fmt.dec %b_val2 : i1
    %fmt_out2 = sim.fmt.concat (%fmt_b, %fmt_b_val2, %fmt_nl)
    sim.proc.print %fmt_out2
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c20_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
