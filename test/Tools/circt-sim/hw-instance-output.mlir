// RUN: circt-sim %s | FileCheck %s

// CHECK: b0=1 b1=1
// CHECK: b0=0 b1=0

hw.module @child(in %in : i1, out out: i1) {
  %true = hw.constant true
  %not_in = comb.xor %in, %true : i1
  hw.output %not_in : i1
}

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c1_i64 = hw.constant 1000000 : i64
  %c5_i64 = hw.constant 5000000 : i64
  %c6_i64 = hw.constant 6000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_b0 = sim.fmt.literal "b0="
  %fmt_b1 = sim.fmt.literal " b1="
  %fmt_nl = sim.fmt.literal "\0A"

  %a = llhd.sig %false : i1
  %b0 = llhd.sig %false : i1
  %b1 = llhd.sig %false : i1

  %a_val = llhd.prb %a : i1
  %inst0.out = hw.instance "inst0" @child(in: %a_val : i1) -> (out: i1)
  %inst1.out = hw.instance "inst1" @child(in: %a_val : i1) -> (out: i1)
  llhd.drv %b0, %inst0.out after %eps : i1
  llhd.drv %b1, %inst1.out after %eps : i1

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
    %b0_val = llhd.prb %b0 : i1
    %b1_val = llhd.prb %b1 : i1
    %fmt_b0_val = sim.fmt.dec %b0_val : i1
    %fmt_b1_val = sim.fmt.dec %b1_val : i1
    %fmt_out = sim.fmt.concat (%fmt_b0, %fmt_b0_val, %fmt_b1, %fmt_b1_val, %fmt_nl)
    sim.proc.print %fmt_out
    %delay1 = llhd.int_to_time %c6_i64
    llhd.wait delay %delay1, ^bb2
  ^bb2:
    %b0_val2 = llhd.prb %b0 : i1
    %b1_val2 = llhd.prb %b1 : i1
    %fmt_b0_val2 = sim.fmt.dec %b0_val2 : i1
    %fmt_b1_val2 = sim.fmt.dec %b1_val2 : i1
    %fmt_out2 = sim.fmt.concat (%fmt_b0, %fmt_b0_val2, %fmt_b1, %fmt_b1_val2, %fmt_nl)
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
