// RUN: circt-sim %s | FileCheck %s

// CHECK: reg=0
// CHECK: reg=1

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i32 = hw.constant 0 : i32
  %c1_i32 = hw.constant 1 : i32
  %c2_i32 = hw.constant 2 : i32
  %c5_i64 = hw.constant 5000000 : i64
  %c17_i64 = hw.constant 17000000 : i64
  %c100_i64 = hw.constant 100000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_reg = sim.fmt.literal "reg="
  %fmt_nl = sim.fmt.literal "\0A"

  %clk = llhd.sig %false : i1
  %rst = llhd.sig %true : i1

  // Clock generator: 10ns period.
  llhd.process {
    llhd.drv %clk, %false after %eps : i1
    cf.br ^bb1
  ^bb1:
    %delay = llhd.int_to_time %c5_i64
    llhd.wait delay %delay, ^bb2
  ^bb2:
    %clk_val = llhd.prb %clk : i1
    %clk_inv = comb.xor %clk_val, %true : i1
    llhd.drv %clk, %clk_inv after %eps : i1
    cf.br ^bb1
  }

  // Reset deassert after 17ns.
  llhd.process {
    llhd.drv %rst, %true after %eps : i1
    %delay = llhd.int_to_time %c17_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %rst, %false after %eps : i1
    llhd.halt
  }

  %clk_val = llhd.prb %clk : i1
  %clk_clock = seq.to_clock %clk_val
  %rst_val = llhd.prb %rst : i1

  %reg = seq.firreg %next clock %clk_clock reset async %rst_val, %false : i1
  %next = comb.xor %reg, %true : i1

  %reg_sig = llhd.sig %false : i1
  llhd.drv %reg_sig, %reg after %eps : i1

  // Print reg value after two clock edges post-reset.
  llhd.process {
    %delay0 = llhd.int_to_time %c17_i64
    llhd.wait delay %delay0, ^bb1
  ^bb1:
    %delay1 = llhd.int_to_time %c5_i64
    llhd.wait delay %delay1, ^print1
  ^print1:
    %reg_val = llhd.prb %reg_sig : i1
    %fmt_val = sim.fmt.dec %reg_val : i1
    %fmt_out = sim.fmt.concat (%fmt_reg, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    %delay2 = llhd.int_to_time %c5_i64
    llhd.wait delay %delay2, ^print2
  ^print2:
    %reg_val2 = llhd.prb %reg_sig : i1
    %fmt_val2 = sim.fmt.dec %reg_val2 : i1
    %fmt_out2 = sim.fmt.concat (%fmt_reg, %fmt_val2, %fmt_nl)
    sim.proc.print %fmt_out2
    llhd.halt
  }

  // Terminate after 100ns.
  llhd.process {
    %delay = llhd.int_to_time %c100_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
