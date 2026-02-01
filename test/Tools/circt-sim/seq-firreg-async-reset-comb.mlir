// RUN: circt-sim %s | FileCheck %s

// CHECK: reg=0

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c1_i64 = hw.constant 1000000 : i64
  %c10_i64 = hw.constant 10000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_reg = sim.fmt.literal "reg="
  %fmt_nl = sim.fmt.literal "\0A"

  %clk = llhd.sig %false : i1
  %rst = llhd.sig %false : i1

  // Assert reset at time 0, deassert at 10ns.
  llhd.process {
    llhd.drv %rst, %true after %eps : i1
    %delay = llhd.int_to_time %c10_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %rst, %false after %eps : i1
    llhd.halt
  }

  %rst_val = llhd.prb %rst : i1
  %rst_comb:1 = llhd.combinational -> i1 {
    llhd.yield %rst_val : i1
  }

  %clk_val = llhd.prb %clk : i1
  %clk_clock = seq.to_clock %clk_val
  %reg = seq.firreg %next clock %clk_clock reset async %rst_comb#0, %false : i1
  %next = comb.xor %reg, %true : i1

  %reg_sig = llhd.sig %false : i1
  llhd.drv %reg_sig, %reg after %eps : i1

  // Print reg after reset asserted (before any clock edge).
  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %reg_val = llhd.prb %reg_sig : i1
    %fmt_val = sim.fmt.dec %reg_val : i1
    %fmt_out = sim.fmt.concat (%fmt_reg, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
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
