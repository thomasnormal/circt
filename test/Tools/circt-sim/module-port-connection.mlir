// RUN: circt-sim %s | FileCheck %s

// Test that module port connections (continuous assignments) work correctly
// when the flattened module has drives connecting parent signals to child signals.

// This simulates the scenario where hw.instance port connections become
// llhd.drv operations after module flattening.

// CHECK: [circt-sim] Starting simulation
// CHECK: Clock toggled, child_clk value:
// CHECK: Clock toggled, child_clk value:
// CHECK: [circt-sim] Simulation completed

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c0_i64 = hw.constant 0 : i64
  %c5000000_i64 = hw.constant 5000000 : i64  // 5 time units
  %c30000000_i64 = hw.constant 30000000 : i64  // 30 time units
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %delta = llhd.constant_time <0ns, 1d, 0e>

  %fmt_pre = sim.fmt.literal "Clock toggled, child_clk value: "
  %fmt_nl = sim.fmt.literal "\0A"

  // Parent clock signal
  %parent_clk = llhd.sig %false : i1

  // Child clock signal (simulating a flattened child module's port)
  %child_clk = llhd.sig %false : i1

  // Continuous assignment: child_clk = parent_clk
  // This should update whenever parent_clk changes
  %parent_clk_val = llhd.prb %parent_clk : i1
  llhd.drv %child_clk, %parent_clk_val after %eps : i1

  // Clock toggle process
  llhd.process {
    %delay = llhd.int_to_time %c5000000_i64
    cf.br ^bb1
  ^bb1:
    llhd.wait delay %delay, ^bb2
  ^bb2:
    %clk_val = llhd.prb %parent_clk : i1
    %new_clk = comb.xor %clk_val, %true : i1
    llhd.drv %parent_clk, %new_clk after %delta : i1
    cf.br ^bb1
  }

  // Monitor process - check that child_clk follows parent_clk
  llhd.process {
    cf.br ^wait
  ^wait:
    %old_child = llhd.prb %child_clk : i1
    llhd.wait (%old_child : i1), ^check
  ^check:
    %new_child = llhd.prb %child_clk : i1
    %changed = comb.icmp ne %old_child, %new_child : i1
    cf.cond_br %changed, ^print, ^wait
  ^print:
    %fmt_val = sim.fmt.bin %new_child : i1
    %fmt_str = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_str
    cf.br ^wait
  }

  // Termination process
  llhd.process {
    %timeout = llhd.int_to_time %c30000000_i64
    llhd.wait delay %timeout, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
