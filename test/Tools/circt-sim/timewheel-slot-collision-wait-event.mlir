// RUN: circt-sim %s --max-time=50000000 2>&1 | FileCheck %s

// Regression: time-wheel slot collisions must not retime earlier events to a
// later absolute time. This test schedules events at 20ns and 30ns (same level-3
// wheel slot under default config) and requires the 20ns clock edge to wake a
// second wait_event before the 30ns checker.
// CHECK: driveA
// CHECK: driveB
// CHECK: done=1 clkA=1 clkB=1
// CHECK: [circt-sim] Simulation completed

hw.module @test() {
  %false = hw.constant false
  %true = hw.constant true
  %c10_i64 = hw.constant 10000000 : i64
  %c20_i64 = hw.constant 20000000 : i64
  %c30_i64 = hw.constant 30000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %msg_a = sim.fmt.literal "driveA\0A"
  %msg_b = sim.fmt.literal "driveB\0A"
  %fmt_pre = sim.fmt.literal "done="
  %fmt_mid = sim.fmt.literal " clkA="
  %fmt_mid2 = sim.fmt.literal " clkB="
  %fmt_nl = sim.fmt.literal "\0A"

  %clkA = llhd.sig %false : i1
  %clkB = llhd.sig %false : i1
  %done = llhd.sig %false : i1

  llhd.process {
    %t = llhd.int_to_time %c10_i64
    llhd.wait delay %t, ^bb1
  ^bb1:
    sim.proc.print %msg_a
    llhd.drv %clkA, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    %t = llhd.int_to_time %c20_i64
    llhd.wait delay %t, ^bb1
  ^bb1:
    sim.proc.print %msg_b
    llhd.drv %clkB, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    moore.wait_event {
      %a_val = llhd.prb %clkA : i1
      %a_m = builtin.unrealized_conversion_cast %a_val : i1 to !moore.l1
      moore.detect_event posedge %a_m : l1
    }
    moore.wait_event {
      %b_val = llhd.prb %clkB : i1
      %b_m = builtin.unrealized_conversion_cast %b_val : i1 to !moore.l1
      moore.detect_event posedge %b_m : l1
    }
    llhd.drv %done, %true after %eps : i1
    llhd.halt
  }

  llhd.process {
    %t = llhd.int_to_time %c30_i64
    llhd.wait delay %t, ^bb1
  ^bb1:
    %done_val = llhd.prb %done : i1
    %a_val = llhd.prb %clkA : i1
    %b_val = llhd.prb %clkB : i1
    %fmt_done = sim.fmt.dec %done_val : i1
    %fmt_a = sim.fmt.dec %a_val : i1
    %fmt_b = sim.fmt.dec %b_val : i1
    %fmt = sim.fmt.concat (%fmt_pre, %fmt_done, %fmt_mid, %fmt_a, %fmt_mid2, %fmt_b, %fmt_nl)
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
