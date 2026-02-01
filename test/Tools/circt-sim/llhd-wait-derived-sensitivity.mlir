// RUN: circt-sim %s | FileCheck %s

// CHECK: tick
// CHECK-NOT: tick

hw.module @test() {
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %c4_i64 = hw.constant 4000000 : i64
  %true = hw.constant true
  %false = hw.constant false
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %a = llhd.sig %false : i1
  %b = llhd.sig %false : i1
  %out = llhd.sig %false : i1

  %fmt_tick = sim.fmt.literal "tick\0A"

  // Toggle b first (should not trigger), then a (should trigger).
  llhd.process {
    %d1 = llhd.int_to_time %c1_i64
    %d2 = llhd.int_to_time %c2_i64
    llhd.wait delay %d1, ^bb1
  ^bb1:
    llhd.drv %b, %true after %eps : i1
    llhd.wait delay %d2, ^bb2
  ^bb2:
    llhd.drv %a, %true after %eps : i1
    llhd.halt
  }

  // Derived sensitivity should only include signals feeding the drive value.
  llhd.process {
    llhd.wait ^bb1
  ^bb1:
    %a_val = llhd.prb %a : i1
    %b_val = llhd.prb %b : i1
    %masked = comb.xor %a_val, %false : i1
    llhd.drv %out, %masked after %eps : i1
    sim.proc.print %fmt_tick
    llhd.wait ^bb1
  }

  llhd.process {
    %d4 = llhd.int_to_time %c4_i64
    llhd.wait delay %d4, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
