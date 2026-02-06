// RUN: circt-sim %s --max-deltas=5 2>&1 | FileCheck %s

// CHECK: done
// CHECK: [circt-sim] Simulation completed
// CHECK-NOT: ERROR(DELTA_OVERFLOW)

hw.module @SelfDrivenFallbackFilter() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %c1_i64 = hw.constant 1000000 : i64
  %false = hw.constant false
  %true = hw.constant true

  %a = llhd.sig %false : i1
  %b = llhd.sig %false : i1

  // Self-driven process with empty observed list; fallback probes include %a/%b.
  llhd.process {
    %a_val = llhd.prb %a : i1
    %b_val = llhd.prb %b : i1
    %a_xor_b = comb.xor %a_val, %b_val : i1
    %next = comb.xor %a_xor_b, %true : i1
    llhd.drv %a, %next after %eps : i1
    llhd.wait ^bb1
  ^bb1:
    %a_val1 = llhd.prb %a : i1
    %b_val1 = llhd.prb %b : i1
    %a_xor_b1 = comb.xor %a_val1, %b_val1 : i1
    %next1 = comb.xor %a_xor_b1, %true : i1
    llhd.drv %a, %next1 after %eps : i1
    llhd.wait ^bb1
  }

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %fmt = sim.fmt.literal "done\0A"
    sim.proc.print %fmt
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
