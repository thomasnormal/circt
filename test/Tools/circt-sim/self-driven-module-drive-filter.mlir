// RUN: circt-sim %s --max-deltas=5 2>&1 | FileCheck %s

// CHECK: self_drive=0
// CHECK: [circt-sim] Simulation finished successfully
// CHECK-NOT: ERROR(DELTA_OVERFLOW)

hw.module @test() {
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %c1_i64 = hw.constant 1000000 : i64
  %false = hw.constant false
  %true = hw.constant true

  %a = llhd.sig %false : i1
  %b = llhd.sig %false : i1

  %proc_val = llhd.process -> i1 {
    %a_val = llhd.prb %a : i1
    %b_val = llhd.prb %b : i1
    %next = comb.xor %a_val, %true : i1
    llhd.wait yield (%next : i1), (%a_val, %b_val : i1, i1), ^bb1
  ^bb1:
    %a_val1 = llhd.prb %a : i1
    %b_val1 = llhd.prb %b : i1
    %next1 = comb.xor %a_val1, %true : i1
    llhd.wait yield (%next1 : i1), (%a_val1, %b_val1 : i1, i1), ^bb1
  }

  llhd.drv %a, %proc_val after %eps : i1

  llhd.process {
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    llhd.drv %b, %true after %eps : i1
    llhd.wait delay %delay, ^bb2
  ^bb2:
    %a_out = llhd.prb %a : i1
    %fmt_pre = sim.fmt.literal "self_drive="
    %fmt_val = sim.fmt.dec %a_out : i1
    %fmt_nl = sim.fmt.literal "\0A"
    %fmt_out = sim.fmt.concat (%fmt_pre, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
