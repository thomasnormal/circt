// RUN: circt-sim %s | FileCheck %s

// Regression: bit drives through llhd.sig.extract on a signal must use and
// update pending epsilon state so same-process probes and chained bit updates
// observe blocking-assignment semantics.
// CHECK: [circt-sim] Starting simulation
// CHECK: immediate=3
// CHECK: settled=3
// CHECK: [circt-sim] Simulation completed

hw.module @top() {
  %c0_i8 = arith.constant 0 : i8
  %c0_i3 = arith.constant 0 : i3
  %c1_i3 = arith.constant 1 : i3
  %true = hw.constant true
  %sig = llhd.sig %c0_i8 : i8

  %fmtImm = sim.fmt.literal "immediate="
  %fmtSet = sim.fmt.literal "settled="
  %fmtNl = sim.fmt.literal "\0A"

  llhd.process {
    %eps = llhd.constant_time <0ns, 0d, 1e>
    %c1000000_i64 = hw.constant 1000000 : i64
    %c1_i64 = hw.constant 1 : i64
    %startDelay = llhd.int_to_time %c1000000_i64
    %stepDelay = llhd.int_to_time %c1_i64

    llhd.wait delay %startDelay, ^bb1
  ^bb1:
    %bit0Ref = llhd.sig.extract %sig from %c0_i3 : <i8> -> <i1>
    llhd.drv %bit0Ref, %true after %eps : i1
    %bit1Ref = llhd.sig.extract %sig from %c1_i3 : <i8> -> <i1>
    llhd.drv %bit1Ref, %true after %eps : i1

    %imm = llhd.prb %sig : i8
    %fmtImmVal = sim.fmt.dec %imm : i8
    %immOut = sim.fmt.concat (%fmtImm, %fmtImmVal, %fmtNl)
    sim.proc.print %immOut

    llhd.wait delay %stepDelay, ^bb2
  ^bb2:
    %settled = llhd.prb %sig : i8
    %fmtSetVal = sim.fmt.dec %settled : i8
    %setOut = sim.fmt.concat (%fmtSet, %fmtSetVal, %fmtNl)
    sim.proc.print %setOut

    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
