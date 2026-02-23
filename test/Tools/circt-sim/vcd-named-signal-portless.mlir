// RUN: circt-sim %s --top top --vcd %t.vcd > %t.out 2>&1
// RUN: FileCheck %s --check-prefix=SIM < %t.out
// RUN: FileCheck %s --check-prefix=VCD < %t.vcd

// Regression: --vcd on a portless design must still emit $var declarations for
// named llhd.sig values. This covers logic lowered as value/unknown structs.

// SIM: [circt-sim] Wrote waveform to
// SIM: [circt-sim] Simulation completed

// VCD: $var wire {{[0-9]+}} {{.+}} named_state $end

hw.module @top() {
  %c1_i64 = hw.constant 1000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>
  %zero = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
  %one = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>

  %state = llhd.sig name "named_state" %zero : !hw.struct<value: i1, unknown: i1>

  llhd.process {
    llhd.drv %state, %one after %eps : !hw.struct<value: i1, unknown: i1>
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
