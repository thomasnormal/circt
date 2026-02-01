// RUN: circt-sim %s | FileCheck %s

// CHECK: b=1

hw.module @test() {
  %true = hw.constant true
  %false = hw.constant false
  %c1_i64 = hw.constant 1000000 : i64
  %c2_i64 = hw.constant 2000000 : i64
  %eps = llhd.constant_time <0ns, 0d, 1e>

  %fmt_b = sim.fmt.literal "b="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %struct = hw.struct_create (%false, %false) : !hw.struct<a: i1, b: i1>
    %injected = hw.struct_inject %struct["b"], %true : !hw.struct<a: i1, b: i1>
    %b = hw.struct_extract %injected["b"] : !hw.struct<a: i1, b: i1>
    %fmt_val = sim.fmt.dec %b : i1
    %fmt_out = sim.fmt.concat (%fmt_b, %fmt_val, %fmt_nl)
    sim.proc.print %fmt_out
    %delay = llhd.int_to_time %c1_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    %delay2 = llhd.int_to_time %c2_i64
    llhd.wait delay %delay2, ^bb2
  ^bb2:
    llhd.halt
  }

  llhd.process {
    %delay = llhd.int_to_time %c2_i64
    llhd.wait delay %delay, ^bb1
  ^bb1:
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
