// RUN: circt-sim %s 2>&1 | FileCheck %s

// CHECK-NOT: interpretOperation failed
// CHECK: arr0=7 arr1=0
// CHECK: [circt-sim] Simulation completed

hw.module @top() {
  %c0_i8 = arith.constant 0 : i8
  %c7_i8 = arith.constant 7 : i8
  %arr_init = hw.array_create %c0_i8, %c7_i8 : i8
  %arr = llhd.sig %arr_init : !hw.array<2xi8>

  %fmt_prefix0 = sim.fmt.literal "arr0="
  %fmt_mid = sim.fmt.literal " arr1="
  %fmt_nl = sim.fmt.literal "\0A"

  llhd.process {
    %c42_i8 = arith.constant 42 : i8
    %c0_i1 = arith.constant 0 : i1
    %t1 = llhd.constant_time <0ns, 1d, 0e>
    %t2 = llhd.constant_time <0ns, 2d, 0e>

    %arr0_ref = llhd.sig.array_get %arr[%c0_i1] : <!hw.array<2xi8>>
    cf.br ^bb1(%arr0_ref : !llhd.ref<i8>)
  ^bb1(%selected: !llhd.ref<i8>):
    llhd.drv %selected, %c42_i8 after %t1 : i8
    llhd.wait delay %t2, ^bb2
  ^bb2:
    %arr0_after_ref = llhd.sig.array_get %arr[%c0_i1] : <!hw.array<2xi8>>
    %arr0_after = llhd.prb %arr0_after_ref : i8
    %c1_i1 = arith.constant 1 : i1
    %arr1_after_ref = llhd.sig.array_get %arr[%c1_i1] : <!hw.array<2xi8>>
    %arr1_after = llhd.prb %arr1_after_ref : i8
    %fmt_val = sim.fmt.dec %arr0_after : i8
    %fmt_val1 = sim.fmt.dec %arr1_after : i8
    %fmt_out = sim.fmt.concat (%fmt_prefix0, %fmt_val, %fmt_mid, %fmt_val1, %fmt_nl)
    sim.proc.print %fmt_out
    sim.terminate success, quiet
    llhd.halt
  }

  hw.output
}
