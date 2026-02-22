// RUN: not circt-sim %s --top top --max-time=500000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time 50000000
// CHECK: SVA assertion failed at time 150000000
// CHECK: SVA assertion failed at time 250000000
// CHECK: SVA assertion failed at time 350000000
// CHECK: SVA assertion failed at time 450000000
// CHECK: 5 SVA assertion failure(s)
// CHECK: exit code 1

// Simple clocked assertion: property is always false, asserted on posedge clk.
// Clock toggles every 50ns (period 100ns), so 5 posedges in 500ns.

module {
  hw.module @top() {
    %false = hw.constant false
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c50000000_i64 = hw.constant 50000000 : i64
    %c1000000000_i64 = hw.constant 1000000000 : i64
    %true = hw.constant true
    %1 = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %2 = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>

    // Clock signal, starts at 0
    %c0_i2 = hw.constant 0 : i2
    %3 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %clk = llhd.sig %3 : !hw.struct<value: i1, unknown: i1>

    // Initial drive: clk = 0
    llhd.drv %clk, %1 after %0 : !hw.struct<value: i1, unknown: i1>

    // Clock toggle process: every 50ns, clk = ~clk
    %9:2 = llhd.process -> !hw.struct<value: i1, unknown: i1>, i1 {
      cf.br ^bb1(%3, %false : !hw.struct<value: i1, unknown: i1>, i1)
    ^bb1(%10: !hw.struct<value: i1, unknown: i1>, %11: i1):
      %12 = llhd.int_to_time %c50000000_i64
      llhd.wait yield (%10, %11 : !hw.struct<value: i1, unknown: i1>, i1), delay %12, ^bb2
    ^bb2:
      %13 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value = hw.struct_extract %13["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown = hw.struct_extract %13["unknown"] : !hw.struct<value: i1, unknown: i1>
      %14 = comb.xor %value, %true : i1
      %15 = comb.xor %unknown, %true : i1
      %16 = comb.and %14, %15 : i1
      %17 = hw.struct_create (%16, %unknown) : !hw.struct<value: i1, unknown: i1>
      cf.br ^bb1(%17, %true : !hw.struct<value: i1, unknown: i1>, i1)
    }
    llhd.drv %clk, %9#0 after %0 if %9#1 : !hw.struct<value: i1, unknown: i1>

    // Property: always false (constant 0)
    %4 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
    %clk_value = hw.struct_extract %4["value"] : !hw.struct<value: i1, unknown: i1>
    %clk_unknown = hw.struct_extract %4["unknown"] : !hw.struct<value: i1, unknown: i1>
    %clk_known = comb.xor %clk_unknown, %true : i1
    %clk_active = comb.and bin %clk_value, %clk_known : i1
    // Property is always false
    verif.clocked_assert %false, posedge %clk_active : i1

    // Terminate after 1000ns
    llhd.process {
      %t = llhd.int_to_time %c1000000000_i64
      llhd.wait delay %t, ^bb1
    ^bb1:
      sim.terminate success, quiet
      llhd.halt
    }
    hw.output
  }
}
