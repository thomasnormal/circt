// RUN: not circt-sim %s --top top --max-time=100000000 2>&1 | FileCheck %s
// CHECK: SVA assertion failed at time
// CHECK: SVA assertion failure(s)
// CHECK: exit code 1

// Runtime semantics: ltl.past observes the input from previous sampled cycles.
// For constant false input with delay=1, the assertion should fail once a
// previous sample exists.

module {
  hw.module @top() {
    %c-1_i32 = hw.constant -1 : i32
    %c0_i32 = hw.constant 0 : i32
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %true = hw.constant true
    %c5000000_i64 = hw.constant 5000000 : i64
    %false = hw.constant false
    %c4_i32 = hw.constant 4 : i32
    %1 = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %clk = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    %2 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
    %value = hw.struct_extract %2["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %2["unknown"] : !hw.struct<value: i1, unknown: i1>
    %3 = comb.xor %unknown, %true : i1
    %4 = comb.and bin %value, %3 : i1
    llhd.process {
      llhd.drv %clk, %1 after %0 : !hw.struct<value: i1, unknown: i1>
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      %7 = llhd.int_to_time %c5000000_i64
      llhd.wait delay %7, ^bb2
    ^bb2:  // pred: ^bb1
      %8 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value_0 = hw.struct_extract %8["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_1 = hw.struct_extract %8["unknown"] : !hw.struct<value: i1, unknown: i1>
      %9 = comb.xor %value_0, %true : i1
      %10 = comb.xor %unknown_1, %true : i1
      %11 = comb.and %9, %10 : i1
      %12 = hw.struct_create (%11, %unknown_1) : !hw.struct<value: i1, unknown: i1>
      llhd.drv %clk, %12 after %0 : !hw.struct<value: i1, unknown: i1>
      cf.br ^bb1
    }
    llhd.process {
      cf.br ^bb1(%c4_i32 : i32)
    ^bb1(%7: i32):  // 2 preds: ^bb0, ^bb4
      %8 = comb.icmp ne %7, %c0_i32 : i32
      cf.cond_br %8, ^bb2, ^bb5
    ^bb2:  // 2 preds: ^bb1, ^bb3
      %9 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value_0 = hw.struct_extract %9["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_1 = hw.struct_extract %9["unknown"] : !hw.struct<value: i1, unknown: i1>
      %10 = comb.xor %unknown_1, %true : i1
      %11 = comb.and bin %value_0, %10 : i1
      llhd.wait (%4 : i1), ^bb3
    ^bb3:  // pred: ^bb2
      %12 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value_2 = hw.struct_extract %12["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_3 = hw.struct_extract %12["unknown"] : !hw.struct<value: i1, unknown: i1>
      %13 = comb.xor %unknown_3, %true : i1
      %14 = comb.xor bin %11, %true : i1
      %15 = comb.and bin %14, %value_2, %13 : i1
      cf.cond_br %15, ^bb4, ^bb2
    ^bb4:  // pred: ^bb3
      %16 = comb.add %7, %c-1_i32 : i32
      cf.br ^bb1(%16 : i32)
    ^bb5:  // pred: ^bb1
      sim.terminate success, quiet
      llhd.halt
    }
    %5 = comb.xor %unknown, %true : i1
    %6 = comb.and bin %value, %5 : i1
    %past = ltl.past %false, 1 : i1
    verif.clocked_assert %past, posedge %6 : !ltl.sequence
    hw.output
  }
}
