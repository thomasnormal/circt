// RUN: circt-bmc --emit-mlir -b 10 --module top %s | FileCheck %s
// XFAIL: *
// Test file has SSA error: %38 and %42 defined inside process but used outside.

module {
  hw.module private @clk_gen(in %valid : !hw.struct<value: i1, unknown: i1>, in %clk : !hw.struct<value: i1, unknown: i1>, out out : !hw.struct<value: i8, unknown: i8>, in %in : !hw.struct<value: i8, unknown: i8>) {
    %c0_i8 = hw.constant 0 : i8
    %c1_i8 = hw.constant 1 : i8
    %0 = llhd.constant_time <0ns, 1d, 0e>
    %1 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i16 = hw.constant 0 : i16
    %true = hw.constant true
    %c0_i2 = hw.constant 0 : i2
    %2 = hw.aggregate_constant [0 : i8, 0 : i8] : !hw.struct<value: i8, unknown: i8>
    %3 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %valid_0 = llhd.sig name "valid" %3 : !hw.struct<value: i1, unknown: i1>
    %clk_1 = llhd.sig name "clk" %3 : !hw.struct<value: i1, unknown: i1>
    %4 = llhd.prb %clk_1 : !hw.struct<value: i1, unknown: i1>
    %value = hw.struct_extract %4["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %4["unknown"] : !hw.struct<value: i1, unknown: i1>
    %5 = comb.xor %unknown, %true : i1
    %6 = comb.and bin %value, %5 : i1
    %7 = hw.bitcast %c0_i16 : (i16) -> !hw.struct<value: i8, unknown: i8>
    %out = llhd.sig %7 : !hw.struct<value: i8, unknown: i8>
    %in_2 = llhd.sig name "in" %7 : !hw.struct<value: i8, unknown: i8>
    %data_reg_0 = llhd.sig %7 : !hw.struct<value: i8, unknown: i8>
    %data_reg_1 = llhd.sig %7 : !hw.struct<value: i8, unknown: i8>
    %data_reg_2 = llhd.sig %7 : !hw.struct<value: i8, unknown: i8>
    llhd.process {
      llhd.drv %data_reg_0, %2 after %1 : !hw.struct<value: i8, unknown: i8>
      llhd.drv %data_reg_1, %2 after %1 : !hw.struct<value: i8, unknown: i8>
      llhd.drv %data_reg_2, %2 after %1 : !hw.struct<value: i8, unknown: i8>
      llhd.drv %out, %2 after %1 : !hw.struct<value: i8, unknown: i8>
      llhd.halt
    }
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 4 preds: ^bb0, ^bb2, ^bb3, ^bb4
      %9 = llhd.prb %clk_1 : !hw.struct<value: i1, unknown: i1>
      %value_3 = hw.struct_extract %9["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_4 = hw.struct_extract %9["unknown"] : !hw.struct<value: i1, unknown: i1>
      %10 = comb.xor %unknown_4, %true : i1
      %11 = comb.and bin %value_3, %10 : i1
      llhd.wait (%6 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %12 = llhd.prb %clk_1 : !hw.struct<value: i1, unknown: i1>
      %value_5 = hw.struct_extract %12["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_6 = hw.struct_extract %12["unknown"] : !hw.struct<value: i1, unknown: i1>
      %13 = comb.xor %unknown_6, %true : i1
      %14 = comb.xor bin %11, %true : i1
      %15 = comb.and bin %14, %value_5, %13 : i1
      cf.cond_br %15, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %16 = llhd.prb %valid_0 : !hw.struct<value: i1, unknown: i1>
      %value_7 = hw.struct_extract %16["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_8 = hw.struct_extract %16["unknown"] : !hw.struct<value: i1, unknown: i1>
      %17 = comb.xor %unknown_8, %true : i1
      %18 = comb.and bin %value_7, %17 : i1
      cf.cond_br %18, ^bb4, ^bb1
    ^bb4:  // pred: ^bb3
      %19 = llhd.prb %in_2 : !hw.struct<value: i8, unknown: i8>
      %value_9 = hw.struct_extract %19["value"] : !hw.struct<value: i8, unknown: i8>
      %unknown_10 = hw.struct_extract %19["unknown"] : !hw.struct<value: i8, unknown: i8>
      %20 = comb.add %value_9, %c1_i8 : i8
      %21 = comb.icmp ne %unknown_10, %c0_i8 : i8
      %22 = comb.replicate %21 : (i1) -> i8
      %23 = hw.struct_create (%20, %22) : !hw.struct<value: i8, unknown: i8>
      llhd.drv %data_reg_0, %23 after %0 : !hw.struct<value: i8, unknown: i8>
      %24 = llhd.prb %data_reg_0 : !hw.struct<value: i8, unknown: i8>
      %value_11 = hw.struct_extract %24["value"] : !hw.struct<value: i8, unknown: i8>
      %unknown_12 = hw.struct_extract %24["unknown"] : !hw.struct<value: i8, unknown: i8>
      %25 = comb.add %value_11, %c1_i8 : i8
      %26 = comb.icmp ne %unknown_12, %c0_i8 : i8
      %27 = comb.replicate %26 : (i1) -> i8
      %28 = hw.struct_create (%25, %27) : !hw.struct<value: i8, unknown: i8>
      llhd.drv %data_reg_1, %28 after %0 : !hw.struct<value: i8, unknown: i8>
      %29 = llhd.prb %data_reg_1 : !hw.struct<value: i8, unknown: i8>
      %value_13 = hw.struct_extract %29["value"] : !hw.struct<value: i8, unknown: i8>
      %unknown_14 = hw.struct_extract %29["unknown"] : !hw.struct<value: i8, unknown: i8>
      %30 = comb.add %value_13, %c1_i8 : i8
      %31 = comb.icmp ne %unknown_14, %c0_i8 : i8
      %32 = comb.replicate %31 : (i1) -> i8
      %33 = hw.struct_create (%30, %32) : !hw.struct<value: i8, unknown: i8>
      llhd.drv %data_reg_2, %33 after %0 : !hw.struct<value: i8, unknown: i8>
      %34 = llhd.prb %data_reg_2 : !hw.struct<value: i8, unknown: i8>
      %value_15 = hw.struct_extract %34["value"] : !hw.struct<value: i8, unknown: i8>
      %unknown_16 = hw.struct_extract %34["unknown"] : !hw.struct<value: i8, unknown: i8>
      %35 = comb.add %value_15, %c1_i8 : i8
      %36 = comb.icmp ne %unknown_16, %c0_i8 : i8
      %37 = comb.replicate %36 : (i1) -> i8
      %38 = hw.struct_create (%35, %37) : !hw.struct<value: i8, unknown: i8>
      llhd.drv %out, %38 after %0 : !hw.struct<value: i8, unknown: i8>
      cf.br ^bb1
    }
    llhd.drv %valid_0, %valid after %1 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %clk_1, %clk after %1 : !hw.struct<value: i1, unknown: i1>
    %8 = llhd.prb %out : !hw.struct<value: i8, unknown: i8>
    llhd.drv %in_2, %in after %1 : !hw.struct<value: i8, unknown: i8>
    hw.output %8 : !hw.struct<value: i8, unknown: i8>
  }
  hw.module @top() {
    %c0_i8 = hw.constant 0 : i8
    %c0_i24 = hw.constant 0 : i24
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i2 = hw.constant 0 : i2
    %true = hw.constant true
    %c1000000000_i64 = hw.constant 1000000000 : i64
    %c50000000_i64 = hw.constant 50000000 : i64
    %c1_i32 = hw.constant 1 : i32
    %c3_i32 = hw.constant 3 : i32
    %c0_i32 = hw.constant 0 : i32
    %1 = hw.aggregate_constant [true, false] : !hw.struct<value: i1, unknown: i1>
    %2 = hw.aggregate_constant [false, false] : !hw.struct<value: i1, unknown: i1>
    %cycle = llhd.sig %c0_i32 : i32
    %3 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %valid = llhd.sig %3 : !hw.struct<value: i1, unknown: i1>
    %clk = llhd.sig %3 : !hw.struct<value: i1, unknown: i1>
    %4 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
    %value = hw.struct_extract %4["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %4["unknown"] : !hw.struct<value: i1, unknown: i1>
    %5 = comb.xor %unknown, %true : i1
    %6 = comb.and bin %value, %5 : i1
    %7 = llhd.prb %valid : !hw.struct<value: i1, unknown: i1>
    %dut.out = hw.instance "dut" @clk_gen(valid: %7: !hw.struct<value: i1, unknown: i1>, clk: %4: !hw.struct<value: i1, unknown: i1>, in: %29: !hw.struct<value: i8, unknown: i8>) -> (out: !hw.struct<value: i8, unknown: i8>) {sv.namehint = "out"}
    llhd.process {
      llhd.drv %cycle, %c0_i32 after %0 : i32
      llhd.drv %clk, %2 after %0 : !hw.struct<value: i1, unknown: i1>
      llhd.drv %valid, %1 after %0 : !hw.struct<value: i1, unknown: i1>
      llhd.halt
    }
    %value_0 = hw.struct_extract %7["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_1 = hw.struct_extract %7["unknown"] : !hw.struct<value: i1, unknown: i1>
    %8 = comb.xor %unknown_1, %true : i1
    %9 = comb.and bin %value_0, %8 : i1
    %10 = comb.concat %c0_i24, %28 : i24, i8
    %value_2 = hw.struct_extract %dut.out["value"] : !hw.struct<value: i8, unknown: i8>
    %unknown_3 = hw.struct_extract %dut.out["unknown"] : !hw.struct<value: i8, unknown: i8>
    %11 = comb.concat %c0_i24, %value_2 : i24, i8
    %12 = seq.to_clock %26
    %13 = seq.compreg %10, %12 : i32  
    %14 = seq.compreg %13, %12 : i32  
    %15 = seq.compreg %14, %12 : i32  
    %16 = seq.compreg %15, %12 : i32  
    %17 = comb.add %16, %c3_i32 : i32
    %18 = comb.icmp eq %11, %17 : i32
    %19 = comb.icmp ne %unknown_3, %c0_i8 : i8
    %20 = comb.xor %19, %true : i1
    %21 = comb.xor %19, %true : i1
    %22 = comb.and bin %20, %18, %21 : i1
    %23 = ltl.delay %22, 4, 0 : i1
    %24 = ltl.implication %9, %23 : i1, !ltl.sequence
    %25 = comb.xor %unknown, %true : i1
    %26 = comb.and bin %value, %25 : i1
    verif.clocked_assert %24, posedge %26 : !ltl.property
    %27 = llhd.prb %cycle : i32
    %28 = comb.extract %27 from 0 : (i32) -> i8
    %29 = hw.struct_create (%28, %c0_i8) {sv.namehint = "in"} : !hw.struct<value: i8, unknown: i8>
    llhd.process {
      cf.br ^bb1
    ^bb1:  // 3 preds: ^bb0, ^bb2, ^bb3
      %30 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value_4 = hw.struct_extract %30["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_5 = hw.struct_extract %30["unknown"] : !hw.struct<value: i1, unknown: i1>
      %31 = comb.xor %unknown_5, %true : i1
      %32 = comb.and bin %value_4, %31 : i1
      llhd.wait (%6 : i1), ^bb2
    ^bb2:  // pred: ^bb1
      %33 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value_6 = hw.struct_extract %33["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_7 = hw.struct_extract %33["unknown"] : !hw.struct<value: i1, unknown: i1>
      %34 = comb.xor %unknown_7, %true : i1
      %35 = comb.xor bin %32, %true : i1
      %36 = comb.and bin %35, %value_6, %34 : i1
      cf.cond_br %36, ^bb3, ^bb1
    ^bb3:  // pred: ^bb2
      %37 = llhd.prb %cycle : i32
      %38 = comb.add %37, %c1_i32 : i32
      cf.br ^bb1
    }
    llhd.drv %cycle, %38 after %0 : i32
    llhd.process {
      %39 = llhd.int_to_time %c50000000_i64
      llhd.wait delay %39, ^bb1
    ^bb1:  // pred: ^bb0
      %40 = llhd.prb %clk : !hw.struct<value: i1, unknown: i1>
      %value_8 = hw.struct_extract %40["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_9 = hw.struct_extract %40["unknown"] : !hw.struct<value: i1, unknown: i1>
      %41 = comb.xor %value_8, %true : i1
      %42 = hw.struct_create (%41, %unknown_9) : !hw.struct<value: i1, unknown: i1>
      llhd.wait yield (%42 : !hw.struct<value: i1, unknown: i1>), delay %39, ^bb1
    }
    llhd.drv %clk, %42 after %0 : !hw.struct<value: i1, unknown: i1>
    llhd.process {
      %43 = llhd.int_to_time %c1000000000_i64
      llhd.wait delay %43, ^bb1
    ^bb1:  // pred: ^bb0
      llhd.halt
    }
    hw.output
  }
}

// CHECK: verif.bmc
