// RUN: circt-bmc --emit-mlir -b 2 --module top %s | FileCheck %s

// CHECK: smt.solver
// CHECK: func.func @bmc_circuit

module {
  hw.module @top(in %i : !hw.struct<value: i1, unknown: i1>, out o : !hw.struct<value: i1, unknown: i1>, out B.x : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, out B.y : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, out A.y : !llhd.ref<!hw.struct<value: i1, unknown: i1>>) {
    %true = hw.constant true
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i2 = hw.constant 0 : i2
    %1 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %i_0 = llhd.sig name "i" %1 : !hw.struct<value: i1, unknown: i1>
    %o = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    %A.y, %A.i = hw.instance "A" @A(B.x: %B.x: !llhd.ref<!hw.struct<value: i1, unknown: i1>>, B.y: %B.y: !llhd.ref<!hw.struct<value: i1, unknown: i1>>) -> (y: !llhd.ref<!hw.struct<value: i1, unknown: i1>>, i: !llhd.ref<!hw.struct<value: i1, unknown: i1>>)
    %B.x, %B.y, %B.o = hw.instance "B" @B(A.y: %A.y: !llhd.ref<!hw.struct<value: i1, unknown: i1>>) -> (x: !llhd.ref<!hw.struct<value: i1, unknown: i1>>, y: !llhd.ref<!hw.struct<value: i1, unknown: i1>>, o: !llhd.ref<!hw.struct<value: i1, unknown: i1>>)
    %2 = llhd.prb %i_0 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %A.i, %2 after %0 : !hw.struct<value: i1, unknown: i1>
    %3 = llhd.prb %B.o : !hw.struct<value: i1, unknown: i1>
    llhd.drv %o, %3 after %0 : !hw.struct<value: i1, unknown: i1>
    %4 = llhd.prb %o : !hw.struct<value: i1, unknown: i1>
    %value = hw.struct_extract %4["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %4["unknown"] : !hw.struct<value: i1, unknown: i1>
    %5 = comb.xor %unknown, %true : i1
    %6 = comb.and bin %value, %5 : i1
    %value_1 = hw.struct_extract %2["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_2 = hw.struct_extract %2["unknown"] : !hw.struct<value: i1, unknown: i1>
    %7 = comb.xor %unknown_2, %true : i1
    %8 = comb.and bin %value_1, %7 : i1
    llhd.process {
      cf.br ^bb2(%4, %2 : !hw.struct<value: i1, unknown: i1>, !hw.struct<value: i1, unknown: i1>)
    ^bb1:  // pred: ^bb2
      %9 = llhd.prb %o : !hw.struct<value: i1, unknown: i1>
      %10 = llhd.prb %i_0 : !hw.struct<value: i1, unknown: i1>
      cf.br ^bb2(%9, %10 : !hw.struct<value: i1, unknown: i1>, !hw.struct<value: i1, unknown: i1>)
    ^bb2(%11: !hw.struct<value: i1, unknown: i1>, %12: !hw.struct<value: i1, unknown: i1>):  // 2 preds: ^bb0, ^bb1
      %value_3 = hw.struct_extract %11["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_4 = hw.struct_extract %11["unknown"] : !hw.struct<value: i1, unknown: i1>
      %value_5 = hw.struct_extract %12["value"] : !hw.struct<value: i1, unknown: i1>
      %unknown_6 = hw.struct_extract %12["unknown"] : !hw.struct<value: i1, unknown: i1>
      %13 = comb.icmp eq %value_3, %value_5 : i1
      %14 = comb.or %unknown_4, %unknown_6 : i1
      %15 = comb.xor %14, %true : i1
      %16 = comb.and %15, %13 : i1
      verif.assert %16 label "" : i1
      llhd.wait (%6, %8 : i1, i1), ^bb1
    }
    llhd.drv %i_0, %i after %0 : !hw.struct<value: i1, unknown: i1>
    hw.output %4, %B.x, %B.y, %A.y : !hw.struct<value: i1, unknown: i1>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  }
  hw.module private @A(in %B.x : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, in %B.y : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, out y : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, out i : !llhd.ref<!hw.struct<value: i1, unknown: i1>>) {
    %true = hw.constant true
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i2 = hw.constant 0 : i2
    %1 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %i = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    %y = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %y, %7 after %0 : !hw.struct<value: i1, unknown: i1>
    %2 = llhd.prb %i : !hw.struct<value: i1, unknown: i1>
    %value = hw.struct_extract %2["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown = hw.struct_extract %2["unknown"] : !hw.struct<value: i1, unknown: i1>
    %3 = comb.xor %value, %true : i1
    %4 = hw.struct_create (%3, %unknown) : !hw.struct<value: i1, unknown: i1>
    llhd.drv %B.x, %4 after %0 : !hw.struct<value: i1, unknown: i1>
    %5 = llhd.prb %B.y : !hw.struct<value: i1, unknown: i1>
    %value_0 = hw.struct_extract %5["value"] : !hw.struct<value: i1, unknown: i1>
    %unknown_1 = hw.struct_extract %5["unknown"] : !hw.struct<value: i1, unknown: i1>
    %6 = comb.xor %value_0, %true : i1
    %7 = hw.struct_create (%6, %unknown_1) : !hw.struct<value: i1, unknown: i1>
    hw.output %y, %i : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  }
  hw.module private @B(out x : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, out y : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, in %A.y : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, out o : !llhd.ref<!hw.struct<value: i1, unknown: i1>>) {
    %0 = llhd.constant_time <0ns, 0d, 1e>
    %c0_i2 = hw.constant 0 : i2
    %1 = hw.bitcast %c0_i2 : (i2) -> !hw.struct<value: i1, unknown: i1>
    %x = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    %y = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %y, %2 after %0 : !hw.struct<value: i1, unknown: i1>
    %o = llhd.sig %1 : !hw.struct<value: i1, unknown: i1>
    llhd.drv %o, %3 after %0 : !hw.struct<value: i1, unknown: i1>
    %2 = llhd.prb %x : !hw.struct<value: i1, unknown: i1>
    %3 = llhd.prb %A.y : !hw.struct<value: i1, unknown: i1>
    hw.output %x, %y, %o : !llhd.ref<!hw.struct<value: i1, unknown: i1>>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>, !llhd.ref<!hw.struct<value: i1, unknown: i1>>
  }
}
