// RUN: circt-opt %s --convert-verif-to-smt --reconcile-unrealized-casts \
// RUN:   -allow-unregistered-dialect | FileCheck %s
//

// CHECK-LABEL: func.func @clock_source_struct_invert
// CHECK: [[LOOP:%.+]] = func.call @bmc_loop
// CHECK: [[NOT:%.+]] = smt.bv.not [[LOOP]] : !smt.bv<1>
// CHECK: [[UNKN:%.+]] = smt.bv.extract %arg{{[0-9]+}} from 0 : (!smt.bv<2>) -> !smt.bv<1>
// CHECK: [[CONCAT:%.+]] = smt.bv.concat [[NOT]], [[UNKN]] : !smt.bv<1>, !smt.bv<1>
// CHECK: func.call @bmc_circuit([[LOOP]], [[CONCAT]]
func.func @clock_source_struct_invert() -> i1 {
  %bmc = verif.bmc bound 1 num_regs 0 initial_values [] attributes {
    bmc_clock_sources = [{arg_index = 1 : i32, clock_pos = 0 : i32, invert = true}]
  } init {
    %false = hw.constant false
    %clk = seq.to_clock %false
    verif.yield %clk : !seq.clock
  } loop {
  ^bb0(%clk: !seq.clock):
    verif.yield %clk : !seq.clock
  } circuit {
  ^bb0(%clk: !seq.clock, %sig: !hw.struct<value: i1, unknown: i1>):
    %value = hw.struct_extract %sig["value"] : !hw.struct<value: i1, unknown: i1>
    verif.assert %value : i1
    verif.yield %value : i1
  }
  func.return %bmc : i1
}
