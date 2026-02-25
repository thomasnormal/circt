// RUN: circt-bmc -b 1 --allow-multi-clock --module top --emit-mlir %s | FileCheck %s

// Ensure mixed explicit + 4-state clocked-assert inputs are both mapped as BMC
// clocks. Previously this failed with:
//   clocked property uses a clock that is not a BMC clock input
// CHECK-LABEL: func.func @top()
// CHECK: smt.solver
// CHECK: smt.declare_fun "aux_clk"
// CHECK: func.call @bmc_circuit

hw.module @top(in %clk_explicit : !seq.clock,
               in %aux_clk : !hw.struct<value: i1, unknown: i1>,
               in %in : i1) {
  %r = seq.compreg %in, %clk_explicit : i1
  %true = hw.constant true
  %aux_v = hw.struct_extract %aux_clk["value"] : !hw.struct<value: i1, unknown: i1>
  %aux_u = hw.struct_extract %aux_clk["unknown"] : !hw.struct<value: i1, unknown: i1>
  %aux_nu = comb.xor %aux_u, %true : i1
  %aux_i1 = comb.and bin %aux_v, %aux_nu : i1
  verif.clocked_assert %r, posedge %aux_i1 : i1
  hw.output
}
