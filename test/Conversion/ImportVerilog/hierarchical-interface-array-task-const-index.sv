// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface array task calls where the array index is a
// constant expression (parameter or folded arithmetic), not a literal token.
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.procedure initial {
// CHECK-COUNT-3: func.call @"IF::ping{{(_[0-9]+)?}}"

interface IF(input bit clk);
  task ping();
    @(posedge clk);
  endtask
endinterface

module M(input bit clk);
  IF ifs[2](clk);
endmodule

module Top #(parameter int PIDX = 1);
  bit clk = 0;
  M m(clk);
  localparam int LIDX = 1;

  initial begin
    m.ifs[PIDX].ping();
    m.ifs[LIDX].ping();
    m.ifs[1 - 0].ping();
  end
endmodule
