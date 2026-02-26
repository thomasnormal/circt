// RUN: circt-verilog %s --ir-moore | FileCheck %s

// Test hierarchical interface task calls through module-instance arrays:
//   module_array[idx].interface_array[idx].task()
// CHECK-LABEL: moore.module @Top()
// CHECK: moore.procedure initial {
// CHECK-COUNT-2: func.call @"IF::ping{{(_[0-9]+)?}}"

interface IF(input bit clk);
  task ping();
    @(posedge clk);
  endtask
endinterface

module Agent(input bit clk);
  IF ifs[2](clk);
endmodule

module Top #(parameter int AIDX = 1, parameter int IIDX = 1);
  bit clk = 0;
  Agent a[2](clk);
  localparam int LIDX = 1;

  initial begin
    a[AIDX].ifs[IIDX].ping();
    a[LIDX].ifs[1 - 0].ping();
  end
endmodule
