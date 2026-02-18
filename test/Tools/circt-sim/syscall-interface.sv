// RUN: circt-verilog %s -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test interface, modport, virtual interface
interface simple_if;
  logic clk;
  logic [7:0] data;

  modport master(output clk, output data);
  modport slave(input clk, input data);
endinterface

module driver(simple_if.master m);
  initial begin
    m.data = 8'hAB;
  end
endmodule

module top;
  simple_if intf();
  driver drv(.m(intf));

  initial begin
    #1;
    // CHECK: data=ab
    $display("data=%h", intf.data);
    $finish;
  end
endmodule
