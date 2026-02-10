// RUN: circt-verilog %s --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=500000000 2>&1 | FileCheck %s

// Test that a DUT with interface signals passed through module ports correctly
// derives sensitivity from the interface field shadow signals. Previously, the
// DUT process would fall back to delta-cycle polling (causing delta overflow)
// because resolveSignalId could not trace through the
// unrealized_conversion_cast(llhd.prb(sig)) pattern used for module port wiring.

interface simple_if(input clk);
  logic [7:0] data;
endinterface

module dut(simple_if in_if, simple_if out_if);
  always @(posedge in_if.clk) begin
    out_if.data <= in_if.data;
  end
endmodule

module top;
  logic clk = 0;
  always #5 clk = ~clk;

  simple_if in_port(clk);
  simple_if out_port(clk);

  dut d(.in_if(in_port), .out_if(out_port));

  initial begin
    in_port.data = 8'd42;
    repeat(3) @(posedge clk);
    // CHECK: PASS: out_port.data = 42
    $display("PASS: out_port.data = %0d", out_port.data);
    $finish;
  end
endmodule
