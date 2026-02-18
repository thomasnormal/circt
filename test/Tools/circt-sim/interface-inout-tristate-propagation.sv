// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd -o %t.mlir
// RUN: circt-sim %t.mlir --top top --max-time=50000000 2>&1 | FileCheck %s

// Regression for interface-body continuous assigns on inout ports.
// The interface drives inout S via (s_oe ? s_o : 1'bz) and mirrors s_i = S.
// If interface continuous assigns are not instantiated per interface instance,
// s_i stays X and monitor checks fail.

interface tri_if(input logic clk, inout logic S);
  logic s_i;
  logic s_o;
  logic s_oe;

  assign S = s_oe ? s_o : 1'bz;
  assign s_i = S;
endinterface

module driver(tri_if intf);
  initial begin
    intf.s_o = 1'b0;
    intf.s_oe = 1'b0;

    #5;
    intf.s_o = 1'b1;
    intf.s_oe = 1'b1;

    #5;
    intf.s_o = 1'b0;
    intf.s_oe = 1'b1;

    #5;
    intf.s_o = 1'b1;
    intf.s_oe = 1'b0;
  end
endmodule

module monitor(tri_if intf);
  initial begin
    #6;
    if (intf.s_i === 1'b1)
      $display("MON_HIGH_OK");
    else
      $display("MON_HIGH_FAIL:%b", intf.s_i);

    #5;
    if (intf.s_i === 1'b0)
      $display("MON_LOW_OK");
    else
      $display("MON_LOW_FAIL:%b", intf.s_i);

    #6;
    if (intf.s_i === 1'b1)
      $display("MON_RELEASE_HIGH_OK");
    else
      $display("MON_RELEASE_HIGH_FAIL:%b", intf.s_i);

    $finish;
  end
endmodule

module top;
  logic clk = 1'b0;
  wire S;
  pullup(S);
  always #1 clk = ~clk;

  tri_if bus(clk, S);
  driver d(bus);
  monitor m(bus);

  // CHECK: MON_HIGH_OK
  // CHECK: MON_LOW_OK
  // CHECK: MON_RELEASE_HIGH_OK
  // CHECK-NOT: MON_{{.*}}_FAIL:
endmodule
