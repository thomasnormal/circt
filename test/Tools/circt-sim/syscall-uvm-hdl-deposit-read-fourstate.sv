// RUN: circt-verilog %s --no-uvm-auto-include --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Regression: uvm_hdl_deposit/uvm_hdl_read on 4-state signals must use the
// value field (upper half of {value,unknown}) and keep unknown bits clear.

import "DPI-C" function int uvm_hdl_deposit(string path, input bit [31:0] val);
import "DPI-C" function int uvm_hdl_read(string path, output bit [31:0] val);

module top;
  reg [31:0] sig;
  bit [31:0] readback;
  int dep_ok;
  int rd_ok;

  initial begin
    sig = 32'h11;
    #1;

    dep_ok = uvm_hdl_deposit("top.sig", 32'hAA);
    #1;

    readback = '0;
    rd_ok = uvm_hdl_read("top.sig", readback);

    $display("DEP=%0d RD=%0d SIG=%0h RB=%0h", dep_ok, rd_ok, sig, readback);
    // CHECK: DEP=1 RD=1 SIG=aa RB=aa

    #1 $finish;
  end
endmodule
