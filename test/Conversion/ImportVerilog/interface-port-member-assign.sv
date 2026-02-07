// RUN: circt-verilog --no-uvm-auto-include %s --verify-diagnostics

// Test that interface port member assignments work correctly.
// This tests the fix for the case where continuous assignments inside a module
// access members of interface ports. Slang categorizes these as hierarchical
// references, but they should be treated as interface port member accesses
// rather than as hierarchical paths that need to be threaded through the
// module hierarchy.

interface SimpleIface(input logic clk);
  logic data_in;
  logic data_out;
endinterface

// Module that takes an interface port and assigns one member to another
module SubModule(SimpleIface iface);
  // This assign should work - accessing members of interface port
  assign iface.data_out = iface.data_in;
endmodule

// Top module that instantiates the interface and sub-module
module TopModule;
  logic clk;
  SimpleIface intf(.clk(clk));
  SubModule sub(.iface(intf));
endmodule
