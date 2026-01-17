// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI functions for tool identification lower to runtime stub calls

import "DPI-C" function string uvm_dpi_get_tool_name_c();
import "DPI-C" function string uvm_dpi_get_tool_version_c();

module test;
  string tool_name;
  string tool_version;

  initial begin
    tool_name = uvm_dpi_get_tool_name_c();
    tool_version = uvm_dpi_get_tool_version_c();
  end
endmodule

// CHECK: func.call @uvm_dpi_get_tool_name_c
// CHECK: func.call @uvm_dpi_get_tool_version_c
