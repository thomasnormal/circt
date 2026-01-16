// RUN: circt-verilog --ir-moore %s 2>&1 | FileCheck %s

// Test that DPI functions for tool identification return proper values

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

// CHECK: moore.constant_string "CIRCT" : i40
// CHECK: moore.int_to_string
// CHECK: moore.constant_string "1.0" : i24
// CHECK: moore.int_to_string
