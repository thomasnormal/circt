// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test basic UVM DPI-C imports
// Verify that DPI functions are declared and called properly

module uvm_dpi_basic_test;
  // Import UVM DPI functions
  import "DPI-C" function int uvm_hdl_deposit(string path, longint value);
  import "DPI-C" function chandle uvm_re_comp(string pattern, bit deglob);
  import "DPI-C" function int uvm_re_exec(chandle rexp, string str);
  import "DPI-C" function void uvm_re_free(chandle rexp);
  import "DPI-C" function string uvm_dpi_get_tool_name_c();
  import "DPI-C" function int uvm_re_compexecfree(string re, string str, bit deglob, output int exec_ret);

  initial begin
    int status;
    int exec_result;
    string tool;
    chandle regex;

    // Test HDL deposit
    status = uvm_hdl_deposit("top.dut.signal", 64'hDEADBEEF);
    // CHECK: func.call @uvm_hdl_deposit

    // Test regex compile
    regex = uvm_re_comp("test.*", 1'b0);
    // CHECK: func.call @uvm_re_comp

    // Test regex exec
    status = uvm_re_exec(regex, "test_string");
    // CHECK: func.call @uvm_re_exec

    // Test regex free
    uvm_re_free(regex);
    // CHECK: func.call @uvm_re_free

    // Test tool name
    tool = uvm_dpi_get_tool_name_c();
    // CHECK: func.call @uvm_dpi_get_tool_name_c

    // Test combined regex operation
    status = uvm_re_compexecfree("pattern", "string", 1'b0, exec_result);
    // CHECK: func.call @uvm_re_compexecfree
  end

  // Verify function declarations
  // CHECK-DAG: func.func private @uvm_hdl_deposit(!moore.string, !moore.i64) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_re_comp(!moore.string, !moore.i1) -> !moore.chandle
  // CHECK-DAG: func.func private @uvm_re_exec(!moore.chandle, !moore.string) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_re_free(!moore.chandle)
  // CHECK-DAG: func.func private @uvm_dpi_get_tool_name_c() -> !moore.string
  // CHECK-DAG: func.func private @uvm_re_compexecfree(!moore.string, !moore.string, !moore.i1, !moore.ref<i32>) -> !moore.i32
endmodule
