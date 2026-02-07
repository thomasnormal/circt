// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s
// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s 2>&1 | FileCheck %s --check-prefix=REMARK

// Test DPI-C import declarations and calls
// These should generate function declarations and calls to runtime stubs

module dpi_test;
  // DPI-C imports for different data types
  import "DPI-C" function int dpi_int_func(int x);
  import "DPI-C" function string dpi_string_func();
  import "DPI-C" function chandle dpi_chandle_func();
  import "DPI-C" function void dpi_void_func(int x, string s);

  // UVM-specific DPI imports
  import "DPI-C" context function int uvm_hdl_deposit(string path, longint value);
  import "DPI-C" function chandle uvm_re_comp(string pattern, bit deglob);
  import "DPI-C" function int uvm_re_exec(chandle rexp, string str);
  import "DPI-C" function void uvm_re_free(chandle rexp);
  import "DPI-C" function string uvm_dpi_get_tool_name_c();

  initial begin
    int result;
    string tool_name;
    chandle regex_handle;

    // Call DPI functions
    result = dpi_int_func(42);
    // CHECK: func.call @dpi_int_func

    tool_name = uvm_dpi_get_tool_name_c();
    // CHECK: func.call @uvm_dpi_get_tool_name_c

    // Test regex DPI calls
    regex_handle = uvm_re_comp("test.*pattern", 1'b0);
    // CHECK: func.call @uvm_re_comp

    result = uvm_re_exec(regex_handle, "test_string");
    // CHECK: func.call @uvm_re_exec

    uvm_re_free(regex_handle);
    // CHECK: func.call @uvm_re_free

    // Test HDL access
    result = uvm_hdl_deposit("top.signal", 64'h1234);
    // CHECK: func.call @uvm_hdl_deposit

    // Void function call
    dpi_void_func(123, "hello");
    // CHECK: func.call @dpi_void_func
  end

  // Check that DPI functions are declared (only those that are actually called)
  // CHECK-DAG: func.func private @dpi_int_func(!moore.i32) -> !moore.i32
  // CHECK-DAG: func.func private @dpi_void_func(!moore.i32, !moore.string)
  // CHECK-DAG: func.func private @uvm_hdl_deposit(!moore.string, !moore.i64) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_re_comp(!moore.string, !moore.i1) -> !moore.chandle
  // CHECK-DAG: func.func private @uvm_re_exec(!moore.chandle, !moore.string) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_re_free(!moore.chandle)
  // CHECK-DAG: func.func private @uvm_dpi_get_tool_name_c() -> !moore.string

  // Check that remarks are emitted for DPI imports (only for functions that are called)
  // REMARK-DAG: remark: DPI-C import 'dpi_int_func' will use runtime stub (link with MooreRuntime)
  // REMARK-DAG: remark: DPI-C import 'dpi_void_func' will use runtime stub (link with MooreRuntime)
  // REMARK-DAG: remark: DPI-C import 'uvm_hdl_deposit' will use runtime stub (link with MooreRuntime)
  // REMARK-DAG: remark: DPI-C import 'uvm_re_comp' will use runtime stub (link with MooreRuntime)
  // REMARK-DAG: remark: DPI-C import 'uvm_re_exec' will use runtime stub (link with MooreRuntime)
  // REMARK-DAG: remark: DPI-C import 'uvm_re_free' will use runtime stub (link with MooreRuntime)
  // REMARK-DAG: remark: DPI-C import 'uvm_dpi_get_tool_name_c' will use runtime stub (link with MooreRuntime)
endmodule
