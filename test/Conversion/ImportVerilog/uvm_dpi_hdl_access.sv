// RUN: circt-verilog --no-uvm-auto-include --ir-moore %s | FileCheck %s

// Test UVM HDL access DPI imports beyond deposit.

module uvm_dpi_hdl_access_test;
  import "DPI-C" function int uvm_hdl_check_path(string path);
  import "DPI-C" function int uvm_hdl_force(string path, longint value);
  import "DPI-C" function int uvm_hdl_release(string path);
  import "DPI-C" function int uvm_hdl_read(string path, output longint value);
  import "DPI-C" function int uvm_hdl_release_and_read(string path, output longint value);

  initial begin
    int status;
    longint value;

    status = uvm_hdl_check_path("top.dut.signal");
    // CHECK: func.call @uvm_hdl_check_path

    status = uvm_hdl_force("top.dut.signal", 64'hCAFE);
    // CHECK: func.call @uvm_hdl_force

    status = uvm_hdl_read("top.dut.signal", value);
    // CHECK: func.call @uvm_hdl_read

    status = uvm_hdl_release("top.dut.signal");
    // CHECK: func.call @uvm_hdl_release

    status = uvm_hdl_release_and_read("top.dut.signal", value);
    // CHECK: func.call @uvm_hdl_release_and_read
  end

  // CHECK-DAG: func.func private @uvm_hdl_check_path(!moore.string) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_hdl_force(!moore.string, !moore.i64) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_hdl_release(!moore.string) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_hdl_read(!moore.string, !moore.ref<i64>) -> !moore.i32
  // CHECK-DAG: func.func private @uvm_hdl_release_and_read(!moore.string, !moore.ref<i64>) -> !moore.i32
endmodule
