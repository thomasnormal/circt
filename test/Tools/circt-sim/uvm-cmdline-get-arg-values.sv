// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: env CIRCT_UVM_ARGS="+MY_OPT=hello +OTHER=world" circt-sim %t.mlir --top top 2>&1 | FileCheck %s

module top;
  import uvm_pkg::*;
  `include "uvm_macros.svh"

  string vals[$];
  int n;

  initial begin
    uvm_cmdline_processor clp;
    clp = uvm_cmdline_processor::get_inst();
    n = clp.get_arg_values("+MY_OPT=", vals);

    // CHECK: n=1
    $display("n=%0d", n);
    if (n > 0)
      // CHECK: v0=hello
      $display("v0=%s", vals[0]);

    if (n == 1 && vals[0] == "hello")
      // CHECK: PASS
      $display("PASS");
    else
      $display("FAIL");
    // CHECK-NOT: FAIL
    $finish;
  end
endmodule
