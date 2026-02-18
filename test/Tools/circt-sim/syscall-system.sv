// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
module top;
  integer ret;

  initial begin
    // $system executes a shell command and returns exit code
    ret = $system("echo hello_from_system");
    // CHECK: hello_from_system
    // CHECK: exit_code=0
    $display("exit_code=%0d", ret);

    // Test with a command that returns non-zero
    ret = $system("false");
    // CHECK: exit_code_false=256
    $display("exit_code_false=%0d", ret);

    $finish;
  end
endmodule
