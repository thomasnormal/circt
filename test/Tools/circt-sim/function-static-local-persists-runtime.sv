// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb 2>&1 | FileCheck %s
// Regression for issue #28: static locals in functions must persist across
// invocations (same as static locals in tasks).

module tb;
  function int fn_counter();
    static int cnt = 0;
    cnt++;
    return cnt;
  endfunction

  task task_counter(output int result);
    static int cnt = 0;
    cnt++;
    result = cnt;
  endtask

  int r;
  int fail = 0;

  initial begin
    if (fn_counter() != 1) fail++;
    if (fn_counter() != 2) fail++;
    if (fn_counter() != 3) fail++;

    task_counter(r); if (r != 1) fail++;
    task_counter(r); if (r != 2) fail++;
    task_counter(r); if (r != 3) fail++;

    if (fail == 0)
      $display("PASS");
    else
      $display("FAIL fail=%0d", fail);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
