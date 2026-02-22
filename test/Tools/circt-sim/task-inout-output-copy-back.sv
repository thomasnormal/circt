// RUN: circt-verilog %s --no-uvm-auto-include --ir-llhd | circt-sim --top top | FileCheck %s

module top;
  int x;

  task inout_task(inout int a);
    a = 1;
    #1;
  endtask

  task output_task(output int a);
    a = 7;
    #1;
  endtask

  initial begin
    x = 0;
    fork
      inout_task(x);
    join_none
    #0;
    $display("MID_INOUT x=%0d", x);
    wait fork;
    $display("END_INOUT x=%0d", x);

    x = 0;
    fork
      output_task(x);
    join_none
    #0;
    $display("MID_OUTPUT x=%0d", x);
    wait fork;
    $display("END_OUTPUT x=%0d", x);

    $finish;
  end
endmodule

// CHECK: MID_INOUT x=0
// CHECK: END_INOUT x=1
// CHECK: MID_OUTPUT x=0
// CHECK: END_OUTPUT x=7
