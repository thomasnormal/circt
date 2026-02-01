// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=200000000000 2>&1 | FileCheck %s

// Test that multiple delays in class methods called from fork branches
// are scheduled correctly. This tests the fix for the bug where the second
// delay in a fork branch was not being scheduled because executeProcess
// returned early when a function suspended.

// CHECK: Starting fork-join at 0
// CHECK: Driver: Starting at 0
// CHECK: Monitor: Starting at 0
// CHECK-DAG: Monitor: After 30ns at 30
// CHECK-DAG: Driver: After 50ns at 50
// CHECK-DAG: Monitor: After 70ns total at 70
// CHECK-DAG: Driver: After 100ns total at 100
// CHECK-DAG: Monitor: After 130ns total at 130
// CHECK-DAG: Driver: After 150ns total at 150
// CHECK: Fork-join complete at 150

module fork_multiple_delays;
  class my_driver;
    task run();
      $display("Driver: Starting at %0t", $time);
      #50ns;
      $display("Driver: After 50ns at %0t", $time);
      #50ns;
      $display("Driver: After 100ns total at %0t", $time);
      #50ns;
      $display("Driver: After 150ns total at %0t", $time);
    endtask
  endclass

  class my_monitor;
    task run();
      $display("Monitor: Starting at %0t", $time);
      #30ns;
      $display("Monitor: After 30ns at %0t", $time);
      #40ns;
      $display("Monitor: After 70ns total at %0t", $time);
      #60ns;
      $display("Monitor: After 130ns total at %0t", $time);
    endtask
  endclass

  initial begin
    my_driver drv = new();
    my_monitor mon = new();

    $display("Starting fork-join at %0t", $time);
    fork
      drv.run();
      mon.run();
    join
    $display("Fork-join complete at %0t", $time);
    $finish;
  end
endmodule
