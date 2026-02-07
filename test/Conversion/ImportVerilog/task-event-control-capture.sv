// RUN: circt-verilog --no-uvm-auto-include %s --ir-hw 2>&1 | FileCheck %s
// Test that event controls inside tasks properly capture module-level signals.
// This ensures that @(posedge clk) inside a task correctly captures the
// module-level clock signal as a function capture argument.

// Previously, this would fail with:
//   error: 'moore.read' op using value defined outside the region
// because the rvalueReadCallback was set to nullptr during event control
// conversion, preventing the capture mechanism from working.

// CHECK-NOT: error:
// CHECK-NOT: 'moore.read' op using value defined outside the region

module task_event_control_capture;
  bit clk;
  bit data;
  int counter;

  // Clock generator
  initial begin
    clk = 1'b0;
    forever #10 clk = ~clk;
  end

  // Task with event control referencing module-level clock
  // The @(posedge clk) needs to capture the 'clk' signal reference
  task wait_and_set(input bit value);
    @(posedge clk);
    data = value;
  endtask

  // Task with multiple event controls
  task wait_two_cycles();
    @(posedge clk);
    @(posedge clk);
  endtask

  // Task with edge and condition in a loop
  task wait_with_counter(input int max_count);
    while (counter < max_count) begin
      @(posedge clk);
      counter = counter + 1;
    end
  endtask

  // Task with negedge
  task wait_negedge();
    @(negedge clk);
  endtask

  initial begin
    $display("Starting test");
    wait_and_set(1);
    wait_two_cycles();
    wait_with_counter(5);
    wait_negedge();
    $display("Test complete");
    $finish;
  end
endmodule

// CHECK: hw.module @task_event_control_capture
// CHECK: llhd.process
