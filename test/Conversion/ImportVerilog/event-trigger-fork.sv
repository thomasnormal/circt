// RUN: circt-verilog %s --ir-moore | FileCheck %s --check-prefix=MOORE
// RUN: circt-verilog %s --ir-hw | FileCheck %s --check-prefix=HW

// Test that event triggers inside fork wake up processes waiting with @(event).
// This tests the fix for the bug where ->event created a temp alloca instead
// of driving the LLHD signal that @(event) waits on.

module test_event_trigger;
  event ev;
  int counter = 0;

  // MOORE-LABEL: moore.procedure initial
  // HW-LABEL: llhd.process
  initial begin
    // Fork a process that triggers the event after a delay
    fork
      begin
        #10;
        // MOORE: moore.event_trigger %{{.*}} : <event>
        ->ev;
      end
    join_none

    // Wait for the event to be triggered
    @(ev);
    counter = 1;

    // Verify we got woken up
    #1;
    if (counter != 1) begin
      $display("FAIL: counter = %d, expected 1", counter);
    end else begin
      $display("PASS: event trigger woke up waiter");
    end
  end
endmodule
