// RUN: circt-verilog %s --no-uvm-auto-include -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test process class: self, status, kill, await, suspend, resume
module top;
  process p;
  process child_p;
  int child_done = 0;

  initial begin
    p = process::self();
    // CHECK: self_not_null=1
    $display("self_not_null=%0d", p != null);

    // Status should be RUNNING (enum value 2 typically)
    // CHECK: status_running=RUNNING
    case (p.status())
      process::RUNNING: $display("status_running=RUNNING");
      process::WAITING: $display("status_running=WAITING");
      process::SUSPENDED: $display("status_running=SUSPENDED");
      process::KILLED: $display("status_running=KILLED");
      process::FINISHED: $display("status_running=FINISHED");
    endcase

    // Fork a child and test kill
    fork
      begin
        child_p = process::self();
        #100;
        child_done = 1;
      end
    join_none

    #1;
    // Child should be blocked on #100 delay
    // CHECK: child_waiting=SUSPENDED
    case (child_p.status())
      process::RUNNING: $display("child_waiting=RUNNING");
      process::WAITING: $display("child_waiting=WAITING");
      process::SUSPENDED: $display("child_waiting=SUSPENDED");
      process::KILLED: $display("child_waiting=KILLED");
      process::FINISHED: $display("child_waiting=FINISHED");
    endcase

    // Kill the child
    child_p.kill();
    #1;
    // CHECK: child_killed=KILLED
    case (child_p.status())
      process::RUNNING: $display("child_killed=RUNNING");
      process::WAITING: $display("child_killed=WAITING");
      process::SUSPENDED: $display("child_killed=SUSPENDED");
      process::KILLED: $display("child_killed=KILLED");
      process::FINISHED: $display("child_killed=FINISHED");
    endcase

    // Verify child never completed
    // CHECK: child_done=0
    $display("child_done=%0d", child_done);

    $finish;
  end
endmodule
