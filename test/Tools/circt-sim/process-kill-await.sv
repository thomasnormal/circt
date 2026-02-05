// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000000 2>&1 | FileCheck %s

// Test process::kill(), process::status(), and process::await().
// IEEE 1800-2017 Section 9.7 "Process control"

module test_process_kill_await;
  process child_p;
  event child_ready;

  initial begin
    fork
      begin
        child_p = process::self();
        ->child_ready;
        forever #10;
      end
    join_none

    @child_ready;
    child_p.kill();

    // CHECK: Child status after kill: KILLED
    if (child_p.status() == process::KILLED)
      $display("Child status after kill: KILLED");
    else
      $display("Child status after kill: NOT KILLED");

    child_p.await();
    // CHECK: Await returned
    $display("Await returned");
    $finish;
  end
endmodule
