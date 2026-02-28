// RUN: circt-verilog %s --ir-hw -o %t.mlir
// RUN: circt-sim %t.mlir --max-time=1000000000 2>&1 | FileCheck %s

// Regression: process::kill() must kill active descendants even if an
// intermediate child in the subtree has already finished.
module test_process_kill_orphan_descendant;
  process root_p;
  process mid_p;
  process leaf_p;
  event leaf_ready;

  initial begin
    fork
      begin
        root_p = process::self();
        fork
          begin
            mid_p = process::self();
            fork
              begin
                leaf_p = process::self();
                ->leaf_ready;
                forever #10;
              end
            join_none
            // Intentionally exit: this leaves leaf_p running with a finished
            // intermediate ancestor in the process tree.
          end
        join
        // Keep root alive so kill() traverses the subtree.
        forever #10;
      end
    join_none

    @leaf_ready;
    #1;

    // CHECK: mid status pre-kill: FINISHED
    if (mid_p.status() == process::FINISHED)
      $display("mid status pre-kill: FINISHED");
    else
      $display("mid status pre-kill: NOT-FINISHED");

    root_p.kill();
    #1;

    // CHECK: leaf status after root kill: KILLED
    if (leaf_p.status() == process::KILLED)
      $display("leaf status after root kill: KILLED");
    else
      $display("leaf status after root kill: NOT-KILLED");

    $finish;
  end
endmodule
