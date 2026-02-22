// RUN: circt-verilog %s --ir-hw -o %t.mlir 2>/dev/null
// RUN: circt-sim %t.mlir --top uvm_wait_for_nba_region_tb 2>&1 | FileCheck %s

// CHECK: PASS0: flag advanced to
// CHECK: PASS1: saw flag=2 after second wait
// CHECK-NOT: FAIL0
// CHECK-NOT: FAIL1
// CHECK: [circt-sim] Simulation completed

import uvm_pkg::*;

module uvm_wait_for_nba_region_tb;
  int flag;

  initial begin
    flag = 0;
    fork
      begin
        #0 flag = 1;
        #0 flag = 2;
      end
      begin
        uvm_wait_for_nba_region();
        if (flag != 0)
          $display("PASS0: flag advanced to %0d after first wait", flag);
        else
          $display("FAIL0: expected non-zero got %0d", flag);

        uvm_wait_for_nba_region();
        if (flag == 2)
          $display("PASS1: saw flag=2 after second wait");
        else
          $display("FAIL1: expected 2 got %0d", flag);
      end
    join

    $finish;
  end
endmodule
