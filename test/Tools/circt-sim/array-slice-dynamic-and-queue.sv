// RUN: circt-verilog %s --ir-llhd --timescale 1ns/1ns --single-unit -o %t.mlir
// RUN: circt-sim %t.mlir --top tb --max-time=10000000 2>&1 | FileCheck %s

// Regression for issue #52: slicing dynamic arrays and queues must preserve
// inclusive [lo:hi] semantics and element values.
module tb;
  int arr[] = '{10, 20, 30, 40, 50};
  int q[$] = '{10, 20, 30, 40, 50};
  int arrSlice[];
  int qSlice[$];

  initial begin
    arrSlice = arr[1:3];
    qSlice = q[1:3];
    if (arrSlice.size() == 3 && arrSlice[0] == 20 && arrSlice[1] == 30 &&
        arrSlice[2] == 40 && qSlice.size() == 3 && qSlice[0] == 20 &&
        qSlice[1] == 30 && qSlice[2] == 40)
      $display("PASS");
    else
      $display(
          "FAIL asz=%0d a0=%0d a1=%0d a2=%0d qsz=%0d q0=%0d q1=%0d q2=%0d",
          arrSlice.size(), arrSlice[0], arrSlice[1], arrSlice[2], qSlice.size(),
          qSlice[0], qSlice[1], qSlice[2]);
    $finish;
  end

  // CHECK: PASS
  // CHECK-NOT: FAIL
endmodule
