// RUN: circt-verilog --ir-moore %s | FileCheck %s

// CHECK-LABEL: moore.module @UnpackedArrayAssignmentPatternOrder
module UnpackedArrayAssignmentPatternOrder;
  logic [7:0] A [8] = '{8'h00, 8'h01, 8'h02, 8'h03, 8'h04, 8'h05, 8'h06, 8'h07};
  logic [7:0] out;

  // CHECK-DAG: %[[C0:.*]] = moore.constant 0 : l8
  // CHECK-DAG: %[[C1:.*]] = moore.constant 1 : l8
  // CHECK-DAG: %[[C2:.*]] = moore.constant 2 : l8
  // CHECK-DAG: %[[C3:.*]] = moore.constant 3 : l8
  // CHECK-DAG: %[[C4:.*]] = moore.constant 4 : l8
  // CHECK-DAG: %[[C5:.*]] = moore.constant 5 : l8
  // CHECK-DAG: %[[C6:.*]] = moore.constant 6 : l8
  // CHECK-DAG: %[[C7:.*]] = moore.constant 7 : l8
  // CHECK: moore.array_create %[[C0]], %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C6]], %[[C7]]
  // CHECK: moore.extract {{.*}} from 7
  always_comb out = A[0];
endmodule
