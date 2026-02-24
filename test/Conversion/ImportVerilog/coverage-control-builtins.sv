// RUN: circt-translate --import-verilog %s | FileCheck %s
// REQUIRES: slang

module top;
  int unsigned i;
  real r;

  initial begin
    // CHECK: %[[C2:.*]] = moore.constant 2 : i32
    // CHECK: %[[C1:.*]] = moore.constant 1 : i32
    // CHECK: %[[CC:.*]] = moore.builtin.coverage_control %[[C2]], %[[C1]]
    i = $coverage_control(2, 1, 0, top);

    // CHECK: %[[CMAXTYPE:.*]] = moore.constant 1 : i32
    // CHECK: %[[CMAX:.*]] = moore.builtin.coverage_get_max %[[CMAXTYPE]]
    i = $coverage_get_max(1, 0, top);

    // CHECK: %[[CGET:.*]] = moore.builtin.get_coverage : f64
    r = $coverage_get(1, 0, top);
  end
endmodule
