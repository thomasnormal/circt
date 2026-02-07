// RUN: circt-verilog --no-uvm-auto-include --ir-hw %s | \
// RUN:   circt-bmc --emit-mlir -b 2 --module=sva_assert_final - | \
// RUN:   FileCheck %s --check-prefix=CHECK-BMC
// REQUIRES: slang

module sva_assert_final(input logic clk, input logic a);
  initial begin
    assert final (a);
  end
endmodule

// CHECK-BMC: func.func @sva_assert_final()
// CHECK-BMC: %[[LOOP:.*]]:3 = scf.for
// CHECK-BMC: %[[FINAL_OK:.*]] = smt.eq %[[LOOP]]#2, %{{.*}} : !smt.bv<1>
// CHECK-BMC: %[[FINAL_FAIL:.*]] = smt.not %[[FINAL_OK]]
// CHECK-BMC: smt.assert %[[FINAL_FAIL]]
