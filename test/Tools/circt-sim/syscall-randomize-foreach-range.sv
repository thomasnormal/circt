// RUN: circt-verilog %s --no-uvm-auto-include --ir-moore --ir-hw --ir-llhd -o %t.mlir 2>&1 && circt-sim %t.mlir --top top 2>&1 | FileCheck %s
// Test that foreach element range constraints are enforced during randomize().
// IEEE 1800-2017 Section 18.5.8: foreach constraints apply to each element.
class item_t;
  rand bit [7:0] arr[4];
  constraint c_all_one {
    foreach (arr[i]) arr[i] inside {[8'd1:8'd1]};
  }
endclass

module top;
  initial begin
    automatic item_t item = new();
    int ok;
    int all_one;

    ok = item.randomize();
    all_one = ((item.arr[0] == 8'd1) &&
               (item.arr[1] == 8'd1) &&
               (item.arr[2] == 8'd1) &&
               (item.arr[3] == 8'd1));

    // CHECK: randomize_ok=1
    $display("randomize_ok=%0d", ok);
    // CHECK: foreach_all_one=1
    $display("foreach_all_one=%0d", all_one);
    $finish;
  end
endmodule
